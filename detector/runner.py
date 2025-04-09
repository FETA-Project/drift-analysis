import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from detector.detector import DriftDetector, Config
from detector.logger import Logger
from detector.analyser import LastWeekAnalyser

def vprint(string, verbose):
    if(verbose):
            print(string)

class ExperimentRunner:
    """
    Experiment runner to automatically split dataset into rounds of drift detection or simulate a model deployment over time.
    Args:
    config_list (list[ExperimentConfig]): 
        List of configurations of the various experiments
    """
    
    def __init__(self, config_list):
        self.config_list = config_list
        self.logger_list = []

    
    def run(self, verbose = True):
        """Perform the defined experiment.
        Args:
        verbose (bool): 
            Print results from the rounds of detection into the console.
        Returns:
            list[detector.Logger]:
                Loggers of the results of each experiment
        """       
        for config in self.config_list:
            self.__run_experiment(config, verbose)
        return self.logger_list

    def __init_detector(self, config, logger):
        imp = pd.Series(config.model.feature_importances_,index = config.model.feature_names_in_) if config.model else None
        global_config = Config(chosen_features = config.chosen_features,feature_importances = imp, drift_test = config.global_test)
        if(config.class_test):
            class_config = Config(chosen_features = config.chosen_features,feature_importances = imp, drift_test = config.class_test, 
                                  class_name = config.target_column)
        if(config.analyser_test):
            analyser_config = Config(chosen_features = config.chosen_features,feature_importances = imp, drift_test = config.analyser_test)
            return DriftDetector(global_config, class_config, logger = logger, analyser = LastWeekAnalyser(analyser_config))
        
        return DriftDetector(global_config, class_config, logger = logger)

    def __run_experiment(self, config, verbose):
        df = config.data
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        value_counts = df[config.target_column].value_counts()
        frequent_categories = value_counts[value_counts > 10 ].index
        df = df[df[config.target_column].isin(frequent_categories)]
       
        print(value_counts)
        # Fit and transform the labels
        df[config.target_column] = label_encoder.fit_transform(df[config.target_column])
        
        vprint(f"Running experiment [{config.experiment_name}]\n", verbose)

        # Check if data is sorted
        if(not df[config.index_column].is_monotonic_increasing):
            df = df.sort_values(by=config.index_column)
        
        #Get reference time window
        cutoff = df[config.index_column].iloc[0] + config.training_window * config.window_length
        df_ref = df[df[config.index_column] < cutoff]
        df = df[df[config.index_column] >= cutoff]

        #Train reference model if needed
        if(config.model):

            vprint("Training reference model", verbose)
            Xdata = df_ref[config.chosen_features]
            ydata = df_ref[config.target_column]
            X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.33, random_state=42, stratify=ydata.values)
            try:
                clf = config.model.fit(X_train, y_train)
            except Exception as e:
                print("Error: inconsistent traing data. Validate split of dataset.", e)
                sys.exit(1)
                
            y_pred = clf.predict(X_test)
            vprint(f"Training finished with validation F1 Score: \n{f1_score(y_test, y_pred, average = 'weighted')}", verbose)

        #Initialise detector
        logger = Logger(config.experiment_name)
        detector = self.__init_detector(config, logger)

        while(len(df) > 0):
            try:
                curr_time = df[config.index_column].iloc[0]
                vprint(f"\nRunning a round of detection with starting index of {curr_time}", verbose)
                cutoff = curr_time + config.window_length
                df_curr = df[df[config.index_column] < cutoff]
                df = df[df[config.index_column] >= cutoff]
                if(config.model):
                    Xdata = df_curr[config.chosen_features]
                    ydata = df_curr[config.target_column]
                    y_pred = clf.predict(Xdata)
    
                    
                    is_drifted = detector.detect(df_ref,df_curr,curr_time, y_pred)
                    vprint(detector.get_drift_statistics(), verbose)
    
                if is_drifted and config.retrain:
                    print("Drift detected, retraining")
    
                    # Update training dataset
                    df_ref = df_ref.tail(len(df_ref)-len(df_curr))
                    df_ref = pd.concat([df_ref,df_curr])
                    Xdata = df_ref[config.chosen_features]
                    ydata = df_ref[config.target_column]
                    clf = config.model.fit(Xdata, ydata)
            except Exception as e:
                print("Warning: Missing or badly formated dataset section", e)
        #y_pred_original = label_encoder.inverse_transform(y_pred)
        df[config.target_column] = label_encoder.inverse_transform(df[config.target_column])

        self.logger_list.append(logger)


   
class ExperimentConfig:
    """Configuration of the automated drift detection experiment.

    Args:
        data (pd.Dataframe):
            Dataset to run the experiment on
        chosen_features (list/pd.Series):
            Features concidered for the drift detection
        index_column (int/str):
            Name of the index column in the Dataframe to do the splitting inte rounds by (sequential or time indexes supported)
        window_length (int/timedelta):
            Amount of data concidered for one round of detection, either count of rows or length of time when using time indexes
        global_test (detector.Test):
            Test to be used for the global detection
        experiment_name (str):
            Name of the experiment, used for differentiating between experiments when comparing multiple ones 
        target_column (int/str, optional):
            Name of the column in the Dataframe corresponding to the target variable (of present)
        class_test (detector.Test, optional):
            Test to be used for the per class detection
        analyser_test (detector.Test, optional):
            Test to be used for the drift type analyser
        use_time_index (bool, optional):
            Switch between sequential and time based indexing
        chunk_length (int, optional):
            Count of how many rounds of detection should be aggregated into one chunk for the reporting usecase.
        model (ML models using scikit interface, optional):
            ML model used for simulated deployment.
        retrain (bool, optional):
            Retrain when drift is detected.
        training_window (int, optional):
            How many rounds of detection are used for the initial training of the model
    """

    def __init__(self, data, chosen_features, index_column, window_length , global_test, experiment_name,
                 target_column = None, class_test = None, analyser_test = None, use_time_index = True, 
                 chunk_length = None, model = None, retrain = False, training_window = None):
        self.data = data
        self.chosen_features = chosen_features
        self.index_column = index_column
        self.window_length = window_length
        self.global_test = global_test
        self.experiment_name = experiment_name
        self.target_column = target_column
        self.class_test = class_test
        self.analyser_test = analyser_test
        self.use_time_index = use_time_index
        self.chunk_length = chunk_length
        self.model = model
        self.retrain = retrain
        self.training_window = training_window
        if(target_column or  model or training_window):
            if(target_column is None or  model is None or training_window is None):
                raise Exception("Target_column, model and training_window have to be specified for automated model testing.")
