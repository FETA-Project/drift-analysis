import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from detector import logger, analyser

class Reporter:
    """
    Reporter to generate various statistics and views of the experiment results stored in logs.

    Args:
    loggers (list[detector.Logger]):
        Loggers of the results of each experiment
    experiment_names (list[str]):
        Names of the experiments corresponding to the loggers  
    chunk_length (int, optional):
        Count of how many rounds of detection should be aggregated into one chunk for chunk by chunk analysis.  
    """
    
    def __init__(self, loggers, experiment_names, chunk_length = 7):
        self.loggers = loggers
        self.experiment_names = experiment_names
        self.chunk_length = chunk_length
        self.log_index = loggers[0].get_logs()["global_drift"].index
        self.global_features = ["is_drifted","drift_strength","share_drifted_features","f1"]

        for logger in loggers:
            if(not (logger.get_logs()["global_drift"].index.equals(self.log_index))):
                raise Exception("The indexes of all experiments have to be the same") 

    def __calculate_main_results(self, logs, index, index_name):
        results = pd.DataFrame()
        for log in logs:
            log = log[self.global_features]
            results = pd.concat([results, log.mean().to_frame().T])
        
        results[index_name] = index
        results = results.set_index(index_name)
        results = results.rename(columns={"is_drifted": "Ratio_of_drift_detections", "drift_strength": "Mean_drift_strength", 
                                                        "share_drifted_features": "Mean_ratio_of_drifted_features", "f1": "Mean_f1_score"})
        return results
   

    def get_global_results(self):
        """Get the broad overview of the experiment information and detected drifts.
        """
        global_values = ['number_of_rounds','first_round','last_round']
        compared_values = ["drift_detection_count","drift_strength_mean","drift_strength_std","share_drifted_features_mean",
                           "share_drifted_features_std", "f1_mean", "f1_std"]
        global_overview = {k: v for k, v in self.loggers[0].get_logs()["overview"].items() if k in global_values}
        
        comparison = pd.DataFrame()
        for logger in self.loggers:
            log_over = logger.get_logs()["overview"]
            comparison = pd.concat([comparison, pd.DataFrame.from_dict({k: [v] for k, v in log_over.items() if k in compared_values}, orient='columns')])

        comparison["experiment"] = self.experiment_names
        comparison = comparison.set_index("experiment")

        return{"overview": global_overview,"comparison": comparison}

    def print_experiment_overview(self):
        """Print the borad overview of the experiment information, detected drifts, tests used etc.
        """
        overview = self.get_global_results()["overview"]
        print(f"{len(self.loggers)} experiments were performed, each with {overview['number_of_rounds']} rounds of detection,")
        try:
            print(f"done between {overview['first_round'].strftime('%Y-%m-%d %X')} and {overview['last_round'].strftime('%Y-%m-%d %X')}")
        except Exception as e:
            print("Experiment was generated based on fixed samples window")

        display(self.get_global_results()["comparison"])

        print("\nThe experiments and their tests were defined as follows:")
        for id, logger in enumerate(self.loggers):
            print(f"Experiment {id}: {self.experiment_names[id]}")
            log = logger.get_logs()
            print(log["description"])
            print(log["test_info"])



    def plot_global_results(self, severity_style, f1_style, detection_style, detection_threshold = None):
        """Plot the graph of f1 scores and drift severities between the experiments.  
        
        Args:
            severity_style (list of dictionaries of pyplot line parameters):
                Visual styles for the lines representing the drift severities in each experiment
            f1_style (list of dictionaries of pyplot line parameters):
                Visual styles for the lines representing the f1 scores in each experiment
            detection_style (list of dictionaries of pyplot line parameters):
                Visual styles for the lines representing the drift detections in each experiment
            detection_threshold (float, optional):
                Add the drift detection threshold line at specified drift severity
        """  
        fig, ax = plt.subplots(figsize=(12, 5))
        ax2 = ax.twinx()

        for id, logger in enumerate(self.loggers):
            logs = logger.get_logs()["global_drift"]
            ax2.plot(logs.drift_strength,linestyle = severity_style[id]["line"], color = severity_style[id]["color"], 
                     alpha = severity_style[id]["alpha"], label =  f"{self.experiment_names[id]} severity")

            ax.plot(logs.f1, linestyle = f1_style[id]["line"], color = f1_style[id]["color"], 
                    alpha = f1_style[id]["alpha"], label =  f"{self.experiment_names[id]} F1 Score")

            [ ax.axvline(x = detection, linestyle = detection_style[id]["line"], color = detection_style[id]["color"], alpha = detection_style[id]["alpha"],
                         label = f"{self.experiment_names[id]} Drift detection") for detection in logs[logs.is_drifted].index ]

        if detection_threshold: 
            ax2.axhline(y = detection_threshold, color = 'g', linestyle = '--', label = "Drift detection threshold") 

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        handles, labels = ax2.get_legend_handles_labels()
        by_label = by_label | dict(zip(labels, handles))

        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.9])
        ax.grid(linestyle=':')
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            ncol=3, 
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("F1 Score of the model")
        ax2.set_ylabel("Detected drift severity")

        return fig

    def plot_analysis_results(self,detection_threshold = None):
        """For each experiment, plot the results of the drift type analysis. 
            
            Args:
                detection_threshold (float, optional):
                    Add the drift detection threshold line at specified drift severity
        """  
        for id, logger in enumerate(self.loggers):
            print(f"Analysis results of {self.experiment_names[id]} experiment:")
            logs = logger.get_logs()["global_drift"]
            print(logs.Drift_type.value_counts())

            fig = plt.figure(figsize = (12,4))
            plt.rc('font', size=12)
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            ax.plot(logs.drift_strength, "b-", alpha = 0.7, label = "Drift severity")
            ax2.plot(logs.f1, "r-", label = "F1 score of the model")
            if(detection_threshold):
                ax.axhline(y = detection_threshold, color = 'g', linestyle = '--', label = "Drift detection threshold") 

            drift_types = list(logs.Drift_type.value_counts().index)
            cmap = plt.cm.get_cmap('Accent', len(drift_types))
            for id, drift_type in enumerate(drift_types):
                for date in logs[logs.Drift_type == drift_type].index :
                    ax2.axvline(x=date, color = cmap(id), linestyle = ':',label = drift_type)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            handles, labels = ax2.get_legend_handles_labels()
            by_label = by_label | dict(zip(labels, handles))

            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.9])
            ax.grid(linestyle=':')
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc='upper center', 
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                ncol=3, 
            )

            ax.set_ylabel("Drift severity")
            ax.set_xlabel("Date")
            ax.grid(linestyle=':')
            plt.show()

    def get_chunk_results(self, sort_by = "Chunk", ascending = True):
        """Aggregate multiple rounds of detection into a chunk and compare the measured statistics. 
        
        Args:
            sort_by (str, optional):
                Name of the column used for sorting the results by
            ascending (bool, optional):
                Sort in the ascending or descending order
        """  
        results = []
        for logger in self.loggers:
            log = logger.get_logs()["global_drift"]
            chunks = []
            for i in range(0, len(log), self.chunk_length):
                chunk = log.iloc[i:i+self.chunk_length]
                chunks.append(chunk)
            
            results.append(self.__calculate_main_results(chunks, [i for i in range(math.ceil(len(log)/self.chunk_length))], "Chunk"))

        return [result.sort_values(by=sort_by, ascending = ascending) for result in results]

    def analyse_feature_drift(self,n_features):
        """Show the most drifted features for each experiment and plot them
        Args:
                n_features (int):
                    Number of most drifted features to show
        """
        print("Most drifted features:")
        for id, logger in enumerate(self.loggers):
            print(f"\nExperiment {self.experiment_names[id]}")
            most_drifted = logger.get_logs()["feature_drift"].mean().sort_values(ascending=False).head(n_features)
            print(most_drifted)

        feature_to_plot = most_drifted.index
        print("\n Compare drifts between experiments:")
        self.plot_feature(feature_to_plot)


    def plot_feature(self, features_to_plot):
        """ Plot the chosen features for each experiment
            
            Args:
                features_to_plot (list[str]):
                    Names of the features to plot
        """ 
        for id, logger in enumerate(self.loggers):
            plt.figure(figsize=(8,5))
            log = logger.get_logs()["feature_drift"]
            for f in features_to_plot:
                plt.plot(log[f],alpha = 0.8, label = f)
            log = logger.get_logs()["global_drift"]
            detected_drifts = log[log.is_drifted].index
            [plt.axvline(x = detection, linestyle = "dotted", color = "black", alpha = 0.5) for detection in detected_drifts]

            plt.legend()
            plt.title(f"Comparison of the most drifted features - experiment {self.experiment_names[id]}")
            plt.xlabel("Date")
            plt.ylabel("Feature drift strength")
            plt.show()
    
    def plot_feature_comparison(self, chosen):
        """ Plot the single chosen feature over all experiments in a single graph
            
            Args:
                chosen (str):
                    Name of the feature to plot
        """ 
        log = self.loggers[0].get_logs()["global_drift"]
        detected_drifts = log[log.is_drifted].index
        
        plt.figure(figsize=(6,4))
        for id, logger in enumerate(self.loggers):
            plt.plot(logger.get_logs()["feature_drift"][chosen],alpha = 0.8, label = f"Experiment - {self.experiment_names[id]}")
        [plt.axvline(x = detection, linestyle = "dotted", color = "black", alpha = 0.5) for detection in detected_drifts]    
        plt.legend()
        plt.title(f"Feature {chosen} drift strength comparison between experiments")
        plt.xlabel("Date")
        plt.ylabel("Feature drift strength")
        plt.show()
    
    def analyse_class_shares(self,n_classes):
        """Show the most represented and classes and their deviations. Compare their F1 scores. 
        
        Args:
            n_classes (int):
                Number of most represented classes to plot.
        """  

        class_shares = self.loggers[0].get_logs()["class_shares"]
        print("Class shares mean:")
        print(class_shares.mean().sort_values(ascending=False).head(n_classes))
        print("\nClass shares deviation:")
        print(class_shares.std().sort_values(ascending=False).head(n_classes))

        classes_to_plot = class_shares.mean().sort_values(ascending=False).head(n_classes).index

        plt.figure(figsize=(8,5))
        for c in classes_to_plot:
            plt.plot(class_shares[c],alpha = 0.8, label = f"Class_{c}")
        plt.legend()
        plt.title("Comparison of class shares of the most common classes")
        plt.xlabel("Date")
        plt.ylabel("Class share")
        plt.show()

        for id, logger in enumerate(self.loggers):
            plt.figure(figsize=(8,5))
            class_f1 = logger.get_logs()["class_f1"]
            for c in classes_to_plot:
                plt.plot(class_f1[c],alpha = 0.8, label = f"Class_{c}")
            plt.legend()
            plt.title(f"Comparison of F1 scores of the most common classes - experiment {self.experiment_names[id]}")
            plt.xlabel("Date")
            plt.ylabel("F1 score")
            plt.show()

    def analyse_class_drift(self,n_classes):
        """Show the most drifted classes and plot their F1 scores. 
        
        Args:
            n_classes (int):
                Number of most drifted classes to show.
        """  

        print("Most drifted classes:")
        for id, logger in enumerate(self.loggers):
            print(f"\nExperiment {self.experiment_names[id]}")
            most_drifted = logger.get_logs()["class_drift"].mean().sort_values().head(n_classes)
            print(most_drifted)

        classes_to_plot = most_drifted.index
        print("\n Compare F1 scores between experiments:")
        self.plot_class("class_f1",classes_to_plot)
        
        print("Compare drifts between experiments:")
        self.plot_class("class_drift",classes_to_plot)

        print("Class pairs with most correlated F1 scores")
        corr_matrix = (self.loggers[0].get_logs()["class_f1"].corr().abs())
        corr_pairs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                          .stack().sort_values(ascending=False))
        print(corr_pairs.head(10))

        print("\nClass pairs with most correlated drifts")
        corr_matrix = (self.loggers[0].get_logs()["class_drift"].corr().abs())
        corr_pairs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                          .stack().sort_values(ascending=False))
        print(corr_pairs.head(10))


    def plot_class(self, log_type, classes_to_plot):
        """ Plot the chosen classes for each experiment
            
            Args:
                classes_to_plot (list[str]):
                    Names of the classes to plot
        """  
        if(log_type not in ["class_f1","class_drift"]):
            raise Exception("'class_f1' and 'class_drift' are the only valid log_types")

        for id, logger in enumerate(self.loggers):
            plt.figure(figsize=(8,5))
            log = logger.get_logs()[log_type]
            for c in classes_to_plot:
                plt.plot(log[c],alpha = 0.8, label = f"Class_{c}")
            log = logger.get_logs()["global_drift"]
            detected_drifts = log[log.is_drifted].index
            [plt.axvline(x = detection, linestyle = "dotted", color = "black", alpha = 0.5) for detection in detected_drifts]

            if log_type == "class_f1":
                plt.title(f"Comparison of class F1 scores  - experiment {self.experiment_names[id]}")
                plt.ylabel("F1 Score")
            if log_type == "class_drift":
                plt.title(f"Comparison of class drift severity scores  - experiment {self.experiment_names[id]}")
                plt.ylabel("Class drift")
            plt.legend()
            #plt.title(f"Comparison of class F1 scores  - experiment {self.experiment_names[id]}")
            plt.xlabel("Date")
            #plt.ylabel("Class drift")
            plt.show()

    def plot_class_comparison(self, chosen):
        """ Plot the single chosen class over all experiments in a single graph
            
            Args:
                chosen (str):
                    Name of the class to plot
        """  
        log = self.loggers[0].get_logs()["global_drift"]
        detected_drifts = log[log.is_drifted].index
        
        plt.figure(figsize=(6,4))
        for id, logger in enumerate(self.loggers):
            plt.plot(logger.get_logs()["class_f1"][chosen],alpha = 0.8, label = f"Experiment - {self.experiment_names[id]}")
        [plt.axvline(x = detection, linestyle = "dotted", color = "black", alpha = 0.5) for detection in detected_drifts]    
        plt.legend()
        plt.title(f"Class {chosen} F1 score comparison between experiments")
        plt.xlabel("Date")
        plt.ylabel("Class F1 score")
        plt.show()

        plt.figure(figsize=(6,4))
        for id, logger in enumerate(self.loggers):
            plt.plot(logger.get_logs()["class_drift"][chosen],alpha = 0.8, label = f"Experiment - {self.experiment_names[id]}")
        [plt.axvline(x = detection, linestyle = "dotted", color = "black", alpha = 0.5) for detection in detected_drifts]    
        plt.legend()
        plt.title(f"Class {chosen} drift strength comparison between experiments")
        plt.xlabel("Date")
        plt.ylabel("Class drift strength")
        plt.show()
        


