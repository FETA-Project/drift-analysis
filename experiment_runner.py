import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cesnet_datazoo.datasets import CESNET_TLS_Year22
from cesnet_datazoo.config import DatasetConfig, AppSelection
from datetime import datetime, timedelta


from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

from scipy import stats
from joblib import dump, load

import warnings
warnings.filterwarnings('ignore') 

from detector.detector import DriftDetector, Config
from detector.logger import Logger
from detector.test import KSTest, WassersteinTest
from detector.analyser import LastWeekAnalyser
from detector.runner import ExperimentRunner, ExperimentConfig
DATE = False

# Action: Change path to the input dataset
df = pd.read_csv("/home/dosoukup/katoda/cesnet-miners22-ppi/miners-design.csv")
# Action: Change label name and class names to convert label string to numbers
df["label"] = df.label.apply(lambda x: 0 if x == "Miner" else 1)

if DATE:

    df = df.sample(frac=1).reset_index(drop=True)
    df = df.reset_index()
    # Action: Change date identifier
    df['date'] = pd.to_datetime(df['date'])
    feat_names = df.columns[:-2]

    # Action: Validate experiment configuration variables (binar/mulitclass, label name, index name, experiment name). No need to change thresholds
    experiment_config = ExperimentConfig(
        data = df,
        chosen_features = feat_names,
        index_column = "date",
        window_length = timedelta(days=1),
        global_test = WassersteinTest(drift_threshold_global=0.04,drift_threshold_single = 0.1), 
        experiment_name = "Baseline",
        target_column = "APP",
        class_test = KSTest(drift_threshold_global=0.475,drift_threshold_single = 0.05), 
        analyser_test = None,
        use_time_index = True,
        chunk_length = 7,
        model = XGBClassifier(), #XGBClassifier(objective="multi:softmax"),
        retrain = False,
        training_window = 7
    )
else:
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.reset_index()
    feat_names = df.columns[:-1]

    experiment_config = ExperimentConfig(
        data = df,
        chosen_features = feat_names,
        index_column = "index",
        window_length = 10000,
        global_test = WassersteinTest(drift_threshold_global=0.04,drift_threshold_single = 0.1), 
        experiment_name = "Baseline",
        target_column = "label",
        class_test = KSTest(drift_threshold_global=0.475,drift_threshold_single = 0.05), 
        analyser_test = None,
        use_time_index = False,
        chunk_length = 7,
        model = XGBClassifier(), #XGBClassifier(objective="multi:softmax"),
        retrain = True,
        training_window = 7
    )

experimet_runner = ExperimentRunner([experiment_config])

logs = experimet_runner.run()

import pickle
# Action: Change name of the output file
with open('logs_miners_retrained.pkl', 'wb') as outp:
    pickle.dump(logs, outp, pickle.HIGHEST_PROTOCOL)

print("Katoda overview  data")
from detector.reporter import Reporter
reporter= Reporter(logs, ["Baseline"], chunk_length=7)
print("Drifted:",len(baseline_log["global_drift"][baseline_log["global_drift"]["is_drifted"] == True ]),"/ Not Drifted:",len(baseline_log["global_drift"][baseline_log["global_drift"]["is_drifted"] == False ]))
# TopN
print("classes", baseline_log["class_drift"].mean().sort_values().head(5))
print("features", baseline_log["feature_drift"].mean().sort_values(ascending=False).head(5))

