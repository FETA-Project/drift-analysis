import pandas as pd

class Logger:
    """
    Logger submodule for storing the results of each round of drift detection.
    """
    def __init__(self, description):
        self.description = description
        self.test_info = []
        self.index = []
        self.is_drifted = []
        self.drift_strength = []
        self.share_drifted_features = []
        self.drift_type = []
        self.f1 = []
        self.class_log = pd.DataFrame()
        self.f1_class_log = pd.DataFrame()
        self.class_shares = pd.DataFrame()
        self.feature_log = pd.DataFrame()

    def log(self,index, drift_statistics, f1_all = None, f1_mean = None, class_drift = None, class_shares = None, feature_drift = None):
        """Store the results from the current round of drift detection
        Args:
        date (Integer/Datetime/pd.Timestamp): 
            Time when the test was carried out, either integer index or timestamp is preferable if applicable
        drift_statistics (dictionary): 
            Statistics obtained from detector.get_drift_statistics() 
        f1_all (pandas.Series, optional): 
            Series of all F1 scores for each class
        f1_mean (Float, optional): 
            Weighted mean of class F1 scores
        """   
        self.index.append(index)
        self.is_drifted.append(drift_statistics["is_drifted"])
        self.drift_strength.append(drift_statistics["drift_strength"])
        self.share_drifted_features.append(drift_statistics["share_drifted_features"])
        if("drift_type" in drift_statistics):
            self.drift_type.append(drift_statistics["drift_type"])
        
        if(f1_mean is not None):
            self.f1.append(f1_mean)
        
        if(f1_all is not None):
            self.f1_class_log = pd.concat([self.f1_class_log, f1_all])

        if(class_drift is not None):
            self.class_log = pd.concat([self.class_log, class_drift])

        if(class_shares is not None):
            self.class_shares = pd.concat([self.class_shares, class_shares])
        
        if(feature_drift is not None):
            self.feature_log = pd.concat([self.feature_log, feature_drift])

    def set_test_info(self, test_info):
        self.test_info = test_info

    def get_logs(self):
        """Get the stored detection results
        Returns:
            dictionary: All the detection results and additional statistics from the experiment
                - global_drift: Global drift detection results
                - class_drift: Per class drift results
                - feature_drift: Per feature drift results
                - class_f1: Per class F1 scores
                - class_shares: Share of all the classes in the time window for drift detection
                - description: Description of the experiment
                - test_info: Definitions and descriptions of the tests used
                - overview: General overview of test results
        """   
        log = {}
        log_data = {
            "is_drifted": self.is_drifted,
            "drift_strength": self.drift_strength,
            "share_drifted_features": self.share_drifted_features
            }
        if(self.drift_type):
            log_data["Drift_type"] = self.drift_type 
        if(self.f1):
            log_data["f1"] = self.f1 
        
        log["global_drift"] = pd.DataFrame(data=log_data, index = self.index)
        if(not self.class_log.empty):
            log["class_drift"] = self.class_log.reindex(sorted(self.class_log.columns), axis=1).set_axis(self.index)
        
        if(not self.feature_log.empty):
            log["feature_drift"] = self.feature_log.reindex(sorted(self.feature_log.columns), axis=1).set_axis(self.index)
        
        if(not self.f1_class_log.empty):
            log["class_f1"] = self.f1_class_log.reindex(sorted(self.f1_class_log.columns), axis=1).set_axis(self.index)
            log["class_shares"] = self.class_shares.reindex(sorted(self.class_shares.columns), axis=1).set_axis(self.index)

        log["description"] = self.description
        
        if(self.test_info): 
            log["test_info"] = self.test_info

        log_overview = {
            "number_of_rounds": len(log["global_drift"].index),
            "first_round": log["global_drift"].index[0],
            "last_round": log["global_drift"].index[-1],
            "drift_detection_count": len(log["global_drift"][log["global_drift"].is_drifted]),
            "drift_strength_mean": log["global_drift"].drift_strength.mean(),
            "drift_strength_std": log["global_drift"].drift_strength.std(),
            "share_drifted_features_mean": log["global_drift"].share_drifted_features.mean(),
            "share_drifted_features_std": log["global_drift"].share_drifted_features.std(),
            }
        
        if(self.drift_type):
            log_overview["drift_type_values"] = log["global_drift"]["Drift_type"].value_counts()
        
        if(self.f1):
            log_overview["f1_mean"] = log["global_drift"].f1.mean()
            log_overview["f1_std"] = log["global_drift"].f1.std()
        
        log["overview"] = log_overview

        return log
