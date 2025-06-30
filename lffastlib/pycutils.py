import numpy as np
import json
import sys

from lffastlib.timer import time_clock

class CustomerFeatures:

    def __init__(self, df, features = None, column_id = None, column_label = None):
        
        if features is None:
            print("ERROR:  CustomerFeatures: you need to provide features for Method: get_from_df")
            exit()

        if column_label is not None:
            self.labels = df[[column_label]].to_numpy()
        
        if column_id is not None:
            self.ids = df[[column_id]].to_numpy()
        
        # Ensure all required columns are in the DataFrame
        missing_columns = set(features) - set(df.columns)
        if missing_columns:
            raise ValueError(f"The DataFrame is missing the following columns: {missing_columns}")
        
              
        # Reorder columns according to the order in features_all_id
        df = df[features]
        self.features = list(df.columns)
        
        self.customer_number, self.features_number = df.shape
        self.customers_features = df.to_numpy()
        

        self.predicted_labels = np.full(len(df), -33, dtype=np.int32)
        
    
    def set_predicted_labels(self,  predicted_labels):
        self.predicted_labels = predicted_labels

        
class LabelingFunctions:

    @time_clock
    def __init__(self, input_data):
        if isinstance(input_data, str) and input_data.endswith('.json'):
            self.json = input_data
            with open(self.json, 'r') as file:
                data = json.load(file)
        elif isinstance(input_data, dict):
            data = input_data
        else:
            raise ValueError("LabelingFunctions Init ERROR: Input must be a JSON file path ending with '.json' or a dictionary.")
            
        self.lf_number = len(data)
        features_all = set()
        
        self.lf_features_shape = np.zeros(len(data), dtype=int)
        self.lf_outcome = np.zeros(len(data), dtype=int)
        self.lf_id = np.zeros(len(data), dtype=str)
        self.lf_features_size = 0
        
        for i, lf in enumerate(data):
            self.lf_id[i] = data[lf]["id"]
            self.lf_outcome[i] = data[lf]["class"]
            features = data[lf]["feature"]
            self.lf_features_shape[i] = len(features)
            features_all.update(features)            

        self.lf_features_size = np.sum(self.lf_features_shape)
        
        self.lf_features = np.zeros(self.lf_features_size, dtype=int)
        self.lf_features_thresholds = np.zeros(self.lf_features_size, dtype=float)
        self.lf_features_compare = np.zeros(self.lf_features_size, dtype=int)
        
        self.lf_features_compare.fill(-1)
        self.lf_features_thresholds.fill(-sys.float_info.min)
        self.lf_features.fill(-1)
        

        self.features_all_id = {element: index for index, element in enumerate(sorted(features_all))}
        
        self.features = [element for index, element in enumerate(sorted(features_all))]

        
        f_count = 0
        for i, lf in enumerate(data):
            
            features = data[lf]["feature"]
            f_ids = [self.features_all_id[f] for f in features]
            compare = data[lf]["sign"]
            compare_id = [1 if op == "<=" else 0 for op in compare]
            thresholds = data[lf]["threshold"]
            
            self.lf_features[f_count: f_count+len(features)] = f_ids
            self.lf_features_thresholds[f_count: f_count+len(features)] = thresholds
            self.lf_features_compare[f_count: f_count+len(features)] = compare_id
            
            f_count += len(features)
        
        if np.any(self.lf_features < 0):
            print("ERROR from_json: negative values in self.lf_features")
            exit(1)
        
        if np.any(self.lf_features_thresholds == -sys.float_info.min):
            print("ERROR from_json: negative values in self.lf_features_thresholds")
            exit(1)
            
        if np.any(self.lf_features_compare < 0):
            print("ERROR from_json: negative values in self.lf_features_compare")
            exit(1)       
        
    def to_dict(self):

        self.lf_config = {}
        f_count = 0

        for i in range(self.lf_number):
            lf_data = {
                "id": self.lf_id[i] ,
                "class": int(self.lf_outcome[i]),
                "feature": [],
                "sign": [],
                "threshold": []
            }

            for j in range(self.lf_features_shape[i]):
                feature_id = self.lf_features[f_count]
                lf_data["feature"].append(self.features[feature_id])
                
                compare_id = self.lf_features_compare[f_count]
                lf_data["sign"].append("<=" if compare_id == 1 else ">")
                
                threshold_value = self.lf_features_thresholds[f_count]
                lf_data["threshold"].append(float(threshold_value))
                
                f_count += 1

            self.lf_config[str(i)] = lf_data
            
    def to_json(self, file_path):
        self.to_dict()
        with open(file_path, 'w') as file:
            json.dump(self.lf_config, file, indent=4)



