import numpy as np
import pandas as pd
import random
import json
from typing import Dict, List, Tuple

from lffastlib.timer import time_clock

class GeneratorLF:
    def __init__(self, **kwargs):
        """
        Initialize the GeneratorLF with parameters for customer features and label functions.

        Keyword Args:
            customer_number: Number of customers to generate (default: 1000000)
            features_number: Number of features per customer (default: 300)
            lf_number: Number of label functions to generate (default: 100000)
            lf_size_range: Range (min, max) for size of each label function (default: [5, 10])
        """
        self.customer_number = kwargs.get('customer_number', 1000000)
        self.features_number = kwargs.get('features_number', 300)
        self.lf_number = kwargs.get('lf_number', 100000)
        self.lf_size_range = kwargs.get('lf_size_range', [5, 10])



        self.generate_label_functions()
        self.generate_customer_features()

    def save(self, filename):
        # Save the best (optimalthresholds) config
        with open(f"{filename}.json", 'w') as f:
            json.dump(self.lf_json, f, indent=4)

        self.df.to_csv(f'{filename}.csv', index=False)

    @time_clock
    def generate_customer_features(self) -> pd.DataFrame:
        """
        Generate customer features matrix with random values in range [0, 100)

        Returns:
            DataFrame with customer features (rows: customers, columns: features)
        """
        # Generate random values between 0 and 100
        features = np.random.rand(self.customer_number, self.features_number) * 100

        # Create column names (f0, f1, ...)
        columns = [f"f{i}" for i in range(self.features_number)]


        self.df = pd.DataFrame(features, columns=columns).assign(id=[f"cust{j}" for j in range(self.customer_number)])

    @time_clock
    def generate_label_functions(self) -> Tuple[Dict, int]:
        """
        Generate label functions with random parameters

        Returns:
            Tuple containing:
                - Dictionary of label functions
                - Count of label functions generated
        """
        self.lf_json = {}
        count_lf = 0

        for i in range(self.lf_number):
            # Determine size of this label function
            lf_size = random.randint(self.lf_size_range[0], self.lf_size_range[1])
            lf_id = f"lf_Q{i}"

            # Randomly select features to use in this LF
            feature_lf_i = random.sample(range(self.features_number), lf_size)


            feature_lf_i = [f"f{i}" for i in feature_lf_i]
            # Generate random thresholds (0-100) for each feature
            threshold_lf_i = [random.uniform(0, 100) for _ in range(lf_size)]

            # Randomly assign signs (1 or -1) for each feature
            sign_lf_i = [random.choice(["<=", ">"]) for _ in range(lf_size)]

            # Randomly assign LF type (you can modify this as needed)
            lf_type = random.choice([1, 0])

            # Create LF parameters dictionary
            lf_param = {
                'id': lf_id,
                'class': lf_type,
                'feature': feature_lf_i,
                'sign': sign_lf_i,
                'threshold': threshold_lf_i
            }

            self.lf_json[count_lf] = lf_param
            count_lf += 1



    def generate_all(self) -> Tuple[pd.DataFrame, Dict, int]:
        """
        Generate both customer features and label functions

        Returns:
            Tuple containing:
                - Customer features DataFrame
                - Label functions dictionary
                - Count of label functions
        """
        features = self.generate_customer_features()
        lfs, count = self.generate_label_functions()
        return features, lfs, count