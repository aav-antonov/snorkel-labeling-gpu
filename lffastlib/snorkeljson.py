import json
import inspect
import numpy as np
import pandas as pd

from typing import Dict, List, Union
from snorkel.labeling import labeling_function, LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel


class SnorkelLFJsonConverter:
    """Convert between Snorkel LabelingFunctions and JSON rules."""

    def __init__(self):
        self.lfs = []  # Stores Snorkel LFs

    def json_to_lfs(self, json_rules: Union[str, Dict]) -> List[LabelingFunction]:
        """Convert JSON rules to Snorkel LabelingFunctions."""
        if isinstance(json_rules, str):

            with open(json_rules, 'r') as file:
                json_rules = json.load(file)
        elif isinstance(json_rules, dict):
            pass
        else:
            raise ValueError(
                "SnorkelLFJsonConverter Init ERROR: Input must be a JSON file path ending with '.json' or a dictionary.")

        self.lfs = []
        for rule_id, rule in json_rules.items():
            lf = self._create_lf_from_rule(rule)
            self.lfs.append(lf)
        return self.lfs

    def _create_lf_from_rule(self, rule: Dict) -> LabelingFunction:
        """Create a single LF from a rule dict."""
        features = rule["feature"]
        signs = rule["sign"]
        thresholds = rule["threshold"]
        lf_class = rule["class"]
        name = rule.get("id", f"lf_{len(self.lfs)}")

        @labeling_function(name=name)
        def lf(x):
            conditions_met = []
            for feat, sign, thresh in zip(features, signs, thresholds):
                value = x[feat]
                if sign == "<=":
                    conditions_met.append(value <= thresh)
                elif sign == ">":
                    conditions_met.append(value > thresh)
            return lf_class if all(conditions_met) else -1  # -1 = abstain

        return lf

    def lfs_to_json(self, lfs: List[LabelingFunction] = None) -> Dict:
        """Convert Snorkel LFs back to JSON rules."""
        lfs = lfs or self.lfs
        json_rules = {}
        for i, lf in enumerate(lfs):
            rule = self._extract_rule_from_lf(lf)
            json_rules[str(i)] = rule
        return json_rules

    def _extract_rule_from_lf(self, lf: LabelingFunction) -> Dict:
        """Extract rule conditions from an LF's source code."""
        source = inspect.getsource(lf)
        lines = [line.strip() for line in source.split("\n")]

        # Parse conditions (simplified; adjust regex for complex cases)
        conditions = []
        for line in lines:
            if "value" in line and ("<=" in line or ">" in line):
                conditions.append(line)

        # Extract features, signs, thresholds
        features, signs, thresholds = [], [], []
        for cond in conditions:
            parts = cond.split()
            feat = parts[parts.index("x[") + 1].replace("]", "").replace("'", "")
            sign = "<=" if "<=" in cond else ">"
            thresh = float(parts[-1])
            features.append(feat)
            signs.append(sign)
            thresholds.append(thresh)

        return {
            "id": lf.name,
            "class": 1,  # Default; adjust if multi-class
            "feature": features,
            "sign": signs,
            "threshold": thresholds
        }

    def apply_lfs_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply LFs to DataFrame and return labels."""
        applier = PandasLFApplier(lfs=self.lfs)
        L_train = applier.apply(df)

        # Compute counts directly from L_train (numpy array)
        results = np.column_stack([
            (L_train == 1).sum(axis=1),  # Counts of '1's
            (L_train == 0).sum(axis=1)  # Counts of '0's
        ])

        return results.flatten()