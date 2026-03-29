import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BiasDetector:
    """
    A tool for detecting and analyzing bias in machine learning models and datasets.
    It focuses on identifying disparities in model performance across different sensitive groups.
    """

    def __init__(self, sensitive_features, target_feature):
        """
        Initializes the BiasDetector.

        Args:
            sensitive_features (list): A list of column names in the DataFrame that represent sensitive attributes (e.g., ["gender", "race"]).
            target_feature (str): The name of the column representing the target variable (e.g., "loan_approved").
        """
        if not isinstance(sensitive_features, list) or not sensitive_features:
            raise ValueError("sensitive_features must be a non-empty list of column names.")
        if not isinstance(target_feature, str) or not target_feature:
            raise ValueError("target_feature must be a non-empty string.")

        self.sensitive_features = sensitive_features
        self.target_feature = target_feature
        self.label_encoders = {}
        logging.info(f"BiasDetector initialized with sensitive features: {sensitive_features} and target: {target_feature}.")

    def _preprocess_data(self, df):
        """
        Internal method to preprocess data, handling categorical sensitive features.
        """
        df_processed = df.copy()
        for feature in self.sensitive_features:
            if df_processed[feature].dtype == 'object' or df_processed[feature].dtype == 'category':
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature])
                else:
                    df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature])
        return df_processed

    def analyze(self, df, predictions=None):
        """
        Analyzes the dataset or model predictions for bias across sensitive features.

        Args:
            df (pd.DataFrame): The input DataFrame containing features and the target variable.
            predictions (pd.Series, optional): Model predictions for the target variable. If None,
                                               only dataset-level bias is analyzed (e.g., demographic parity).

        Returns:
            dict: A report containing various bias metrics.
        """
        logging.info("Starting bias analysis...")
        df_processed = self._preprocess_data(df)
        report = {}

        # Ensure target feature is numeric for metrics calculation
        y_true = df_processed[self.target_feature]
        if y_true.dtype == 'object' or y_true.dtype == 'category':
            if self.target_feature not in self.label_encoders:
                self.label_encoders[self.target_feature] = LabelEncoder()
                y_true = self.label_encoders[self.target_feature].fit_transform(y_true)
            else:
                y_true = self.label_encoders[self.target_feature].transform(y_true)

        if predictions is not None:
            # Model-level bias metrics (e.g., disparate impact, equal opportunity)
            y_pred = predictions
            if y_pred.dtype == 'object' or y_pred.dtype == 'category':
                if self.target_feature in self.label_encoders:
                    y_pred = self.label_encoders[self.target_feature].transform(y_pred)
                else:
                    logging.warning("Predictions are categorical but no encoder found for target. Assuming predictions match target encoding.")
                    y_pred = LabelEncoder().fit_transform(y_pred) # Fallback

            for feature in self.sensitive_features:
                groups = df_processed[feature].unique()
                if len(groups) < 2: # Need at least two groups to compare
                    logging.warning(f"Skipping bias analysis for {feature}: not enough unique groups.")
                    continue

                # Example: Demographic Parity Difference (DPD)
                # P(Y_pred=1 | A=group1) - P(Y_pred=1 | A=group2)
                positive_outcome_rates = []
                for group_val in groups:
                    group_mask = (df_processed[feature] == group_val)
                    if group_mask.sum() > 0:
                        positive_outcome_rate = y_pred[group_mask].mean()
                        positive_outcome_rates.append(positive_outcome_rate)
                        report[f"DPD_Positive_Outcome_Rate_{feature}_Group{group_val}"] = positive_outcome_rate

                if len(positive_outcome_rates) > 1:
                    dpd = max(positive_outcome_rates) - min(positive_outcome_rates)
                    report[f"Demographic_Parity_Difference_{feature}"] = dpd
                    logging.info(f"Demographic Parity Difference for {feature}: {dpd:.4f}")

                # Example: Equal Opportunity Difference (EOD)
                # P(Y_pred=1 | A=group1, Y_true=1) - P(Y_pred=1 | A=group2, Y_true=1)
                if 1 in y_true:
                    true_positive_rates = []
                    for group_val in groups:
                        group_and_positive_mask = (df_processed[feature] == group_val) & (y_true == 1)
                        if group_and_positive_mask.sum() > 0:
                            true_positive_rate = y_pred[group_and_positive_mask].mean()
                            true_positive_rates.append(true_positive_rate)
                            report[f"EOD_True_Positive_Rate_{feature}_Group{group_val}"] = true_positive_rate

                    if len(true_positive_rates) > 1:
                        eod = max(true_positive_rates) - min(true_positive_rates)
                        report[f"Equal_Opportunity_Difference_{feature}"] = eod
                        logging.info(f"Equal Opportunity Difference for {feature}: {eod:.4f}")

                # Accuracy Parity
                accuracy_scores = []
                for group_val in groups:
                    group_mask = (df_processed[feature] == group_val)
                    if group_mask.sum() > 0:
                        acc = accuracy_score(y_true[group_mask], y_pred[group_mask])
                        accuracy_scores.append(acc)
                        report[f"Accuracy_{feature}_Group{group_val}"] = acc
                if len(accuracy_scores) > 1:
                    acc_parity = max(accuracy_scores) - min(accuracy_scores)
                    report[f"Accuracy_Parity_Difference_{feature}"] = acc_parity
                    logging.info(f"Accuracy Parity Difference for {feature}: {acc_parity:.4f}")

        else:
            logging.info("No predictions provided. Analyzing dataset-level bias.")
            for feature in self.sensitive_features:
                groups = df_processed[feature].unique()
                if len(groups) < 2:
                    logging.warning(f"Skipping dataset bias analysis for {feature}: not enough unique groups.")
                    continue

                # Example: Target distribution parity
                target_distribution = []
                for group_val in groups:
                    group_mask = (df_processed[feature] == group_val)
                    if group_mask.sum() > 0:
                        target_mean = y_true[group_mask].mean()
                        target_distribution.append(target_mean)
                        report[f"Target_Mean_{feature}_Group{group_val}"] = target_mean

                if len(target_distribution) > 1:
                    target_parity_diff = max(target_distribution) - min(target_distribution)
                    report[f"Target_Distribution_Parity_Difference_{feature}"] = target_parity_diff
                    logging.info(f"Target Distribution Parity Difference for {feature}: {target_parity_diff:.4f}")

        logging.info("Bias analysis completed.")
        return report

if __name__ == "__main__":
    # Example Usage
    data = {
        'age': [25, 30, 35, 40, 28, 32, 38, 45, 29, 33, 36, 41, 27, 31, 39, 46],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'education': ['High', 'Uni', 'High', 'Uni', 'High', 'Uni', 'High', 'Uni', 'High', 'Uni', 'High', 'Uni', 'High', 'Uni', 'High', 'Uni'],
        'income': [50000, 60000, 55000, 70000, 52000, 62000, 58000, 75000, 51000, 61000, 56000, 71000, 53000, 63000, 59000, 76000],
        'loan_approved': [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)

    # Simulate model predictions (e.g., from a pre-trained model)
    # Let's assume the model is slightly biased against 'Female' for loan approval
    predictions = pd.Series([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])

    # Initialize BiasDetector
    detector = BiasDetector(sensitive_features=['gender', 'education'], target_feature='loan_approved')

    # Analyze dataset-level bias (without predictions)
    print("\n--- Dataset-level Bias Analysis ---")
    dataset_bias_report = detector.analyze(df)
    for metric, value in dataset_bias_report.items():
        print(f"- {metric}: {value:.4f}")

    # Analyze model-level bias (with predictions)
    print("\n--- Model-level Bias Analysis ---")
    model_bias_report = detector.analyze(df, predictions=predictions)
    for metric, value in model_bias_report.items():
        print(f"- {metric}: {value:.4f}")

    # Example with no bias (predictions match true labels)
    print("\n--- Model-level Bias Analysis (No Bias Example) ---")
    no_bias_predictions = df['loan_approved']
    no_bias_report = detector.analyze(df, predictions=no_bias_predictions)
    for metric, value in no_bias_report.items():
        print(f"- {metric}: {value:.4f}")

    logging.info("\nBiasDetector examples finished.")
