import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class FraudDataLoader:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = RobustScaler()
        
    def load_and_explore(self):
        print("Dataset Shape:", self.df.shape)
        print("\nFraud Rate:", self.df['Class'].value_counts(normalize=True))
        print("\nFirst 5 rows:")
        print(self.df.head())
        return self.df
    
    def preprocess(self, test_size=0.2, random_state=42):
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
    def apply_smote(self):
        smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        print(f"SMOTE Training set: {self.X_train_smote.shape}")
        print(f"Fraud balance after SMOTE: {np.bincount(self.y_train_smote)}")
