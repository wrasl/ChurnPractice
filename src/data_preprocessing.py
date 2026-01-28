import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Class for known dataset issue: 'TotalCharges' having empty strings instead of NaNs
class TotalChargesTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

        num_missing = X['TotalCharges'].isna().sum()
        logging.info(f"Missing TotalCharges after coercion: {num_missing}")

        return X

# Define column categories and target
NUMERIC_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']

BINARY_COLUMNS = ['SeniorCitizen', 'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

MULTI_COLUMNS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]

# --------------   TARGET = 'Churn'

# Define preprocessing pipelines for column categories
# SimpleImputer handles missing values, OneHotEncoder encodes categorical variables to binary (needed for numerical algorithms)
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

binary_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='if_binary'))
])

multi_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing pipeline combining all column-specific pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, NUMERIC_COLUMNS),
    ('bin', binary_pipeline, BINARY_COLUMNS),
    ('multi', multi_pipeline, MULTI_COLUMNS)
])

# Function that builds the complete preprocessing pipeline
# Called in train.py before model training
def build_preprocessing_pipeline():

    pipeline = Pipeline(steps=[
        ('total_charges_transformer', TotalChargesTransformer()),
        ('preprocessor', preprocessor)
    ])

    return pipeline
