
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import io
import streamlit as st

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from typing import List, Union

def missing_value_imputation(df: pd.DataFrame, column: str, method: str = 'mean') -> pd.DataFrame:
    """
    Impute missing values while preserving the original data type.
    
    Args:
    df (pd.DataFrame): Input dataframe
    column (str): Column to impute
    method (str): Imputation method ('mean', 'median', 'mode', or 'constant')
    
    Returns:
    pd.DataFrame: Dataframe with imputed values
    """
    original_dtype = df[column].dtype
    
    if pd.api.types.is_numeric_dtype(original_dtype):
        if method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            imputer = SimpleImputer(strategy=method)
        
        df[column] = imputer.fit_transform(df[[column]])
        
        # Convert back to original dtype if it was integer
        if pd.api.types.is_integer_dtype(original_dtype):
            df[column] = df[column].round().astype(original_dtype)
    else:
        # For non-numeric types, use mode imputation
        df[column] = df[column].fillna(df[column].mode()[0])
    
    return df

def one_hot_encoding(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Perform one-hot encoding on a categorical column.
    If the column doesn't exist, check if it has already been encoded.
    """
    if column in df.columns:
        # The column exists, so we perform one-hot encoding
        return pd.get_dummies(df, columns=[column], prefix=column)
    else:
        # Check if the column has already been encoded
        encoded_columns = [col for col in df.columns if col.startswith(f"{column}_")]
        if encoded_columns:
            print(f"Column '{column}' has already been encoded into {len(encoded_columns)} categories.")
            return df
        else:
            print(f"Warning: Column '{column}' not found and no encoded columns detected. Skipping one-hot encoding for this column.")
            return df

def normalization(df: pd.DataFrame, column: str, method: str = 'min-max') -> pd.DataFrame:
    """
    Normalize a numeric column.
    
    Args:
    df (pd.DataFrame): Input dataframe
    column (str): Column to normalize
    method (str): Normalization method ('min-max' or 'z-score')
    
    Returns:
    pd.DataFrame: Dataframe with normalized column
    """
    if str(method).lower() == 'min-max' or str(method).replace('-', '').replace('_', '').lower() == 'minmax':
        scaler = MinMaxScaler()
    elif str(method).lower() == 'z-score' or str(method).replace('-', '').replace('_', '').lower() == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Use 'min-max' or 'z-score'.")
    
    df[column] = scaler.fit_transform(df[[column]])
    return df

def outlier_removal(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a numeric column.
    
    Args:
    df (pd.DataFrame): Input dataframe
    column (str): Column to remove outliers from
    method (str): Method to identify outliers ('iqr' or 'z-score')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if method.lower() == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method.lower() == 'z-score' or str(method).replace('-', '').replace('_', '').lower() == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores < threshold]
    else:
        raise ValueError("Invalid outlier removal method. Use 'iqr' or 'z-score'.")
    
    return df

def bin_numeric_column(df: pd.DataFrame, column: str, bins: int = 5, labels: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Bin a numeric column into categories.
    
    Args:
    df (pd.DataFrame): Input dataframe
    column (str): Column to bin
    bins (int): Number of bins
    labels (List[str] or None): Labels for the bins
    
    Returns:
    pd.DataFrame: Dataframe with binned column
    """
    df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)
    return df

def create_date_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Create date-related features from a date column.
    
    Args:
    df (pd.DataFrame): Input dataframe
    column (str): Date column to extract features from
    
    Returns:
    pd.DataFrame: Dataframe with new date-related features
    """
    df[column] = pd.to_datetime(df[column])
    df[f'{column}_year'] = df[column].dt.year
    df[f'{column}_month'] = df[column].dt.month
    df[f'{column}_day'] = df[column].dt.day
    df[f'{column}_dayofweek'] = df[column].dt.dayofweek
    df[f'{column}_quarter'] = df[column].dt.quarter
    return df

def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Remove features with low variance.
    
    Args:
    df (pd.DataFrame): Input dataframe
    threshold (float): Variance threshold
    
    Returns:
    pd.DataFrame: Dataframe with low variance features removed
    """
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    return df[df.columns[selector.get_support(indices=True)]]


def binary_encoding(df: pd.DataFrame, column: str, true_value: str = 'Yes', false_value: str = 'No') -> pd.DataFrame:
    """
    Perform binary encoding on a column with two categories.
    
    Args:
    df (pd.DataFrame): Input dataframe
    column (str): Column to encode
    true_value (str): Value to be encoded as 1 (default 'Yes')
    false_value (str): Value to be encoded as 0 (default 'No')
    
    Returns:
    pd.DataFrame: Dataframe with binary encoded column
    """
    if column in df.columns:
        df[column] = df[column].map({true_value: 1, false_value: 0})
        print(f"Binary encoded column '{column}' with {true_value}->1 and {false_value}->0")
    else:
        print(f"Warning: Column '{column}' not found. Skipping binary encoding for this column.")
    return df
