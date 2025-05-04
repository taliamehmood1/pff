import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

def preprocess_data(data, handle_missing='Fill with mean/mode', handle_outliers='None'):
    """
    Preprocess the financial data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to preprocess
    handle_missing : str
        Method to handle missing values ('Drop rows', 'Fill with mean/mode', 'None')
    handle_outliers : str
        Method to handle outliers ('Clip outliers', 'None')
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    """
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Handle missing values
    if handle_missing == 'Drop rows':
        processed_data = processed_data.dropna()
    elif handle_missing == 'Fill with mean/mode':
        numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric columns with mean
        for col in numeric_cols:
            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            mode_value = processed_data[col].mode()[0] if not processed_data[col].mode().empty else "Unknown"
            processed_data[col] = processed_data[col].fillna(mode_value)
    
    # Handle outliers - only for numeric columns
    if handle_outliers == 'Clip outliers':
        numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            q1 = processed_data[col].quantile(0.25)
            q3 = processed_data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            processed_data[col] = np.clip(processed_data[col], lower_bound, upper_bound)
    
    return processed_data

def engineer_features(data, feature_cols, target_col, scale_features=True, add_polynomial=False, poly_degree=2):
    """
    Perform feature engineering on the data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed data
    feature_cols : list
        List of column names to use as features
    target_col : str
        Column name to use as target
    scale_features : bool
        Whether to scale the features
    add_polynomial : bool
        Whether to add polynomial features
    poly_degree : int
        Degree of polynomial features
    
    Returns:
    --------
    tuple
        (X, y, feature_names)
    """
    # Extract features and target
    X = data[feature_cols].values
    y = data[target_col].values
    feature_names = feature_cols.copy()
    
    # Add polynomial features if requested
    if add_polynomial:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X = poly.fit_transform(X)
        
        # Update feature names
        poly_features = []
        for i in range(len(feature_cols)):
            poly_features.append(feature_cols[i])
            
        for i in range(len(feature_cols)):
            for j in range(i, len(feature_cols)):
                poly_features.append(f"{feature_cols[i]}*{feature_cols[j]}")
                
        feature_names = poly_features[:X.shape[1]]
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, feature_names

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    test_size : float
        Proportion of the data to include in the test split
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
