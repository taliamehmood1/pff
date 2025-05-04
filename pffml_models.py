import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    silhouette_score
)

def train_linear_regression(X_train, X_test, y_train, y_test, fit_intercept=True, normalize=False):
    """
    Train a linear regression model
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    y_train : numpy.ndarray
        Training target
    y_test : numpy.ndarray
        Testing target
    fit_intercept : bool
        Whether to fit an intercept
    normalize : bool
        Whether to normalize the features
        
    Returns:
    --------
    tuple
        (model, predictions, evaluation)
    """
    # Train the model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Store evaluation metrics
    evaluation = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
    
    return model, predictions, evaluation

def train_logistic_regression(X_train, X_test, y_train, y_test, C=1.0, max_iter=100):
    """
    Train a logistic regression model
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    y_train : numpy.ndarray
        Training target
    y_test : numpy.ndarray
        Testing target
    C : float
        Regularization strength
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    tuple
        (model, predictions, evaluation)
    """
    # Convert target to binary if not already
    y_train_binary = (y_train > np.median(y_train)).astype(int)
    y_test_binary = (y_test > np.median(y_test)).astype(int)
    
    # Train the model
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train_binary)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test_binary, predictions)
    precision = precision_score(y_test_binary, predictions, zero_division=0)
    recall = recall_score(y_test_binary, predictions, zero_division=0)
    cf_matrix = confusion_matrix(y_test_binary, predictions)
    
    # Store evaluation metrics
    evaluation = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cf_matrix
    }
    
    return model, predictions, evaluation

def train_kmeans_clustering(X_train, X_test, n_clusters=3, max_iter=300):
    """
    Train a K-Means clustering model
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    n_clusters : int
        Number of clusters
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    tuple
        (model, predictions, evaluation)
    """
    # Train the model
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
    model.fit(X_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    inertia = model.inertia_
    
    # Calculate silhouette score if more than one cluster
    if n_clusters > 1:
        silhouette = silhouette_score(X_test, predictions)
    else:
        silhouette = 0
    
    # Store evaluation metrics
    evaluation = {
        'inertia': inertia,
        'silhouette': silhouette
    }
    
    return model, predictions, evaluation
