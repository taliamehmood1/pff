import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_split(train_size, test_size):
    """
    Visualize the train/test split
    
    Parameters:
    -----------
    train_size : int
        Number of training samples
    test_size : int
        Number of testing samples
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Pie chart visualization
    """
    fig = px.pie(
        values=[train_size, test_size],
        names=['Training Data', 'Testing Data'],
        title="Train/Test Split",
        color_discrete_sequence=['rgba(44, 160, 101, 0.85)', 'rgba(31, 119, 180, 0.85)'],
        hole=0.4
    )
    
    fig.update_traces(textinfo='percent+label', pull=[0, 0.1])
    
    return fig

def visualize_feature_importance(feature_importance):
    """
    Visualize feature importance
    
    Parameters:
    -----------
    feature_importance : pandas.DataFrame
        DataFrame with columns 'Feature' and 'Importance'
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart of feature importance
    """
    fig = px.bar(
        feature_importance.sort_values('Importance', ascending=True).tail(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 10 Feature Importance",
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig

def visualize_results(model_type, X_test, y_test, predictions, target_name="Target", feature_names=None):
    """
    Visualize model results
    
    Parameters:
    -----------
    model_type : str
        Type of model ('Linear Regression', 'Logistic Regression', 'K-Means Clustering')
    X_test : numpy.ndarray
        Testing features
    y_test : numpy.ndarray or None
        Testing target (None for clustering)
    predictions : numpy.ndarray
        Model predictions
    target_name : str
        Name of the target variable
    feature_names : list
        List of feature names
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization of the results
    """
    if model_type == "Linear Regression":
        # Create scatter plot of actual vs predicted values
        fig = px.scatter(
            x=y_test,
            y=predictions,
            labels={'x': f'Actual {target_name}', 'y': f'Predicted {target_name}'},
            title="Actual vs Predicted Values"
        )
        
        # Add identity line
        min_val = min(min(y_test), min(predictions))
        max_val = max(max(y_test), max(predictions))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Identity Line'
            )
        )
        
        return fig
    
    elif model_type == "Logistic Regression":
        # Create a figure with subplots
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Prediction Distribution", "Feature Correlation"))
        
        # Distribution of predictions
        fig.add_trace(
            go.Histogram(
                x=predictions,
                nbinsx=2,
                name="Predictions",
                marker_color='rgba(31, 119, 180, 0.7)'
            ),
            row=1, col=1
        )
        
        # Scatter plot of first two features colored by prediction
        if X_test.shape[1] >= 2:
            feature_names = feature_names if feature_names is not None else [f"Feature {i+1}" for i in range(X_test.shape[1])]
            
            scatter_data = pd.DataFrame({
                'x': X_test[:, 0],
                'y': X_test[:, 1],
                'prediction': predictions
            })
            
            fig.add_trace(
                go.Scatter(
                    x=scatter_data['x'],
                    y=scatter_data['y'],
                    mode='markers',
                    marker=dict(
                        color=scatter_data['prediction'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name="Data Points"
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text=feature_names[0], row=1, col=2)
            fig.update_yaxes(title_text=feature_names[1], row=1, col=2)
        
        fig.update_layout(
            title_text="Logistic Regression Results",
            height=500,
            width=900
        )
        
        return fig
    
    elif model_type == "K-Means Clustering":
        # If we have 2D data, we can visualize the clusters directly
        if X_test.shape[1] >= 2:
            feature_names = feature_names if feature_names is not None else [f"Feature {i+1}" for i in range(X_test.shape[1])]
            
            # Create DataFrame for plotting
            cluster_df = pd.DataFrame({
                'x': X_test[:, 0],
                'y': X_test[:, 1],
                'cluster': predictions
            })
            
            # Create scatter plot colored by cluster
            fig = px.scatter(
                cluster_df,
                x='x',
                y='y',
                color='cluster',
                color_continuous_scale='Viridis',
                title="K-Means Clustering Results",
                labels={'x': feature_names[0], 'y': feature_names[1], 'cluster': 'Cluster'}
            )
            
            return fig
        
        # For higher dimensional data, we can visualize the distribution of points in each cluster
        else:
            fig = px.histogram(
                predictions,
                title="Distribution of Clusters",
                nbins=len(np.unique(predictions)),
                labels={'value': 'Cluster', 'count': 'Number of Points'},
                color_discrete_sequence=['rgba(31, 119, 180, 0.7)']
            )
            
            return fig
