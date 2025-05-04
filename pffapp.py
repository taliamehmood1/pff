import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from ml_models import train_linear_regression, train_logistic_regression, train_kmeans_clustering
from data_processing import preprocess_data, split_data, engineer_features
from visualization import visualize_results, visualize_feature_importance, visualize_split
from utils import download_results, format_stock_data, load_sample_data

# Configure the page
st.set_page_config(
    page_title="Financial Data Analysis - AF3005",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = None

# Welcome Page
def welcome_page():
    st.title("📊 Financial Data Analysis with Machine Learning")
    st.subheader("AF3005 - Programming for Finance (Spring 2025)")
    
    # Display finance-themed image
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1534951009808-766178b47a4f", 
                 caption="Welcome to Financial Data Analysis")
    
    st.markdown("""
    ## 🌟 Welcome to the Financial Data Analysis Application
    
    This interactive application allows you to analyze financial data using machine learning techniques.
    You can upload your own financial dataset or fetch real-time stock data from Yahoo Finance.
    
    ### 📋 Features:
    - Data loading and preprocessing
    - Feature engineering and selection
    - Model training and evaluation
    - Interactive visualizations
    - Downloadable results
    
    ### 🚀 How to Use:
    1. Use the sidebar to select your data source
    2. Follow the step-by-step ML pipeline
    3. Analyze the results and visualizations
    
    Let's get started! 👇
    """)
    
    if st.button("Begin Analysis", key="begin_analysis"):
        st.session_state.step = 1
        st.rerun()

# Sidebar navigation
def sidebar():
    st.sidebar.title("📈 Financial Analysis")
    st.sidebar.markdown("---")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio("Select data source:", ["Upload Dataset", "Yahoo Finance API"])
    
    if data_source == "Upload Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload financial dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.sidebar.success("Dataset successfully loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {e}")
        
    else:  # Yahoo Finance API
        ticker_symbol = st.sidebar.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):", "AAPL")
        period = st.sidebar.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        interval = st.sidebar.selectbox("Select time interval:", ["1d", "1wk", "1mo"], index=0)
        
        if st.sidebar.button("Fetch Stock Data"):
            try:
                with st.sidebar:
                    with st.spinner("Fetching stock data..."):
                        stock_data = yf.download(ticker_symbol, period=period, interval=interval)
                        if not stock_data.empty:
                            stock_data = stock_data.reset_index()
                            stock_data = format_stock_data(stock_data)
                            st.session_state.data = stock_data
                            st.session_state.real_time_data = {
                                'ticker': ticker_symbol, 
                                'period': period, 
                                'interval': interval
                            }
                            st.success(f"Stock data for {ticker_symbol} successfully fetched!")
                        else:
                            st.error("No data found for the specified ticker symbol.")
            except Exception as e:
                st.sidebar.error(f"Error fetching stock data: {e}")
    
    # Display ML Pipeline steps
    st.sidebar.markdown("---")
    st.sidebar.subheader("ML Pipeline")
    
    # Create a disabled button for the current step and enabled buttons for completed steps
    steps = [
        "1. Load Data", 
        "2. Preprocessing", 
        "3. Feature Engineering", 
        "4. Train/Test Split", 
        "5. Model Training", 
        "6. Evaluation", 
        "7. Results Visualization"
    ]
    
    for i, step in enumerate(steps, 1):
        if i < st.session_state.step:
            if st.sidebar.button(step, key=f"step_{i}"):
                st.session_state.step = i
                st.rerun()
        elif i == st.session_state.step:
            st.sidebar.markdown(f"**→ {step}**")
        else:
            st.sidebar.button(step, key=f"step_{i}", disabled=True)
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Application"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.step = 0
        st.rerun()
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info("Developed for AF3005 - Programming for Finance course at FAST-NUCES Islamabad")

# Step 1: Load Data
def load_data_step():
    st.header("Step 1: Load Data")
    
    if st.session_state.data is None:
        st.warning("Please select a data source from the sidebar to proceed.")
        st.image("https://images.unsplash.com/photo-1642465789831-a176eb4a1b14", 
                 caption="Waiting for data input...")
        return
    
    st.success("Data successfully loaded! 🎉")
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(st.session_state.data.head())
    
    # Show data information
    st.subheader("Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Rows", st.session_state.data.shape[0])
        st.metric("Missing Values", st.session_state.data.isna().sum().sum())
    with col2:
        st.metric("Number of Columns", st.session_state.data.shape[1])
        st.metric("Data Types", len(st.session_state.data.dtypes.unique()))
    
    # Display basic statistics
    with st.expander("View Descriptive Statistics"):
        st.dataframe(st.session_state.data.describe().round(2))
    
    # Continue button
    if st.button("Continue to Preprocessing", key="to_preprocessing"):
        st.session_state.step = 2
        st.rerun()

# Step 2: Preprocessing
def preprocessing_step():
    st.header("Step 2: Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please go back to Step 1.")
        return
    
    # Display data type information
    st.subheader("Data Types")
    dtypes_list = [str(dtype) for dtype in st.session_state.data.dtypes.values]
    dtypes_df = pd.DataFrame({
        "Column": st.session_state.data.columns, 
        "Data Type": dtypes_list
    })
    st.dataframe(dtypes_df)
    
    # Display missing values information
    st.subheader("Missing Values Analysis")
    missing_values = st.session_state.data.isnull().sum()
    missing_df = pd.DataFrame({"Column": missing_values.index, 
                              "Missing Values": missing_values.values,
                              "Percentage": (missing_values.values / len(st.session_state.data) * 100).round(2)})
    st.dataframe(missing_df)
    
    # Display outlier visualization
    st.subheader("Outlier Analysis")
    numerical_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numerical_cols) > 0:
        selected_col = st.selectbox("Select column for outlier visualization:", numerical_cols)
        fig = px.box(st.session_state.data, y=selected_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numerical columns available for outlier analysis.")
    
    # Preprocessing options
    st.subheader("Preprocessing Options")
    
    col1, col2 = st.columns(2)
    with col1:
        handle_missing = st.radio("Handle Missing Values:", ["Drop rows", "Fill with mean/mode", "None"])
    with col2:
        handle_outliers = st.radio("Handle Outliers:", ["Clip outliers", "None"])
    
    # Process data button
    if st.button("Process Data", key="process_data"):
        with st.spinner("Processing data..."):
            processed_data = preprocess_data(
                st.session_state.data,
                handle_missing=handle_missing,
                handle_outliers=handle_outliers
            )
            st.session_state.processed_data = processed_data
            
            # Calculate and display the changes
            st.subheader("Preprocessing Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", st.session_state.data.shape[0])
            with col2:
                st.metric("Processed Rows", processed_data.shape[0])
            with col3:
                rows_diff = processed_data.shape[0] - st.session_state.data.shape[0]
                st.metric("Rows Difference", rows_diff, delta_color="inverse")
            
            st.success("Data preprocessing completed successfully! 🧹")
    
    # Continue button (only show if processed_data exists)
    if st.session_state.processed_data is not None:
        if st.button("Continue to Feature Engineering", key="to_feature_eng"):
            st.session_state.step = 3
            st.rerun()

# Step 3: Feature Engineering
def feature_engineering_step():
    st.header("Step 3: Feature Engineering")
    
    if st.session_state.processed_data is None:
        st.warning("No processed data available. Please go back to Step 2.")
        return
    
    # Display current data
    st.subheader("Current Data")
    st.dataframe(st.session_state.processed_data.head())
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Target variable selection
    numerical_cols = st.session_state.processed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numerical_cols) == 0:
        st.error("No numerical columns available for analysis. Please check your data.")
        return
    
    target_col = st.selectbox("Select target variable (for regression/classification):", 
                              numerical_cols,
                              key="target_selector")
    
    # Feature selection
    feature_cols = st.multiselect("Select features for analysis:", 
                                 [col for col in numerical_cols if col != target_col],
                                 default=[col for col in numerical_cols[:3] if col != target_col],
                                 key="feature_selector")
    
    if len(feature_cols) == 0:
        st.warning("Please select at least one feature.")
        return
    
    # Feature transformation options
    st.subheader("Feature Transformation")
    
    col1, col2 = st.columns(2)
    with col1:
        scale_features = st.checkbox("Scale numerical features", value=True)
    with col2:
        add_polynomial = st.checkbox("Add polynomial features", value=False)
        if add_polynomial:
            poly_degree = st.slider("Polynomial degree:", min_value=2, max_value=5, value=2)
    
    # Apply feature engineering button
    if st.button("Engineer Features", key="engineer_features"):
        with st.spinner("Engineering features..."):
            X, y, feature_names = engineer_features(
                st.session_state.processed_data,
                feature_cols=feature_cols,
                target_col=target_col,
                scale_features=scale_features,
                add_polynomial=add_polynomial,
                poly_degree=poly_degree if add_polynomial else None
            )
            
            st.session_state.features = feature_names
            st.session_state.target = target_col
            
            # Visualize features
            st.subheader("Feature Correlation Matrix")
            
            # Create correlation matrix for selected features
            if len(feature_cols) > 1:
                corr = st.session_state.processed_data[feature_cols].corr()
                fig = px.imshow(corr, 
                               text_auto=True, 
                               color_continuous_scale='RdBu_r',
                               title="Feature Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature-target relationship
            st.subheader("Feature-Target Relationship")
            
            selected_feature = st.selectbox("Select feature to visualize:", feature_cols)
            # Create a scatter plot without trendline
            fig = px.scatter(st.session_state.processed_data, 
                            x=selected_feature, 
                            y=target_col,
                            title=f"{selected_feature} vs {target_col}")
            
            # Add a regression line manually instead of using trendline="ols"
            if len(st.session_state.processed_data) > 1:  # Need at least 2 points for regression
                x_values = st.session_state.processed_data[selected_feature].values
                y_values = st.session_state.processed_data[target_col].values
                
                # Simple linear regression calculation
                valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_valid = x_values[valid_indices]
                y_valid = y_values[valid_indices]
                
                if len(x_valid) > 1:  # Need at least 2 valid points
                    slope, intercept = np.polyfit(x_valid, y_valid, 1)
                    x_range = np.linspace(min(x_valid), max(x_valid), 100)
                    y_range = slope * x_range + intercept
                
                    # Add the regression line to the figure
                    fig.add_trace(go.Scatter(
                        x=x_range, 
                        y=y_range, 
                        mode='lines', 
                        name=f'Trend (y = {slope:.2f}x + {intercept:.2f})',
                        line=dict(color='red', dash='dash')
                    ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store engineered data in session state
            st.session_state.X = X
            st.session_state.y = y
            
            st.success("Feature engineering completed successfully! ✨")
    
    # Continue button (only show if features and target are selected)
    if 'X' in st.session_state and 'y' in st.session_state:
        if st.button("Continue to Train/Test Split", key="to_split"):
            st.session_state.step = 4
            st.rerun()

# Step 4: Train/Test Split
def train_test_split_step():
    st.header("Step 4: Train/Test Split")
    
    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.warning("Feature engineering not completed. Please go back to Step 3.")
        return
    
    # Split options
    st.subheader("Split Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size (%):", min_value=10, max_value=40, value=20)
        test_size = test_size / 100  # Convert to proportion
    with col2:
        random_state = st.number_input("Random seed:", min_value=0, max_value=999, value=42)
    
    # Apply split button
    if st.button("Split Data", key="split_data"):
        with st.spinner("Splitting data..."):
            X_train, X_test, y_train, y_test = split_data(
                st.session_state.X,
                st.session_state.y,
                test_size=test_size,
                random_state=random_state
            )
            
            # Store split data in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # Visualize split
            st.subheader("Train/Test Split Visualization")
            
            fig = visualize_split(len(X_train), len(X_test))
            st.plotly_chart(fig, use_container_width=True)
            
            # Display split information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", len(X_train))
            with col2:
                st.metric("Testing Samples", len(X_test))
            with col3:
                st.metric("Total Features", X_train.shape[1])
            
            st.success("Data split completed successfully! 🧩")
    
    # Continue button (only show if data is split)
    if 'X_train' in st.session_state and st.session_state.X_train is not None:
        if st.button("Continue to Model Training", key="to_model"):
            st.session_state.step = 5
            st.rerun()

# Step 5: Model Training
def model_training_step():
    st.header("Step 5: Model Training")
    
    if 'X_train' not in st.session_state or st.session_state.X_train is None:
        st.warning("Data split not completed. Please go back to Step 4.")
        return
    
    # Model selection
    st.subheader("Model Selection")
    
    model_type = st.selectbox(
        "Select model type:", 
        ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
    )
    
    # Model parameters
    st.subheader("Model Parameters")
    
    if model_type == "Linear Regression":
        col1, col2 = st.columns(2)
        with col1:
            fit_intercept = st.checkbox("Fit intercept", value=True)
        with col2:
            normalize = st.checkbox("Normalize", value=False)
    
    elif model_type == "Logistic Regression":
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("Regularization strength (C):", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        with col2:
            max_iter = st.slider("Maximum iterations:", min_value=100, max_value=1000, value=100, step=100)
    
    elif model_type == "K-Means Clustering":
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3)
        with col2:
            max_iter = st.slider("Maximum iterations:", min_value=100, max_value=1000, value=300, step=100)
    
    # Train model button
    if st.button("Train Model", key="train_model"):
        with st.spinner("Training model..."):
            # Training progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Train selected model
            if model_type == "Linear Regression":
                model, predictions, evaluation = train_linear_regression(
                    st.session_state.X_train,
                    st.session_state.X_test,
                    st.session_state.y_train,
                    st.session_state.y_test,
                    fit_intercept=fit_intercept,
                    normalize=normalize
                )
                
                # Store feature importance for linear regression
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.features,
                    'Importance': np.abs(model.coef_)
                }).sort_values('Importance', ascending=False)
                st.session_state.feature_importance = feature_importance
                
            elif model_type == "Logistic Regression":
                model, predictions, evaluation = train_logistic_regression(
                    st.session_state.X_train,
                    st.session_state.X_test,
                    st.session_state.y_train,
                    st.session_state.y_test,
                    C=C,
                    max_iter=max_iter
                )
                
                # Store feature importance for logistic regression
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.features,
                    'Importance': np.abs(model.coef_[0])
                }).sort_values('Importance', ascending=False)
                st.session_state.feature_importance = feature_importance
                
            elif model_type == "K-Means Clustering":
                model, predictions, evaluation = train_kmeans_clustering(
                    st.session_state.X_train,
                    st.session_state.X_test,
                    n_clusters=n_clusters,
                    max_iter=max_iter
                )
            
            # Store model and results in session state
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.predictions = predictions
            st.session_state.evaluation = evaluation
            
            st.success(f"{model_type} model trained successfully! 🚀")
    
    # Continue button (only show if model is trained)
    if 'model' in st.session_state and st.session_state.model is not None:
        if st.button("Continue to Evaluation", key="to_evaluation"):
            st.session_state.step = 6
            st.rerun()

# Step 6: Model Evaluation
def evaluation_step():
    st.header("Step 6: Model Evaluation")
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Model not trained. Please go back to Step 5.")
        return
    
    # Display evaluation metrics
    st.subheader("Performance Metrics")
    
    model_type = st.session_state.model_type
    evaluation = st.session_state.evaluation
    
    # Create metrics based on model type
    if model_type in ["Linear Regression", "Logistic Regression"]:
        col1, col2, col3 = st.columns(3)
        
        if model_type == "Linear Regression":
            with col1:
                st.metric("R² Score", round(evaluation['r2'], 4))
            with col2:
                st.metric("Mean Absolute Error", round(evaluation['mae'], 4))
            with col3:
                st.metric("Root Mean Squared Error", round(evaluation['rmse'], 4))
            
            # Residual plot
            st.subheader("Residual Analysis")
            residuals = st.session_state.y_test - st.session_state.predictions
            
            fig = px.scatter(
                x=st.session_state.predictions, 
                y=residuals,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title="Residual Plot"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_type == "Logistic Regression":
            with col1:
                st.metric("Accuracy", round(evaluation['accuracy'], 4))
            with col2:
                st.metric("Precision", round(evaluation['precision'], 4))
            with col3:
                st.metric("Recall", round(evaluation['recall'], 4))
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cf_matrix = evaluation['confusion_matrix']
            
            fig = px.imshow(
                cf_matrix,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Class 0', 'Class 1'],
                y=['Class 0', 'Class 1'],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
    elif model_type == "K-Means Clustering":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Silhouette Score", round(evaluation['silhouette'], 4))
        with col2:
            st.metric("Inertia", round(evaluation['inertia'], 2))
        
        # Cluster visualization if 2D
        if st.session_state.X_test.shape[1] >= 2:
            st.subheader("Cluster Visualization (First 2 Dimensions)")
            
            # Create dataframe for plotting
            plot_df = pd.DataFrame({
                'Feature 1': st.session_state.X_test[:, 0],
                'Feature 2': st.session_state.X_test[:, 1],
                'Cluster': st.session_state.predictions
            })
            
            fig = px.scatter(
                plot_df, 
                x='Feature 1', 
                y='Feature 2', 
                color='Cluster',
                title="Cluster Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance visualization (for regression models)
    if model_type in ["Linear Regression", "Logistic Regression"] and st.session_state.feature_importance is not None:
        st.subheader("Feature Importance")
        
        fig = visualize_feature_importance(st.session_state.feature_importance)
        st.plotly_chart(fig, use_container_width=True)
    
    # Continue button
    if st.button("Continue to Results Visualization", key="to_results"):
        st.session_state.step = 7
        st.rerun()

# Step 7: Results Visualization
def results_visualization_step():
    st.header("Step 7: Results Visualization")
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Model not trained. Please go back to Step 5.")
        return
    
    model_type = st.session_state.model_type
    
    # Create visualization based on model type
    st.subheader("Results Visualization")
    
    fig = visualize_results(
        model_type=model_type,
        X_test=st.session_state.X_test,
        y_test=st.session_state.y_test if 'y_test' in st.session_state else None,
        predictions=st.session_state.predictions,
        target_name=st.session_state.target if 'target' in st.session_state else "Target",
        feature_names=st.session_state.features if 'features' in st.session_state else None
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Real-time updates (if using stock data)
    if 'real_time_data' in st.session_state and st.session_state.real_time_data is not None:
        st.subheader("Real-time Stock Data")
        
        ticker = st.session_state.real_time_data['ticker']
        
        if st.button("Refresh Stock Data", key="refresh_stock"):
            with st.spinner(f"Fetching latest data for {ticker}..."):
                try:
                    # Fetch the latest data
                    latest_data = yf.download(
                        ticker, 
                        period='1d', 
                        interval='1m'
                    )
                    
                    if not latest_data.empty:
                        # Show the latest price
                        latest_price = latest_data['Close'].iloc[-1]
                        previous_close = latest_data['Close'].iloc[0]
                        price_change = latest_price - previous_close
                        percent_change = (price_change / previous_close) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{ticker} Latest Price", f"${latest_price:.2f}")
                        with col2:
                            st.metric("Change", f"${price_change:.2f}", delta=f"{percent_change:.2f}%")
                        with col3:
                            st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
                        
                        # Plot intraday data
                        fig = px.line(
                            latest_data.reset_index(), 
                            x='Datetime', 
                            y='Close', 
                            title=f"{ticker} Intraday Prices"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No real-time data available for {ticker} right now.")
                except Exception as e:
                    st.error(f"Error fetching real-time data: {e}")
    
    # Download section
    st.subheader("Download Results")
    
    if model_type in ["Linear Regression", "Logistic Regression"]:
        # Create results dataframe
        results_df = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': st.session_state.predictions
        })
        
        download_results(results_df, "model_predictions.csv")
    
    elif model_type == "K-Means Clustering":
        # Create results dataframe with cluster assignments
        if st.session_state.X_test.shape[1] <= 10:  # Only if not too many features
            # Create a DataFrame with feature names and cluster assignments
            features_data = pd.DataFrame(
                st.session_state.X_test, 
                columns=[f"Feature_{i+1}" if st.session_state.features is None else st.session_state.features[i] 
                        for i in range(st.session_state.X_test.shape[1])]
            )
            features_data['Cluster'] = st.session_state.predictions
            
            download_results(features_data, "cluster_assignments.csv")
    
    # Conclusion
    st.markdown("---")
    st.subheader("Analysis Complete! 🎉")
    st.markdown("""
    You have successfully completed the financial data analysis using machine learning.
    
    **What's next?**
    - Try different models or parameters
    - Analyze different datasets
    - Export your results for further analysis
    
    Thank you for using this application!
    """)
    
    # Display finance image
    st.image("https://images.unsplash.com/photo-1488459716781-31db52582fe9", 
             caption="Financial Analysis Complete")

# Main application flow
def main():
    sidebar()
    
    # Display appropriate step based on session state
    if st.session_state.step == 0:
        welcome_page()
    elif st.session_state.step == 1:
        load_data_step()
    elif st.session_state.step == 2:
        preprocessing_step()
    elif st.session_state.step == 3:
        feature_engineering_step()
    elif st.session_state.step == 4:
        train_test_split_step()
    elif st.session_state.step == 5:
        model_training_step()
    elif st.session_state.step == 6:
        evaluation_step()
    elif st.session_state.step == 7:
        results_visualization_step()

if __name__ == "__main__":
    main()
