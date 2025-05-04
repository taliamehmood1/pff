import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime

def download_results(dataframe, filename="results.csv"):
    """
    Create a download button for a dataframe
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The dataframe to download
    filename : str
        The name of the file to download
    """
    # Convert dataframe to csv
    csv = dataframe.to_csv(index=False)
    
    # Create download button
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)

def format_stock_data(stock_data):
    """
    Format stock data for analysis
    
    Parameters:
    -----------
    stock_data : pandas.DataFrame
        The stock data from yfinance
    
    Returns:
    --------
    pandas.DataFrame
        Formatted stock data
    """
    # Reset index if Datetime is the index
    if 'Date' in stock_data.columns or 'Datetime' in stock_data.columns:
        if 'Date' in stock_data.columns:
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        else:
            stock_data['Date'] = pd.to_datetime(stock_data['Datetime'])
    
    # Calculate additional technical indicators
    # 1. Daily Returns
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    
    # 2. Moving Averages
    stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    
    # 3. Relative Strength Index (RSI) - Basic calculation
    delta = stock_data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Volatility (standard deviation of returns)
    stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=20).std() * np.sqrt(20)
    
    # 5. Price changes
    stock_data['Price_Change'] = stock_data['Close'] - stock_data['Open']
    
    # Drop NaN values resulting from calculations
    stock_data = stock_data.dropna()
    
    return stock_data

def load_sample_data():
    """
    Load a sample financial dataset for demonstration
    
    Returns:
    --------
    pandas.DataFrame
        Sample financial data
    """
    # Create a sample dataset with stock-like data
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2022-01-01', periods=252, freq='B')  # Business days for a year
    
    # Create price data
    open_prices = np.random.normal(loc=100, scale=1, size=len(dates))
    high_prices = open_prices + np.random.uniform(0, 2, size=len(dates))
    low_prices = open_prices - np.random.uniform(0, 2, size=len(dates))
    close_prices = open_prices + np.random.normal(loc=0, scale=1, size=len(dates))
    volume = np.random.randint(1000000, 10000000, size=len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    })
    
    # Add technical indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Simple momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(20)
    
    # Drop NaN values
    df = df.dropna()
    
    return df
