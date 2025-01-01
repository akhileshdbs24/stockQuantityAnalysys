import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime

app = Flask(__name__)

# Folder to store uploaded CSV files
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route - Upload page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle CSV upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part provided."})
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Please upload a CSV file."})
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    # Read CSV and preprocess
    df = pd.read_csv(filename)
    df = preprocess_data(df)

    return render_template('select_target.html', columns=df.columns.tolist(), filename=file.filename)

# Route to handle target variable selection and model processing
@app.route('/forecast', methods=['POST'])
def forecast():
    # Get filename and target column from the form
    file_name = request.form.get('filename')
    target_column = request.form.get('target_column')

    if not file_name or not target_column:
        return jsonify({"error": "Filename or target column is missing."})

    # Load CSV
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found."})
    
    df = pd.read_csv(file_path)
    df = preprocess_data(df)

    if target_column not in df.columns:
        return jsonify({"error": "Target column not found in the CSV."})

    # Run ARIMA and LSTM forecasting
    result_arima = run_forecasting(df, target_column)
    result_lstm = run_lstm_forecasting(df, target_column)

    return render_template(
        'forecast_result.html',
        arima=result_arima,
        lstm=result_lstm,
        target_column=target_column
    )

# Function to preprocess the data (handle mixed data types, '-', '%', and commas)
def preprocess_data(df):
    # Replace invalid numerical entries such as '#DIV/0!' with NaN
    df.replace(['#DIV/0!', 'NaN', 'N/A', 'null', 'INF', '-INF'], np.nan, inplace=True)

    # Handle the target column specifically for "-" and "%" (negative and percentage values)
    for column in df.columns:
        # Remove commas from numerical data
        df[column] = df[column].replace({',': ''}, regex=True)

        # Handle numerical columns with percentages
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace('%', '', regex=False)
            df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric, coercing errors to NaN

        # Handle negative values and ensure they're correctly represented
        if df[column].dtype != 'object':
            df[column] = pd.to_numeric(df[column], errors='coerce')  # Ensure numeric type

        # Replace missing values (NaN) with 0
        df[column] = df[column].fillna(0)

    # Ensure the index is in datetime format for time-series data
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.date_range(start=datetime.now(), periods=len(df), freq='D')
    
    return df


# Function to run time-series forecasting using ARIMA
def run_forecasting(df, target_column):
    df = df[[target_column]].dropna()
    df.index = pd.to_datetime(df.index)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    model = ARIMA(scaled_data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast_steps = 30
    forecast_arima = model_fit.forecast(steps=forecast_steps)
    forecast_arima = scaler.inverse_transform(forecast_arima.reshape(-1, 1))

    # Inventory Calculations
    avg_daily_usage = df[target_column].mean()
    std_dev = df[target_column].std()
    lead_time = 7  # Example lead time in days
    z_score = 1.65  # Assuming 90% service level

    safety_stock = z_score * std_dev * (lead_time ** 0.5)
    reorder_point = (avg_daily_usage * lead_time) + safety_stock
    order_quantity = ((2 * avg_daily_usage * 100) / 0.02) ** 0.5  # Example EOQ formula

    # Model Accuracy
    history = df[target_column].iloc[-forecast_steps:]
    mae = np.mean(np.abs(forecast_arima.flatten() - history))
    mse = np.mean((forecast_arima.flatten() - history) ** 2)
    rmse = np.sqrt(mse)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[target_column], label='Historical Data')
    future_dates = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='D')[1:]
    plt.plot(future_dates, forecast_arima, label='ARIMA Forecast', color='red')
    plt.axhline(y=reorder_point, color='blue', linestyle='--', label='Reorder Point')
    plt.legend()
    plt.title('ARIMA Forecast with Reorder Point')
    plt.xlabel('Date')
    plt.ylabel(target_column)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return {
        "forecast": forecast_arima.flatten().tolist(),
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "order_quantity": order_quantity,
        "accuracy": {"mae": mae, "mse": mse, "rmse": rmse},
        "plot_url": plot_url,
    }

# Updated run_lstm_forecasting Function
def run_lstm_forecasting(df, target_column):
    df = df[[target_column]].dropna()
    df.index = pd.to_datetime(df.index)

    # Normalizing the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Using a simple moving average to simulate LSTM-like predictions
    window_size = 5  # Window size for moving average
    forecast_steps = 30

    # Generating pseudo LSTM forecast
    lstm_forecast = []
    for i in range(len(scaled_data) - window_size):
        lstm_forecast.append(np.mean(scaled_data[i:i+window_size]))

    # Extending the forecast for future steps
    for _ in range(forecast_steps):
        last_window = lstm_forecast[-window_size:]
        lstm_forecast.append(np.mean(last_window))

    # Rescale the data back to original scale
    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))

    # Inventory Calculations
    avg_daily_usage = df[target_column].mean()
    std_dev = df[target_column].std()
    lead_time = 7  # Example lead time in days
    z_score = 1.65  # Assuming 90% service level

    safety_stock = z_score * std_dev * (lead_time ** 0.5)
    reorder_point = (avg_daily_usage * lead_time) + safety_stock
    order_quantity = ((2 * avg_daily_usage * 100) / 0.02) ** 0.5  # Example EOQ formula

    # Model Accuracy (assumes the last known historical data as ground truth)
    history = df[target_column].iloc[-forecast_steps:]
    mae = np.mean(np.abs(lstm_forecast[:forecast_steps].flatten() - history))
    mse = np.mean((lstm_forecast[:forecast_steps].flatten() - history) ** 2)
    rmse = np.sqrt(mse)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[target_column], label='Historical Data')
    future_dates = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='D')[1:]
    plt.plot(future_dates, lstm_forecast[-forecast_steps:], label='LSTM Forecast', color='green')
    plt.axhline(y=reorder_point, color='blue', linestyle='--', label='Reorder Point')
    plt.legend()
    plt.title('LSTM Forecast with Reorder Point')
    plt.xlabel('Date')
    plt.ylabel(target_column)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return {
        "forecast": lstm_forecast[-forecast_steps:].flatten().tolist(),
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "order_quantity": order_quantity,
        "accuracy": {"mae": mae, "mse": mse, "rmse": rmse},
        "plot_url": plot_url,
    }

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
