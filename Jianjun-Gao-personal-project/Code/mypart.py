# %%
# This part is APIfetch.py
import pandas as pd
import requests
from datetime import datetime, timedelta

pd.set_option('display.max_columns', None)

# %%
api_key = 'kNGGuBdk94L1uVsH1n1r3GMiF1sq6NlF'

# Define the API URL for common stock list on NASDAQ and NYSE
api_url = 'https://financialmodelingprep.com/api/v3/stock/list?apikey=kNGGuBdk94L1uVsH1n1r3GMiF1sq6NlF'

# Make the API request
response = requests.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data_json = response.json()

    # Check if the response contains multiple JSON objects
    if isinstance(data_json, list):
        # Concatenate all JSON objects into a single DataFrame
        data_frames = [pd.json_normalize(item) for item in data_json]
        data = pd.concat(data_frames, ignore_index=True)
    else:
        # If it's a single JSON object, directly convert it to a DataFrame
        data = pd.json_normalize(data_json)

else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

# Filtering NYSE stocks
filtered_data_NYSE = data[(data['exchangeShortName'] == 'NYSE') & (data['type'] == 'stock')]
filtered_data_NYSE = filtered_data_NYSE[filtered_data_NYSE['symbol'].str.contains(r'[.-]', regex=True) & filtered_data_NYSE['symbol'].str.len().between(1,4)]

# Filtering NASDAQ stocks
filtered_data_NASDAQ = data[(data['exchangeShortName'] == 'NASDAQ') & (data['type'] == 'stock')]
filtered_data_NASDAQ = filtered_data_NASDAQ[
    ~filtered_data_NASDAQ['symbol'].str.contains(r'[.-]', regex=True) & filtered_data_NASDAQ[
        'symbol'].str.len().between(1, 4)]

# Combining NYSE and NASDAQ filtered data into one DataFrame
common_stocks = pd.concat([filtered_data_NYSE, filtered_data_NASDAQ], ignore_index=True)
common_stocks = common_stocks[['symbol', 'name', 'exchangeShortName']]

# Define the API URL for stock list that has statements
api_url = 'https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey=kNGGuBdk94L1uVsH1n1r3GMiF1sq6NlF'
# Make the API request
response = requests.get(api_url)
statement_stocks = response.json()

# Finding the intersection of common_stocks and statement_stocks
common_stocks_with_statements = common_stocks[common_stocks['symbol'].isin(statement_stocks)]
common_stocks_with_statements.reset_index(inplace=True, drop=True)

# Define the list of financial statement types
statement_types = ['income-statement', 'balance-sheet-statement', 'cash-flow-statement']

# Base URL for financial statements
base_url = 'https://financialmodelingprep.com/api/v3/'

# Initialize an empty list to store the final combined DataFrames
final_data_frames = []

# Iterate over each symbol in common_stocks_with_statements
for symbol in common_stocks_with_statements['symbol']:
    # Initialize a list to store the DataFrames for each type of statement for the current symbol
    combined_symbol_data = []

    for statement in statement_types:
        # Define the API URL for each financial statement
        api_url = f'{base_url}{statement}/{symbol}?period=annual&apikey={api_key}'

        # Make the API request
        response = requests.get(api_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data_json = response.json()

            # Check if the response contains multiple JSON objects
            if isinstance(data_json, list) and len(data_json) > 0:
                # Normalize JSON data into a DataFrame
                df = pd.json_normalize(data_json)
                # Remove the link and finalLink columns
                df = df.drop(columns=['link', 'finalLink'], errors='ignore')
                combined_symbol_data.append(df)
        else:
            print(f"Failed to fetch data for {symbol} - {statement}. Status code: {response.status_code}")

    # If all three statements were successfully fetched and combined
    if len(combined_symbol_data) == len(statement_types):
        # Merge the DataFrames on common columns (date, symbol, calendarYear, period)
        symbol_data_combined = combined_symbol_data[0]
        for df in combined_symbol_data[1:]:
            symbol_data_combined = pd.merge(symbol_data_combined, df, on=['date', 'symbol', 'calendarYear', 'period'],
                                            how='outer')

        # Append the combined DataFrame for the current symbol to the final list
        final_data_frames.append(symbol_data_combined)

# Combine all symbols' data row-wise into the final DataFrame
final_data = pd.concat(final_data_frames, ignore_index=True)

# Drop rows with missing values
final_data = final_data.dropna()

# Drop unnecessary columns
final_data = final_data.drop(['reportedCurrency_x', 'cik_x', 'fillingDate_x', 'acceptedDate_x', 'period',
                              'reportedCurrency_y', 'cik_y', 'fillingDate_y', 'acceptedDate_y', 'netIncome_y',
                              'depreciationAndAmortization_y',
                              'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate', 'inventory_y'], axis=1)

# Add the name column from common_stocks_with_statements to final_data
final_data = final_data.merge(common_stocks_with_statements[['symbol', 'name']], on='symbol', how='left')

# Reorder columns to place 'name' next to 'symbol'
columns = list(final_data.columns)
symbol_index = columns.index('symbol')
columns.insert(symbol_index + 1, columns.pop(columns.index('name')))
final_data = final_data[columns]

# Function to check if a column contains only zeros
def check_all_zeros(column):
    return (column == 0).all()


# Apply the function to each column in the DataFrame
zero_columns = final_data.columns[final_data.apply(check_all_zeros)]

# Print the columns that contain only zeros
print("Columns that contain only zeros:")
print(zero_columns)

# Display the final DataFrame
print(final_data)

# Save the final DataFrame to CSV and Excel
final_data.to_csv('final_data.csv', index=False)
final_data.to_excel('final_data.xlsx', index=False)


# %%
import pandas as pd
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

df = pd.read_csv('final_data.csv')

# %%
api_key = 'kNGGuBdk94L1uVsH1n1r3GMiF1sq6NlF'

# Function to get adjClose price, with efficiency improvements
def get_adj_close(symbol, date, api_key):
    max_days_to_check = 30  # Limit how far back to check (e.g., up to 30 days)
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    
    for _ in range(max_days_to_check):
        date_str = date_obj.strftime('%Y-%m-%d')
        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={date_str}&to={date_str}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        
        # Check if data is available
        if 'historical' in data and len(data['historical']) > 0:
            return data['historical'][0]['adjClose']
        
        # Move to the previous day
        date_obj -= timedelta(days=1)
    
    return None  # Return None if no data was found within the date range

# Function to fetch adjClose in parallel
def fetch_adj_close_parallel(row):
    symbol, date = row['symbol'], row['date'].replace('/', '-')
    return get_adj_close(symbol, date, api_key)

# Create a ThreadPoolExecutor to parallelize the fetching process
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_adj_close_parallel, row): idx for idx, row in df.iterrows()}
    
    for future in as_completed(futures):
        idx = futures[future]
        try:
            df.at[idx, 'adjClose'] = future.result()
        except Exception as e:
            print(f"Error fetching data for row {idx}: {e}")
            df.at[idx, 'adjClose'] = None

# Save the updated DataFrame to a new CSV
df.to_csv('final_data_with_adjClose.csv', index=False)
df.to_excel('final_data_with_adjClose.xlsx',index=False)








# This part is how I add labels on data, and feature engineering
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, SimpleRNN, Bidirectional, Layer
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# data
df = pd.read_csv('final_data_with_adjClose.csv')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

def create_labels(group):
    group = group.sort_values('calendarYear')
    group['price_direction'] = (group['adjClose'].shift(-1) > group['adjClose']).astype(int)
    return group

df = df.groupby('symbol', group_keys=False).apply(create_labels)
df = df.dropna(subset=['price_direction'])

features = ['revenue', 'grossProfit', 'operatingIncome', 'netIncome_x',
            'cashAndCashEquivalents', 'totalAssets', 'totalLiabilities', 'adjClose']
df['eps_normalized'] = df['eps'] / df['adjClose']
df['debt_to_equity'] = df['totalDebt'] / (df['totalEquity'] + 1e-6)  # Avoid division by zero
df['market_cap'] = df['adjClose'] * df['weightedAverageShsOut']
features.extend(['eps_normalized', 'debt_to_equity', 'market_cap'])

scaler = MinMaxScaler()
def scale_group(group):
    group[features] = scaler.fit_transform(group[features])
    return group

df = df.groupby('symbol', group_keys=False).apply(scale_group)

# model Bidirectional-LSTM, RNN, CNN-LSTM
def build_lstm_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_rnn_model():
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(SimpleRNN(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_cnn_lstm_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


