import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Loading the data
data = pd.read_csv('sales_data.csv')

data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Ploting the sales data
plt.figure(figsize=(10, 6))
plt.plot(data['sales'])
plt.title(' Multi Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales (LKR)')
plt.show()

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['sales'].values.reshape(-1, 1))

# Creating training and test sets
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

train_data = scaled_data[:training_data_len, :]
test_data = scaled_data[training_data_len:, :]

# Printing the lengths of the training and test data to ensure they are correct
print(f'Length of training data: {len(train_data)}')
print(f'Length of test data: {len(test_data)}')

# Creating the training data
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Adjusting look_back period if needed
look_back = 30  # Reduced look_back period
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Print shapes to debug
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# Checking if X_test is empty
if X_test.shape[0] == 0:
    raise ValueError("X_test is empty")

# Reshaping the data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Making predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculating RMSE
train_score = np.sqrt(np.mean(np.square(y_train - train_predict[:, 0])))
test_score = np.sqrt(np.mean(np.square(y_test - test_predict[:, 0])))
print(f'Train RMSE: {train_score}, Test RMSE: {test_score}')

print(f'train_predict shape: {train_predict.shape}')
print(f'test_predict shape: {test_predict.shape}')

# Plotting the results
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[look_back:len(train_predict) + look_back, :] = train_predict

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
start_index = len(train_predict) + (look_back * 2)
end_index = start_index + len(test_predict)

print(f'start_index: {start_index}, end_index: {end_index}, len(scaled_data): {len(scaled_data)}')

test_plot[start_index:end_index, :] = test_predict

plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Sales')
plt.plot(train_plot, label='Training Predictions')
plt.plot(test_plot, label='Testing Predictions')
plt.title('Multi Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
