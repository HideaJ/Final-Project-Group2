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

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)][features].values
        y = data.iloc[i + seq_length]['price_direction']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5
X, y = [], []
for symbol, group in df.groupby('symbol'):
    if len(group) >= seq_length + 1:
        X_symbol, y_symbol = create_sequences(group.reset_index(drop=True), seq_length)
        if X_symbol.ndim == 3:
            X.append(X_symbol)
            y.append(y_symbol)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Attention layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(1,),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# model
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

def build_lstm_attention_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


models = {
    "LSTM": build_lstm_model(),
    "RNN": build_rnn_model(),
    "CNN-LSTM": build_cnn_lstm_model(),
    "LSTM-Attention": build_lstm_attention_model()
}

# training
histories = {}
final_results = {}
for name, model in models.items():
    print(f"Training {name} model...")
    optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(dataset, epochs=50, validation_data=(X_test, y_test), shuffle=False, callbacks=[reduce_lr, early_stopping])
    histories[name] = history

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    final_results[name] = test_accuracy
    print(f"{name} Test Accuracy: {test_accuracy:.4f}")

print("\nFinal Model Accuracies:")
for name, accuracy in final_results.items():
    print(f"{name}: {accuracy:.4f}")

plt.figure()
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Validation Accuracy Comparison')
plt.savefig('validation_accuracy_comparison_all_models.png')
plt.show()







