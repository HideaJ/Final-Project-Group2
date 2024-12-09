
def scale_group(group):
    features = ['revenue', 'grossProfit', 'operatingIncome', 'netIncome_x',
                'cashAndCashEquivalents', 'totalAssets', 'totalLiabilities', 'adjClose']
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import AdamW

class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(1,),
                                 initializer='zeros', trainable=True)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def build_lstm_attention_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(5, 8)))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

lstm_attention_model = build_lstm_attention_model()
optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
lstm_attention_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = lstm_attention_model.fit(dataset, epochs=50, validation_data=(X_test, y_test),
                                   shuffle=False, callbacks=[reduce_lr, early_stopping])
import matplotlib.pyplot as plt

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('LSTM-Attention Validation Accuracy')
plt.legend()
plt.savefig('lstm_attention_validation_accuracy.png')
plt.show()

