import pandas as pd
import numpy as np
from keras.layers import Dense, Input, LSTM
from keras.models import Sequential
from keras.backend import clear_session
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from DesertSeeker import Study
import warnings
warnings.filterwarnings("ignore")

EPOCHS = 100
BATCH_SIZE = 64
def objective(trial):
    # read dataset
    df= pd.read_csv("Tetuan City power consumption.csv")
    df= df.head(2500)
    
    # split data
    X= df.drop(columns= ['Zone 3  Power Consumption'])
    y= df[['Zone 3  Power Consumption']]
    
    # create timestamp
    X['DateTime'] = pd.to_datetime(X['DateTime'])
    X['timestamp'] = X['DateTime'].astype('int')
    X = X.set_index(['DateTime'])
    
    # normalizing data
    scaler = PowerTransformer(method='box-cox')
    X['normed_general diffuse flows'] = scaler.fit_transform(X[['general diffuse flows']])
    X['normed_diffuse flows'] = scaler.fit_transform(X[['diffuse flows']])
    X['normed_humidity'] = np.clip(X['Humidity'], a_min=40, a_max=90)
    X = X.drop(['Humidity', 'general diffuse flows', 'diffuse flows'], axis=1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)
    
    X , y = create_sequences(X_scaled, y_scaled, window_size=144)
    
    # split data
    train_size = int(len(X) * 0.8)  # Use 80% of the data for training
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    validation_size = int(len(X_test) * 0.5) # split test to test and validation
    X_valid, X_test = X_test[:validation_size], X_test[validation_size:]
    y_valid, y_test = y_test[:validation_size], y_test[validation_size:]
    
    # train model
    clear_session()

    # Optimize the number of layers, number of neurons and weight
    n_layer = trial.suggest_int('layers', 1, 3)
    weight_decay = trial.suggest_float('weight decay', 1e-3, 1e-10)

    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))
    model.add(LSTM(13, activation='relu', return_sequences=True))
    for i in range(n_layer):
        num_neurons = trial.suggest_int(f'neuron {i}', 8, 16)
        model.add(LSTM(
            num_neurons,
            activation='relu',
            return_sequences=True,
            kernel_regularizer=l2(weight_decay)
        ))
    model.add(Dense(1, activation='swish'))

    learning_rate = trial.suggest_float("lr", 1e-1, 1e-5)
    
    model.compile(
        loss='mae',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['mean_absolute_percentage_error']
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        shuffle=False,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose= False
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[0]
    
def create_sequences(X, y, window_size):
    seq_X = []
    seq_y = []
    for i in range(len(X) - window_size):
        seq_X.append(X[i:i+window_size])
        seq_y.append(y[i+window_size])

    return np.array(seq_X), np.array(seq_y)

if __name__ == "__main__":
    study= Study(pop_size=5, iteration=10)
    values, score= study.optimize(objective)
    