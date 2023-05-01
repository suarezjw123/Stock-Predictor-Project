#!/usr/bin/env python
# coding: utf-8

def lstm_function(ticker):
    # adding modules
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import yfinance as yf
    import pandas_ta as ta
    data = yf.download(tickers = ticker, start = '2012-03-11',end = '2022-07-10')
    data.head(10)
    # Adding indicators

    data['RSI']=ta.rsi(data.Close, length=15)
    
    data['EMAS']=ta.ema(data.Close, length=150)

    data['SMAF']=ta.sma(data.Close, length=20)

    data['ATR']=ta.atr(data.High, data.Low, data.Close, length=14)
    data['ATR5']=ta.atr(data.High, data.Low, data.Close, length=5)
    data['ATR10']=ta.atr(data.High, data.Low, data.Close, length=10)
    data['ATR20']=ta.atr(data.High, data.Low, data.Close, length=20)

    data['CCI']=ta.cci(data.High, data.Low, data.Close, length=14)

    data['OBV']=ta.obv(data.Close, data.Volume)

    data['PVI']=ta.pvi(data.Close, data.Volume)
    data['PVIEMA']=ta.ema(data.PVI, length=20)

    # list of target variables (we are using 'Target' as our target variable)
    data['Target'] = data['Adj Close']-data.Open
    data['Target'] = data['Target'].shift(-1)

    data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

    data['TargetNextClose'] = data['Adj Close'].shift(-1)

    data.dropna(inplace=True)
    data.reset_index(inplace = True)
    data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)


    data_set = data.iloc[:, 0:11]#.values
    pd.set_option('display.max_columns', None)

    data_set.head(20)

    # Applying a scaler using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(data_set)

    # multiple feature from data provided to the model
    
    # the input data will be stored in X
    X = []

    # backcandles are the number of previous days in the data
    backcandles = 30
    print(data_set_scaled.shape[0])
    for j in range(8):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i-backcandles:i, j])

    #move axis from 0 to position 2
    X=np.moveaxis(X, [0], [2])

    #the target data will be stored in y
    X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-3])
    y=np.reshape(yi,(len(yi),1))

    #print(X)
    #print(X.shape)
    #print(y)
    #print(y.shape)

    # split data into train test sets
    splitlimit = int(len(X)*0.8)        
    #print(splitlimit)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)
    #print(y_train)


    #Adding different Machine Learning modules
    from keras.layers import Dropout
    from keras.layers import Dense

    import tensorflow as tf
    from keras import optimizers
    from keras.models import Model
    from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
    import numpy as np
    np.random.seed(10)

    from keras.layers import Conv1D, MaxPooling1D, Bidirectional

    lstm_input = Input(shape=(backcandles, 8), name='lstm_input')   
    inputs = Conv1D(32, 3, activation='relu', padding='same')(lstm_input) #Added
    inputs = MaxPooling1D(pool_size=2)(inputs) #Added
    inputs = Bidirectional(LSTM(64))(inputs) #Changed
    inputs = Dense(64, name='dense_layer')(inputs)
    inputs = Activation('relu')(inputs)
    inputs = Dropout(0.15)(inputs)
    output = Dense(1, activation='linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)


    # y_pred is the prediction data
    # y_test is the testing data
    
    y_pred = model.predict(X_test)
    #y_pred=np.where(y_pred > 0.43, 1,0)
    for i in range(10):
        return y_test, y_pred

