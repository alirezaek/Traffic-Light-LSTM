
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)


dataframe = pd.read_csv('DataSet-1/dataset1-all.csv',header=None, engine='python')
dataset = dataframe[[0,1,2,3,4,5,6,7]].values
dataset = dataset.astype('float32')
print("datasetin ",dataset.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print("trainx " , trainX)

model = Sequential()
model.add(LSTM(8,return_sequences=True, input_shape=(look_back,8)))
model.add(LSTM(8))
model.add(Dense(8))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=512)
model.save('lstm_model1state.h5')
# model = load_model('lstm_model1state.h5')

trainPredict = model.predict(trainX)
print ("trainpredict ",trainPredict)

testPredict = model.predict(testX)
eval = model.evaluate(testX,testY)
print("evaluate ",eval)
print("trainY ",trainY)


trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score normal: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score normal: %.2f RMSE' % (testScore))

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY[:,:], trainPredict[:,:]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


testPredictPlot = testPredict[0:100,0]
testYPlot = testY[0:100,0]
plt.plot(testPredictPlot)
plt.plot(testYPlot)
plt.show()