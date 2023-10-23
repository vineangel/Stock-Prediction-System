
from django.shortcuts import render
from django.db.models import Max

import pandas as pd


from datetime import datetime
from django.utils import timezone
from django.shortcuts import render
import pandas_datareader as pdr
import numpy as np
import base64
import matplotlib.pyplot as plt
from io import BytesIO

def generate_stock_prediction_plot(stock_symbol,future_days):
  
#............. Data Collection...............
    symbol=stock_symbol
    # Initialize pandas-datareader
    import yfinance as yf

    # Get the current date
    current_date = datetime.now().date().strftime('%Y-%m-%d')
    current_time = timezone.now()
    print("current date : " + str(current_date))
    print("current time : " + str(current_time))
   
    # Fetch the dataset from Yahoo Finance with the current date as the end date
    df = yf.download(symbol, start="2022-01-01", end=current_date)
    print(df.columns)
    print(df)

    # df=pd.read_csv('AAPL.csv')
#............. Data Preprocessing...............
    df1=df.reset_index()['Close']
    # df1=df.set_index('date', inplace=True)['close']
    import matplotlib.pyplot as plt
    plt.plot(df1)
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

#..........LSTM ...........
    import numpy
    ##takes a time series dataset and creates input-output pairs
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        ##ensure that enough data points are available for creating input-output pairs.
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)
        
    time_step = 100
    ##X_train contains a numpy.array(dataX[])
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    ## reshaping your data to have a single feature at each time step.
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    import tensorflow as tf
    from tensorflow import keras

    # Load the model from the HDF5 file
    model = keras.models.load_model("stock.h5")


    train_predict = model.predict(X_train)
    test_predict=model.predict(X_test)
    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    ### Plotting
    # shift train predictions for plotting
    #look_back = timestep
    look_back=100
    #empty_like() will create a new array with the same shape and data type as df1.
    #create a new NumPy array that has the same shape and data type as the input array df1
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df1),label='Actual Prices', color='orange')
    # plt.plot(trainPredictPlot,label='train prediction Prices', color='green')
    # plt.plot(testPredictPlot,label='test prediction Prices', color='yellow')
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title("  Stock Price ")
    plt.legend()
  
    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    # plt.show()
    # y1_min, y1_max = plt.ylim()
    plt.close()

    # Convert the plot to base64 for embedding in HTML
    plot_dataa = base64.b64encode(buffer.getvalue()).decode('utf-8')

    len(test_data)
    #len = 441
    print("len(test_data)")
    print(len(test_data))
    x=len(test_data)-100
    x_input=test_data[x:].reshape(1,-1)
    x_input.shape
    #reshaping it into 2d array
    #test data predicted output green, blue complete data set , training data prediction orange
    temp_input=list(x_input)
    #converts the NumPy array x_input into a Python list
    temp_input=temp_input[0].tolist()
    #accessing first element

    # Initialize the list to store predictions
    lst_output = []

    # Use the last 'time_step' data points from the test data as the initial input
    x_input = test_data[-time_step:].reshape(1, -1)  # Assuming 'time_step' is the same as used during training

    for i in range(future_days):
        if len(temp_input) >= time_step:
            x_input = np.array(temp_input[-time_step:]).reshape(1, -1, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
        else:
            # If 'temp_input' doesn't have enough data points, you can add some default or random values.
            # For example, you can add zeros:
            x_input = np.zeros((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())

#..........Error calculation...........
    import math
    from sklearn.metrics import mean_squared_error
    trainerror=math.sqrt(mean_squared_error(y_train,train_predict))
    testerror=math.sqrt(mean_squared_error(y_test,test_predict))
    print("trainerror : "+ str(trainerror))
    print("testerror : "+ str(testerror))
#..........Error calculation ends...........
    print(lst_output)
    # 'lst_output' now contains predictions for the next 'future_days' days based on user input
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    day_new = np.arange(len(df1) - future_days, len(df1))
    day_pred = np.arange(len(df1), len(df1) + future_days)
    # The scale of the x-axis and y-axis in a matplotlib plot is determined by the data and the figure size, and it's typically calculated automatically by the library to fit the data within the specified figure size. change be changed using plt.ylim(y1_min, y1_max)
    # Generate the plot
    plt.plot(day_new, scaler.inverse_transform(df1[-future_days:]), label='Actual Prices', color='blue')
    plt.plot(day_pred,  scaler.inverse_transform(lst_output), label='Predicted Prices', color='green')
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title("Future Stock Price Prediction")
    plt.legend()
    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    # plt.show()
    plt.close()
    # Convert the plot to base64 for embedding in HTML
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

#..........LSTM ends...........

    return plot_data,plot_dataa


def index(request):
    if request.method == 'POST':
        stock_symbol = request.POST.get('stock_symbol', '') 
        future_days = int(request.POST.get('future_days', 30))  # Default to 30 days if no input
       
        plot_data,plot_dataa= generate_stock_prediction_plot(stock_symbol,future_days)
        
        context = { 'stock_symbol': stock_symbol,  'plot_data': plot_data,'plot_dataa': plot_dataa, 'future_days': future_days}
    else:
        context = {'qinfo': None,'plot_data': None,'plot_dataa': None, 'future_days': None}

    return render(request, './polls/index.html', context)