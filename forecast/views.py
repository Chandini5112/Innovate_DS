from django.shortcuts import render
import requests # This library helps us to fetch data from API
import pandas as pd # This library helps us to handle and manipulate data
import numpy as np # This library helps us to work with numerical operations
from sklearn.model_selection import train_test_split # to split the data into training and testing
from sklearn.preprocessing import LabelEncoder # to convert categorical to numerical values
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # models for Regression and Classification tasks
from sklearn.metrics import mean_squared_error # to evaluate the models
from datetime import datetime, timedelta # to work with date and time
import pytz # to work with timezones
import os
from django.conf import settings

API_KEY = 'ab607ca5523f15c3836ddf61ac2397d6' # Replace with your actual API Key
BASE_URL = 'https://api.openweathermap.org/data/2.5/' #base URL for making API requests

#1.Fetch Current Weather Data
def fetch_current_weather(city):
    url = f'{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric' # Construct the API request url
    response = requests.get(url) # sends the get request to API
    data = response.json()
    return {
        'city' : data['name'],
        'temperature' : round(data['main']['temp']),
        'feels_like' : round(data['main']['feels_like']),
        'temp_min' : round(data['main']['temp_min']),
        'temp_max' : round(data['main']['temp_max']),
        'humidity' : round(data['main']['humidity']),
        'description' : data['weather'][0]['description'],
        'country' : data['sys']['country'],
        'wind_gust_dir' : data['wind']['deg'],
        'pressure' : data['main']['pressure'],
        'Wind_Gust_Speed' : data['wind']['speed'],
        'clouds' : data['clouds']['all'],
        'Visibility' : data['visibility']
    }

#2. Read Historical Data
def read_historical_data(fileName):
  df = pd.read_csv(fileName) # load csv file into dataFrame
  df = df.dropna() # remove rows with missing values
  df = df.drop_duplicates() # remove rows with duplicates
  return df

#3. Prepare Data for Training
def prepare_data(data):
  le = LabelEncoder() # create LabelEncoder instance
  data['WindGustDir'] = le.fit_transform(data['WindGustDir']) # convert categorical to numerical values
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

  #define the feature variables and target variables
  x = data[['MinTemp','MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']] # feature variables
  y = data['RainTomorrow'] # target variable
  return x, y, le # return feature variable, target variable and label encoder

#4. Train Rain prediction model
def train_rain_model(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # split the data into training and testing
  model = RandomForestClassifier(n_estimators=100, random_state=42) # create a random forest classifier
  model.fit(x_train, y_train) # train the model
  y_pred = model.predict(x_test) # make predictions on the test data
  print('Mean Squared Error For Rain Model')
  print(mean_squared_error(y_test, y_pred)) # evaluate the model
  return model # return the trained model

#5. Prepare Regression Data
def prepare_regression_data(data, feature):
  x, y = [], [] # Initialize lists for feature and target values
  for i in range(len(data) - 1):
    x.append(data[feature].iloc[i])
    y.append(data[feature].iloc[i + 1])
  x = np.array(x).reshape(-1, 1)
  y = np.array(y)
  return x, y

#6. Train Regression Model
def train_regression_model(x, y):
  model = RandomForestRegressor(n_estimators=100, random_state=42) # create a random forest regressor
  model.fit(x, y) # train the model
  return model

#7. Predict Future
def predict_future(model, current_value):
  predictions = [current_value]
  for i in range(5):
    next_value = model.predict(np.array([[predictions[-1]]]))
    predictions.append(next_value[0])
  return predictions[1:]

#8. Weather Analysis Function


def weather_view(request):
  if request.method == 'POST':
    city = request.POST.get('city')
    # Store the initial current_weather data
    initial_current_weather = fetch_current_weather(city)

      #load historical data
    file_path = os.path.join('C:\\Users\\chand\\OneDrive\\Documents\\projects\\weatherForecast\\weatherProject\\forecast\\static\\data\\weather.csv')
    historical_data = read_historical_data(file_path)

      #prepare and train the rain prediction model
    x, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(x, y)

      # Map wind directions to compass points using the initial data
    wind_deg = initial_current_weather['wind_gust_dir'] % 360
    compass_points = [
          ('N', 0, 11.25), ('NNE', 11.25, 33.75), ('NE', 33.75, 56.25), ('ENE', 56.25, 78.75),
          ('E', 78.75, 101.25), ('ESE', 101.25, 123.75), ('SE', 123.75, 146.25), ('SSE', 146.25, 168.75), ('S', 168.75, 191.25), ('SSW', 191.25, 213.75), ('SW', 213.75, 236.25), ('WSW', 236.25, 258.75), ('W', 258.75, 281.25), ('WNW', 281.25, 303.75), ('NW', 303.75, 326.25), ('NNW', 326.25, 348.75)
      ]
    compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

      # Create a new dictionary for the rain prediction model
    current_weather_for_rain_model = {
    'MinTemp' : initial_current_weather['temp_min'],
    'MaxTemp' : initial_current_weather['temp_max'],
    'WindGustDir' : compass_direction_encoded,
    'WindGustSpeed' : initial_current_weather['Wind_Gust_Speed'],
    'Humidity' : initial_current_weather['humidity'],
    'Pressure' : initial_current_weather['pressure'],
    'Temp' : initial_current_weather['temperature'],
}

    current_df = pd.DataFrame([current_weather_for_rain_model])

      # Rain Prediction
    rain_prediction = rain_model.predict(current_df)[0]

      #prepare regression model for temperature and humidity
    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    x_humidity, y_humidity = prepare_regression_data(historical_data, 'Humidity')

    temp_model = train_regression_model(x_temp, y_temp)
    humidity_model = train_regression_model(x_humidity, y_humidity)

      # Predict future temperature and humidity using values from the initial data
    future_temp = predict_future(temp_model, initial_current_weather['temperature'])
    future_humidity = predict_future(humidity_model, initial_current_weather['humidity'])

      # prepare time for future predictions
    timezone = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(timezone)
    next_hour = current_time + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours = i)).strftime("%H:00") for i in range(5)]
    forecast = list(zip(future_times, future_temp, future_humidity))

    #pass data to template
    context = {
        'location' : city, 
        'temperature' : initial_current_weather['temperature'],
        'temp_min' : initial_current_weather['temp_min'],
        'temp_max' : initial_current_weather['temp_max'],
        'feels_like' : initial_current_weather['feels_like'],
        'humidity' : initial_current_weather['humidity'],
        'clouds' : initial_current_weather['clouds'],
        'description' : initial_current_weather['description'],
        'city' : initial_current_weather['city'],
        'country' : initial_current_weather['country'],
        'time' : datetime.now(),
        'date' : datetime.now().strftime("%B %d, %Y"),
        'wind' : initial_current_weather['Wind_Gust_Speed'],
        'pressure' : initial_current_weather['pressure'],
        'Visibility' : initial_current_weather['Visibility'],
        'forecast': forecast,
      }
    return render(request, 'weather.html', context)
  return render(request, 'weather.html')
