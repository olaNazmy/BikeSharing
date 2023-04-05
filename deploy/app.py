from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np 
from datetime import datetime


app = FastAPI()

scaler = joblib.load('C:/Users/asd/Python_/Bikes/deploy/models/scaler.h5')
model = joblib.load('C:/Users/asd/Python_/Bikes/deploy/models/model.h5')


@app.get('/')
def index() :
    return 'Hello, Ola Nazmy, Bikes Project'


#@app.get('/predict/{Hour}') 
#def predict_bikes(Hour: int) :
#   return {'hour' : Hour}

## to send data from the client
class InputData(BaseModel) :
    season : int
    hr : int
    holiday : int
    weekday : int
    workingday : int
    weathersit : float 
    temp : float 
    atemp : float
    hum : float
    windspeed : float
    Year : int
    Month : int
    Day : int
    
    #Take Data From User
@app.post('/predict') 
def predict_bikes(input_data : InputData) :
    Season = input_data.season
    Hour = input_data.hr        
    Holiday = input_data.holiday
    Weekday = input_data.weekday
    Workingday = input_data.workingday
    Weatherist = input_data.weathersit
    Temperature = input_data.temp 
    Atemp = input_data.atemp
    Humidity = input_data.hum
    Wind_speed = input_data.windspeed
    Year = input_data.Year
    Month = input_data.Month
    Day = input_data.Day
    
    #return {'Hour' : Hour} # just for test
    ## data preprocessing 
    scaled_data = scaler.transform(np.array([[ Temperature, Humidity, Wind_speed, Atemp,Weatherist]]))

    input_vector = [Hour] + list(scaled_data[0]) + [Holiday, Workingday, Month, Weekday,Season,Year,Month] 
    prediction = model.predict(np.array([input_vector]))
    
    return {'Bikes Number' : int(prediction)}








    