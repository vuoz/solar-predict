import polars
from model import Model,LstmModel
import torch
import requests
from dataloader import Coordinates, DataframeWithWeatherAsDict,Dataloader;
from datetime import datetime
import os
import dotenv
import matplotlib.pyplot as plt;
import pandas as pd;

def get_weather_data(day:str,cords:Coordinates,historical:bool)->tuple[DataframeWithWeatherAsDict|None,Exception|None]:
    
    url = ""
    if historical:
        url = "https://archive-api.open-meteo.com/v1/archive";
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    params = {
	        "latitude": cords.latitude,
	        "longitude": cords.longitude,
	        "start_date": day,
	        "end_date": day,
	        "hourly": ["temperature_2m", "precipitation", "cloud_cover", "sunshine_duration","global_tilted_irradiance","relative_humidity_2m", "wind_speed_10m"],
	        "daily": ["sunrise", "sunset", "sunshine_duration"],
	        "timezone": "Europe/Berlin"
        }
    resp =requests.get(url,params=params)
    if resp.status_code != 200:
        return (None,Exception(f"Failed to get weather data for window {day}, error: {resp.status_code}"))
    data = resp.json()
    data_struct = DataframeWithWeatherAsDict(df=polars.DataFrame(),weather=data)
    return (data_struct,None)

def inference_mlp(date:str,default_date:str,model_path:str):

    if date != "" and (len(date) != 10 or datetime.strptime(date,"%Y-%m-%d") == None):
        print("Invalid date format")
        exit(1)
    if date == "":
        date = default_date

    model = Model(input_size=24*8)
    model.load_state_dict(torch.load(model_path))

    date_to_check = datetime.strptime(date,"%Y-%m-%d")
    curr_date = datetime.now()
    historical = False
    if date_to_check < curr_date:
        historical = True
    else:
        historical = False
 
    weather,err = get_weather_data(date,Coordinates(float(os.environ["Lat"]),float(os.environ["Long"])),historical)
    if err != None:
        print(err)
        exit()
    if weather == None:
        print("Could not get weather")
        exit()
    print(weather.weather)
    train_data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()
    lable = None
    for datapoint in train_data:
        if str(datapoint.df.get_column("Date")[0]) == date:
            lable  = datapoint.df_to_lable_normalized()

    if lable == None:
        print("No lable found")
        exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = weather.weather_to_feature_vec().to(device).flatten()
    output = model(input)
    output_tensor = torch.Tensor(output)
    np_output = output_tensor.cpu().detach().numpy()
    sum_nn =   (np_output * 0.0833).sum() 
    sum_lable =   (lable * 0.0833).sum() 
    print(f"Sum of NN output: {sum_nn} kWh",f"Sum of Lable: {sum_lable} kWh")
    

    #Plot the output in comparison to the lable for a specified day
    time_points = pd.date_range(start="00:00", end="23:55", freq="5min")
    plt.figure(figsize=(10,5))
    plt.scatter(time_points,np_output, label='NN Output', color='red', marker='o')  
    plt.scatter(time_points,lable, label='Label Data', color='green', marker='o')  

    plt.title('Neural Network Output and Label Data Overlay')
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Production [kW]')
    plt.legend()
    plt.show()

def inference_lstm(date:str,default_date:str):
    if date != "" and (len(date) != 10 or datetime.strptime(date,"%Y-%m-%d") == None):
        print("Invalid date format")
        exit(1)
    if date == "":
        date = default_date

    model = LstmModel()
    model.load_state_dict(torch.load("model_lstm.pth"))

    date_to_check = datetime.strptime(date,"%Y-%m-%d")
    curr_date = datetime.now()
    historical = False
    if date_to_check < curr_date:
        historical = True
    else:
        historical = False
    weather,err = get_weather_data(date,Coordinates(float(os.environ["Lat"]),float(os.environ["Long"])),historical)
    if err != None:
        print(err)
        exit(1)
    if weather == None:
        print("Could not get weather")
        exit(1)
    train_data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()
    lable = None
    for datapoint in train_data:
        if str(datapoint.df.get_column("Date")[0]) == date:
            lable  = datapoint.to_lable_normalized_hours_accurate()
            # concat the individual tensors to one root tensor to be abled to plot it
            lable = lable.flatten()
    if lable == None:
        print("No lable found")
        exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = weather.weather_to_feature_vec().to(device)
    output_tensor = torch.Tensor().to(device)
    prev_out = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0])
    for hour in input:
        print("this is output tensor",output_tensor)
        output_hour = model(hour.to(device),prev_out.to(device))   
        prev_out = output_hour
        print(output_hour)
        output_tensor = torch.cat((output_tensor.to(device),output_hour.to(device)),dim=0)
    np_output = output_tensor.cpu().detach().numpy()
    sum_nn =   (np_output * 0.0833).sum() 
    sum_lable =   (lable * 0.0833).sum() 
    print(f"Sum of NN output: {sum_nn} kWh",f"Sum of Lable: {sum_lable} kWh")
    

    #Plot the output in comparison to the lable for a specified day
    time_points = pd.date_range(start="00:00", end="23:55", freq="5min")
    plt.figure(figsize=(10,5))
    plt.scatter(time_points,np_output, label='NN Output', color='red', marker='o')  
    plt.scatter(time_points,lable, label='Label Data', color='green', marker='o')  

    plt.title('Neural Network Output and Label Data Overlay')
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Production [kW]')
    plt.legend()
    plt.show()


# this is just the version that of inference that is used to test the actual ability of the model
# once the model works, i will add a real and abstraced version of a inference function/ class that can be used to acutally run the model and execute predictions once the model has the required accuracy
if __name__ == "__main__":
    dotenv.load_dotenv()
    default_date = "2024-08-22"
    print(f"Please provide a date in the following format: YYYY-MM-DD, Default is {default_date}")
    date = input()
    inference_mlp(date,default_date,"models/summer.pth")

