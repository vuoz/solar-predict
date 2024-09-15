import polars
from scipy.sparse import data
from customTypes import MinMaxWeather
from model import Model,LstmModel
import torch
import requests
from dataloader import Coordinates, DataframeWithWeatherAsDict,Dataloader, split_dfs_by_season,min_max_normalize;
from datetime import datetime
import os
import dotenv
import matplotlib.pyplot as plt;
import pandas as pd;
def reverse_min_max(normalized_values, original_min, original_max):
    return [value * (original_max - original_min) + original_min for value in normalized_values]

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
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m", "sunshine_duration", "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance", "diffuse_radiation_instant", "direct_normal_irradiance_instant", "global_tilted_irradiance_instant"],
	        "daily": ["sunrise", "sunset", "sunshine_duration"],
	        "timezone": "Europe/Berlin"
        }
    resp =requests.get(url,params=params)
    if resp.status_code != 200:
        return (None,Exception(f"Failed to get weather data for window {day}, error: {resp.status_code}"))
    data = resp.json()
    data_struct = DataframeWithWeatherAsDict(df=polars.DataFrame(),weather=data)
    return (data_struct,None)
def normalize_weather_create_inference_vec(min_max:MinMaxWeather,day)->torch.Tensor:
    percipitation_normalized = min_max_normalize([x for x in day.weather["hourly"]["precipitation"]],min_max.percipitation[1],0)
    temp_2m_normalized = min_max_normalize([x for x in day.weather["hourly"]["temperature_2m"]],min_max.temp[1],0)
    cc_normalized = min_max_normalize([x for x in day.weather["hourly"]["cloud_cover"]],min_max.cloud_cover[1],0)
    irradiance_normalized = min_max_normalize([x for x in day.weather["hourly"]["global_tilted_irradiance"]],min_max.irradiance[1],0)
    wind_normalized = min_max_normalize([x for x in day.weather["hourly"]["wind_speed_10m"]],min_max.wind_speed[1],0)
    humidity_normalized = min_max_normalize([x for x in day.weather["hourly"]["relative_humidity_2m"]],min_max.humidity[1],0)
    diffuse_radiation_normalized = min_max_normalize([x for x in day.weather["hourly"]["diffuse_radiation"]],min_max.diffuse_radiation[1],0)
    direct_normal_irradiance_normalized = min_max_normalize([x for x in day.weather["hourly"]["direct_normal_irradiance"]],min_max.direct_normal_irradiance[1],0)
    diffuse_radiation_instant_normalized = min_max_normalize([x for x in day.weather["hourly"]["diffuse_radiation_instant"]],min_max.diffuse_radiation_instant[1],0)
    direct_normal_irradiance_instant_normalized = min_max_normalize([x for x in day.weather["hourly"]["direct_normal_irradiance_instant"]],min_max.direct_normal_irradiance_instant[1],0)
    global_tilted_instant_normalized = min_max_normalize([x for x in day.weather["hourly"]["global_tilted_irradiance_instant"]],min_max.global_tilted_irradiance_instant[1],0)


    day.weather["hourly"]["precipitation"] = percipitation_normalized
    day.weather["hourly"]["temperature_2m"] = temp_2m_normalized
    day.weather["hourly"]["cloud_cover"] = cc_normalized
    day.weather["hourly"]["global_tilted_irradiance"] = irradiance_normalized
    day.weather["hourly"]["wind_speed_10m"] = wind_normalized
    day.weather["hourly"]["relative_humidity_2m"] = humidity_normalized
    day.weather["hourly"]["diffuse_radiation"] = diffuse_radiation_normalized
    day.weather["hourly"]["direct_normal_irradiance"] = direct_normal_irradiance_normalized
    day.weather["hourly"]["diffuse_radiation_instant"] = diffuse_radiation_instant_normalized
    day.weather["hourly"]["direct_normal_irradiance_instant"] = direct_normal_irradiance_instant_normalized
    day.weather["hourly"]["global_tilted_irradiance_instant"] = global_tilted_instant_normalized
    return day.weather_to_feature_vec()



def inference_mlp(date:str,default_date:str):
    if date != "" and (len(date) != 10 or datetime.strptime(date,"%Y-%m-%d") == None):
        print("Invalid date format")
        exit(1)
    if date == "":
        date = default_date
    model_pth = ""
    season = ""
    date_datetime = datetime.strptime(date,"%Y-%m-%d")
    if date_datetime.month in [12,1,2]:
        model_pth = "models/winter.pth"
        season = "winter"
    elif date_datetime.month in [3,4,5]:
        model_pth = "models/spring.pth"
        season = "spring"
    elif date_datetime.month in [6,7,8]:
        model_pth = "models/summer.pth"
        season = "summer"
    elif date_datetime.month in [9,10,11]:
        model_pth = "models/autumn.pth"
        season = "autumn"
    else:
        print("Invalid date")
        exit(1)

    print(f"Using model: {model_pth}")


    model = Model(input_size=24*13)
    model.load_state_dict(torch.load(model_pth))

    date_to_check = datetime.strptime(date,"%Y-%m-%d")
    curr_date = datetime.now()
    historical = False
    if date_to_check < curr_date:
        print("Using historical data")
        historical = True
    else:
        historical = False
 
    # need to normalized this. to the same scale as in training
    weather,err = get_weather_data(date,Coordinates(float(os.environ["Lat"]),float(os.environ["Long"])),historical)
    if err != None:
        print(err)
        exit()
    if weather == None:
        print("Could not get weather")
        exit()

    train_data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()
    seasons = split_dfs_by_season(train_data)
    min_max_seasons = seasons.normalize_seasons()
    scaling_values = None
    if season == "winter":
        scaling_values = min_max_seasons.winter
    elif season == "summer":
        scaling_values = min_max_seasons.summer
    elif season == "spring":
        scaling_values = min_max_seasons.spring
    elif season == "autumn":
        scaling_values = min_max_seasons.autumn
    if scaling_values == None:
        print("Error determining season")
        exit()


    
    

    data = seasons.get_data_by_date(date)
    if data == None:
        print("Date not found in data")
        exit()
    label = data.to_lable_normalized_and_smoothed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if weather == None:
        print("Weather not found")
        exit()

    
    input = normalize_weather_create_inference_vec(scaling_values,weather).to(device).flatten()
    input_training = data.weather_to_feature_vec().to(device).flatten()

    output = model(input)
    output_tensor = torch.Tensor(output).to(device)

    # applying rolling mean to nn output to smooth the curve
    windows = output_tensor.unfold(dimension=0, size=12, step=1).to(device)
    rolling_mean = windows.mean(dim=1)
    padding_value = rolling_mean[0].item()
    padding = torch.full((11,),padding_value).to(device)
    rolling_mean_padded = torch.cat((padding,rolling_mean),dim=0).to(device)

    np_output = rolling_mean_padded.cpu().detach().numpy()
    sum_nn =   (np_output * 0.0833).sum() 
    sum_lable =   (label * 0.0833).sum() 

    print(f"Sum of NN output: {sum_nn} kWh",f"Sum of Lable: {sum_lable} kWh")

    label_broadcasted = reverse_min_max(label,scaling_values.power_production[0],scaling_values.power_production[1])
    nn_output_broadcasted = reverse_min_max(rolling_mean_padded,scaling_values.power_production[0],scaling_values.power_production[1])
    nn_out_broadcasted = torch.tensor(nn_output_broadcasted)
    label_broadcasted = torch.tensor(label_broadcasted)
    


    

    #Plot the output in comparison to the lable for a specified day
    time_points = pd.date_range(start="00:00", end="23:55", freq="5min")
    plt.figure(figsize=(10,5))
    plt.scatter(time_points,nn_out_broadcasted, label='NN Output', color='red', marker='o')  
    plt.scatter(time_points,label_broadcasted, label='Label Data', color='green', marker='o')  

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
    model_pth = ""
    season = ""
    date_datetime = datetime.strptime(date,"%Y-%m-%d")
    if date_datetime.month in [12,1,2]:
        model_pth = "models/lstm_winter.pth"
        season = "winter"
    elif date_datetime.month in [3,4,5]:
        model_pth = "models/lstm_spring.pth"
        season = "spring"
    elif date_datetime.month in [6,7,8]:
        model_pth = "models/lstm_summer.pth"
        season = "summer"
    elif date_datetime.month in [9,10,11]:
        model_pth = "models/lstm_autumn.pth"
        season = "autumn"
    else:
        print("Invalid date")
        exit(1)

    print(f"Using model: {model_pth}")



    model = LstmModel()
    model.load_state_dict(torch.load(model_pth))

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
    # still need to normalized the values
    train_data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()
    seasonal_data = split_dfs_by_season(train_data)
    min_max_seasons =seasonal_data.normalize_seasons()
    scaling_values = None
    if season == "winter":
        scaling_values = min_max_seasons.winter
    elif season == "summer":
        scaling_values = min_max_seasons.summer
    elif season == "spring":
        scaling_values = min_max_seasons.spring
    elif season == "autumn":
        scaling_values = min_max_seasons.autumn
    if scaling_values == None:
        print("Error dertiming season")
        exit()


    data_for_date = seasonal_data.get_data_by_date(date)
    if data_for_date == None:
        print("Date not found in data")
        exit()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    weather = data_for_date
    input = weather.weather_to_feature_vec().to(device)
    output_tensor = model(input)
  
    # applying rolling mean to nn output to smooth the curve
    windows = output_tensor.flatten().unfold(dimension=0, size=12, step=1).to(device)
    rolling_mean = windows.mean(dim=1)
    padding_value = rolling_mean[0].item()
    padding = torch.full((11,),padding_value).to(device)
    rolling_mean_padded = torch.cat((padding,rolling_mean),dim=0).to(device)


    np_output = rolling_mean_padded.cpu().detach().numpy()
    sum_nn =   (np_output * 0.0833).sum() 
    sum_lable =   (data_for_date.to_lable_normalized_smoothed_and_hours_accurate().flatten() * 0.0833).sum() 
    print(f"Sum of NN output: {sum_nn} kWh",f"Sum of Lable: {sum_lable} kWh")

    label_broadcasted = reverse_min_max(data_for_date.to_lable_normalized_smoothed_and_hours_accurate().flatten(),scaling_values.power_production[0],scaling_values.power_production[1])
    nn_output_broadcasted = reverse_min_max(rolling_mean_padded,scaling_values.power_production[0],scaling_values.power_production[1])
    nn_out_broadcasted = torch.tensor(nn_output_broadcasted)
    label_broadcasted = torch.tensor(label_broadcasted)
    

    #Plot the output in comparison to the lable for a specified day
    time_points = pd.date_range(start="00:00", end="23:55", freq="5min")
    plt.figure(figsize=(10,5))
    plt.scatter(time_points,nn_out_broadcasted, label='NN Output', color='red', marker='o')  
    plt.scatter(time_points,label_broadcasted, label='Label Data', color='green', marker='o')  

    plt.title('Neural Network Output and Label Data Overlay')
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Production [kW]')
    plt.legend()
    plt.show()


# this is just the version that of inference that is used to test the actual ability of the model
# once the model works, i will add a real and abstraced version of a inference function/ class that can be used to acutally run the model and execute predictions once the model has the required accuracy
if __name__ == "__main__":
    dotenv.load_dotenv()
    default_date = "2024-06-05"
    print(f"Please provide a date in the following format: YYYY-MM-DD, Default is {default_date}")
    date = input()
    inference_mlp(date,default_date)
