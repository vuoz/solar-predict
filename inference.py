import polars
from customTypes import MinMaxWeather
from model import LstmModel
import torch
import requests
from dataloader import Coordinates, DataframeWithWeatherAsDict,Dataloader, split_dfs_by_season,min_max_normalize;
from datetime import datetime
import os
import dotenv
import matplotlib.pyplot as plt;
import pandas as pd;

# function to reverse the min_max scaling down during data preprocessing
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

# this function is used to normalize input data to the neural network, to the level it experienced during training 
# if we don't do this we loose out on accurary
# this essentially takes the min_max values for each feature and then normalizes the input data to the same scale
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

    # assign to the now normalized data
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


def inference_lstm(date:str,default_date:str):

    #check date format
    if date != "" and (len(date) != 10 or datetime.strptime(date,"%Y-%m-%d") == None):
        print("Invalid date format")
        exit(1)

    # check if the user provided a date
    if date == "":
        date = default_date
    model_pth = ""
    season = ""

    # determin the season and therby which model to use
    date_datetime = datetime.strptime(date,"%Y-%m-%d")
    if date_datetime.month in [12,1,2]:
        model_pth = "models/lstm_new_winter.pth"
        season = "winter"
    elif date_datetime.month in [3,4,5]:
        model_pth = "models/lstm_new_spring.pth"
        season = "spring"
    elif date_datetime.month in [6,7,8]:
        model_pth = "models/lstm_new_summer.pth"
        season = "summer"
    elif date_datetime.month in [9,10,11]:
        model_pth = "models/lstm_new_autumn.pth"
        season = "autumn"
    else:
        print("Invalid date")
        exit(1)

    print(f"Using model: {model_pth}")



    # initialize the model
    model = LstmModel()
    model.load_state_dict(torch.load(model_pth))

    # determin wether we have to use the historical api or the forecast api
    date_to_check = datetime.strptime(date,"%Y-%m-%d")
    curr_date = datetime.now()
    historical = False
    if date_to_check < curr_date:
        historical = True
    else:
        historical = False

    # get the input data for the date
    weather,err = get_weather_data(date,Coordinates(float(os.environ["Lat"]),float(os.environ["Long"])),historical)
    if err != None:
        print(err)
        exit(1)
    if weather == None:
        print("Could not get weather")
        exit(1)

    # we still get the dataloader,since we need to obtain the min_max values for the inputdata
    train_data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()
    seasonal_data = split_dfs_by_season(train_data)
    min_max_seasons =seasonal_data.normalize_seasons()
    scaling_values = None
    # determin which min_max values to use for downscaling
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

    # this is just for comparison later
    data_for_date = seasonal_data.get_data_by_date(date)
    if data_for_date == None:
        print("Found no data to compare")
    
    # get the cuda device, if on is available and move the model to that device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # transform the input data to a feature vec
    input = weather.weather_to_feature_vec().to(device)

    # add artificial batch dimension
    input_reshaped = input.unsqueeze(0)


    # split the data by hours
    splits = torch.split(input_reshaped.to(device),split_size_or_sections=1,dim=1)

    # create the output tensor within which the outputs will be aggregated
    output_tensor = torch.Tensor().to(device)
    past_out = []
    for i,split_x  in enumerate(splits):



        # create the weather window
        weather_window  = torch.tensor([]).to(device)
        fourth_to_last = splits[i-4].to(device)
        third_to_last = splits[i-3].to(device)
        second_to_last = splits[i-2].to(device)
        previous = splits[i-1].to(device)
        if fourth_to_last is not None:
            weather_window = torch.cat((weather_window,fourth_to_last),dim=1)
        else:
            weather_window = torch.cat((weather_window,torch.zeros(split_x.shape).to(device)),dim=1)
        if third_to_last is not None:
            weather_window = torch.cat((weather_window,third_to_last),dim=1)
        else:
            weather_window = torch.cat((weather_window,torch.zeros(split_x.shape).to(device)),dim=1)
        if second_to_last is not None:
            weather_window = torch.cat((weather_window,second_to_last),dim=1)
        else:
            weather_window = torch.cat((weather_window,torch.zeros(split_x.shape).to(device)),dim=1)
        if previous is not None:
            weather_window = torch.cat((weather_window,previous),dim=1)
        else:
            weather_window = torch.cat((weather_window,torch.zeros(split_x.shape).to(device)),dim=1)

        weather_window = torch.cat((weather_window,split_x),dim=1).to(device)
        try:
            in_advance = splits[i+1]
            weather_window = torch.cat((weather_window,in_advance),dim=1).to(device)
        except IndexError:
            weather_window = torch.cat((weather_window,torch.zeros(split_x.shape).to(device)),dim=1).to(device)




        # create the past output window
        past_window = torch.tensor([])
        def reshape(x):
            return x.view(x.size(0), -1)
        if len(past_out) >= 5:
            fourth_to_last = past_out[-4]
            past_window = torch.cat((past_window.to(device),reshape(fourth_to_last.to(device))),dim=1)
        else:
            past_window = torch.cat((past_window.to(device),reshape(torch.zeros([1,1,12]).to(device))),dim=1)
        if len(past_out) >= 4:
            third_to_last = past_out[-3]
            past_window = torch.cat((past_window.to(device),reshape(third_to_last.to(device))),dim=1)
        else:
            past_window = torch.cat((past_window.to(device),reshape(torch.zeros([1,1,12]).to(device))),dim =1)
        if len(past_out) >= 3:
            second_to_last = past_out[-2]
            past_window = torch.cat((past_window.to(device),reshape(second_to_last).to(device)),dim=1)
        else:
            past_window = torch.cat((past_window.to(device),reshape(torch.zeros([1,1,12])).to(device)),dim =1)
        if len(past_out) >= 2:
            latest_out = past_out[-1]
            past_window = torch.cat((past_window.to(device),reshape(latest_out).to(device)),dim =1)
        else:
            past_window = torch.cat((past_window.to(device),reshape(torch.zeros([1,1,12]).to(device))),dim =1)



        # reshaped inputs to flatten around the sequence dimension => so concat sequences
        flattened_data = weather_window.view(weather_window.size(0), -1).to(device)

        # run model
        output_tensor_inner = model(flattened_data,past_window.to(device))

        #append to temporary storage that is used to build the windows
        past_out.append(output_tensor_inner)

        # append to output tensor
        output_tensor = torch.cat((output_tensor,output_tensor_inner),dim=0)
  
    # applying rolling mean to neural network output to smooth the curve
    windows = output_tensor.flatten().unfold(dimension=0, size=12, step=1).to(device)
    rolling_mean = windows.mean(dim=1)
    padding_value = rolling_mean[0].item()
    padding = torch.full((11,),padding_value).to(device)
    rolling_mean_padded = torch.cat((padding,rolling_mean),dim=0).to(device)


    # upscale the output
    
    if data_for_date != None:
        label_broadcasted = reverse_min_max(data_for_date.to_lable_normalized_smoothed_and_hours_accurate().flatten(),scaling_values.power_production[0],scaling_values.power_production[1])
        label_broadcasted = torch.tensor(label_broadcasted)
    else:
        label_broadcasted = None
    nn_output_broadcasted = reverse_min_max(rolling_mean_padded,scaling_values.power_production[0],scaling_values.power_production[1])
    nn_out_broadcasted = torch.tensor(nn_output_broadcasted)
    

    # get the sum of all values to get the total production
    np_output = nn_out_broadcasted.cpu().detach().numpy()
    sum_nn =   (np_output * 0.0833).sum() 
    if data_for_date != None:
        sum_lable =   (label_broadcasted.cpu().detach().numpy() * 0.0833).sum() 
    else:
        sum_lable ="Not found"
    print(f"Sum of NN output: {sum_nn} kWh",f"Sum of Lable: {sum_lable} kWh")

    #Plot the output in comparison to the lable for a specified day
    time_points = pd.date_range(start="00:00", end="23:55", freq="5min")
    plt.figure(figsize=(10,5))
    plt.scatter(time_points,nn_out_broadcasted, label='NN Output', color='red', marker='o')  
    if data_for_date != None:
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
    default_date = "2024-09-12"
    print(f"Please provide a date in the following format: YYYY-MM-DD, Default is {default_date}")
    date = input()
    inference_lstm(date,default_date)
