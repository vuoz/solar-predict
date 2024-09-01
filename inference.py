import polars
from model import Model
import torch
import requests
from dataloader import Coordinates, DataframeWithWeatherAsDict,Dataloader;
from datetime import datetime
import os
import dotenv
import matplotlib.pyplot as plt;
import pandas as pd;

def get_weather_data(day:str,cords:Coordinates)->tuple[DataframeWithWeatherAsDict|None,Exception|None]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
	        "latitude": cords.latitude,
	        "longitude": cords.longitude,
	        "start_date": day,
	        "end_date": day,
	        "hourly": ["temperature_2m", "precipitation", "cloud_cover", "sunshine_duration"],
	        "daily": ["sunrise", "sunset", "sunshine_duration"],
	        "timezone": "Europe/Berlin"
        }
    resp =requests.get(url,params=params)
    if resp.status_code != 200:
        return (None,Exception(f"Failed to get weather data for window {day}-{day}, error: {resp.status_code}"))
    data = resp.json()
    data_struct = DataframeWithWeatherAsDict(df=polars.DataFrame(),weather=data)
    return (data_struct,None)


# this is just the version that of inference that is used to test the actual ability of the model
# once the model works, i will add a real and abstraced version of a inference function/ class
if __name__ == "__main__":
    dotenv.load_dotenv()
    print("Please provide a data in the following format: YYYY-MM-DD")

    date = input()
    if len(date) != 10 or datetime.strptime(date,"%Y-%m-%d") == None:
        print("Invalid date format")
        exit(1)

    model = Model(input_size=24*5)
    model.load_state_dict(torch.load("model.pth"))
    weather,err = get_weather_data(date,Coordinates(float(os.environ["Lat"]),float(os.environ["Long"])))
    if err != None:
        print(err)
        exit(1)
    train_data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()
    lable = None
    for datapoint in train_data:
        if str(datapoint.df.get_column("Date")[0]) == date:
            lable  = datapoint.df_to_lable()
    if lable == None:
        print("No lable found")
        exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = weather.wether_to_feature_vec().to(device).flatten()
    output = model(input)
    output_tensor = torch.Tensor(output)
    np_output = output_tensor.cpu().detach().numpy()

    #Plot the output in comparison to the lable for a specified day
    time_points = pd.date_range(start="00:00", end="23:55", freq="5min")
    plt.figure(figsize=(10,5))
    plt.plot(time_points,np_output, label='NN Output', color='red', marker='o')  
    plt.plot(time_points,lable, label='Label Data', color='green', marker='o')  

    plt.title('Neural Network Output and Label Data Overlay')
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Production [kW]')
    plt.legend()
    plt.show()





