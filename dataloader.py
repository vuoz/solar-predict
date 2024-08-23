import os;
import dill as pickle;
import matplotlib.pyplot as plt;
import polars as pl;
import json;
import requests;
from dataclasses import dataclass;
from datetime import datetime;
from customTypes import WeatherData;
import dotenv;
from customTypes import customWeatherDecoder;

@dataclass
class Coordinates():
    latitude: float
    longitude: float

@dataclass
class SenecExportType():
    time:datetime
    gridexport: float
    usage: float
    acculevel: float
    accudischarge:float
    production: float
    accuvolatge: float
    accucurrent:float


@dataclass
class DataframeWithWeather():
    df: pl.DataFrame
    weather: WeatherData

 
  
class Dataloader():
    def __init__(self, path,coordinates:Coordinates):
        self.path = path
        self.coords = coordinates

    def get_data_files(self)->list[str]:
        csv_files = [] 
        for file in os.listdir(self.path):
            if file.endswith(".csv"):
                csv_files.append(self.path+"/"+file)
        return csv_files
    def load(self):
        with open("training_data.pkl", 'rb') as f:
            loaded_dataframes_with_weather = pickle.load(f)
        for df_with_weather in loaded_dataframes_with_weather:
            print(df_with_weather.weather.daily.sunrise)
        pass
    def prepare_and_save(self):
        
        def read_csv(file_name:str)-> tuple[list[DataframeWithWeather]|None,Exception|None]:
            df = pl.read_csv(file_name,separator=";",ignore_errors=True)
            df = df.with_columns(
                    pl.col(col).str.replace(",", ".").cast(pl.Float64) for col in df.columns[1:] 
            )
            df = df.with_columns(
                pl.col("Uhrzeit").str.to_datetime().alias("Timestamp")
            )

            df = df.with_columns(
                pl.col("Timestamp").dt.date().alias("Date")
            )
            df = df.with_columns(
                (pl.col("Stromerzeugung [kW]") * 0.0833).alias("Energy [kWh]")
            )   

            dfs_per_day = [df.filter(pl.col("Date") == date) for date in df.select(pl.col("Date")).unique().to_series()]
            sanitized_dfs:list[DataframeWithWeather] = []
            for df in dfs_per_day:
                #one day should ideally have 288 data points.
                # every day under 270 data points is disregarded
                if len(df.rows()) < 270:
                    continue
             
                date = df.get_column("Date")[0]
                #match solar system data with weather data for day
                # currently we get data for each day individually, but the the future you could do batching
                (weather,err) = self.get_weather_for_date(date,date)
                if err != None:
                    continue
                if weather == None:
                    continue
                sanitized_dfs.append(DataframeWithWeather(df=df,weather=weather))
            return (sanitized_dfs,None)



        csv_files = self.get_data_files()
        data_days:list[DataframeWithWeather] = []
        for file in csv_files:
            print(f"Processing file {file}")
            (days,err)= read_csv(file)
            if err != None:
                continue
            if days == None:
                continue
            # can be made much faster
            for day in days:
                data_days.append(day)

        with open("training_data.pkl","wb") as file:
            pickle.dump(data_days,file)


        # then aggreagte the date into an pickle file to store for later use

        
    def get_weather_for_date(self,date_start:str,date_end: str)-> tuple[WeatherData|None,Exception|None]:  
        url = "https://archive-api.open-meteo.com/v1/archive";
        params = {
	        "latitude": self.coords.latitude,
	        "longitude": self.coords.longitude,
	        "start_date": date_start,
	        "end_date": date_end,
	        "hourly": ["temperature_2m", "precipitation", "cloud_cover", "sunshine_duration"],
	        "daily": ["sunrise", "sunset", "sunshine_duration"],
	        "timezone": "Europe/Berlin"
        }
        resp =requests.get(url,params=params)
        if resp.status_code != 200:
            return (None,Exception(f"Failed to get weather data for window {date_start}-{date_end}, error: {resp.status_code}"))
        weather = json.loads(resp.text, object_hook=customWeatherDecoder) 
        return (weather,None)


    # currently just for testing purposes; to be able to understand the data
    def visualize(self):

        def read_csv_and_display_daily_data(file_name : str):
            df = pl.read_csv(file_name,separator=";")
            df = df.with_columns(
                    pl.col(col).str.replace(",", ".").cast(pl.Float64) for col in df.columns[1:] 
            )
            
            df = df.with_columns(
                pl.col("Uhrzeit").str.to_datetime().alias("Timestamp")
            )

            df = df.with_columns(
                pl.col("Timestamp").dt.date().alias("Date")
            )
            df = df.with_columns(
                (pl.col("Stromerzeugung [kW]") * 0.0833).alias("Energy [kWh]")
            )   

            dfs_per_day = [df.filter(pl.col("Date") == date) for date in df.select(pl.col("Date")).unique().to_series()]
            sanitized_dfs:list[DataframeWithWeather] = []
            for df in dfs_per_day:
                if len(df.rows()) < 270:
                    continue
                date = df.get_column("Date")[0]
                (weather,err) = self.get_weather_for_date(date,date)
                if err != None:
                    print(err)
                if weather == None:
                    print("weather is none")
                    continue
                sanitized_dfs.append(DataframeWithWeather(df=df,weather=weather))

            example_date_df = sanitized_dfs[5]
            print(example_date_df.weather.daily.sunrise)
            example_date_df.df.select("Stromerzeugung [kW]")
            total_energy = example_date_df.df["Energy [kWh]"].sum()
            print(total_energy)
            
            df_pandas = example_date_df.df.to_pandas()
            plt.figure(figsize=(5, 3))
            plt.scatter(df_pandas['Timestamp'], df_pandas['Stromerzeugung [kW]'], color='blue')
            plt.xlabel('Time')
            plt.ylabel('Electricity Production [kW]')
            plt.title('Electricity Production Over Time')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()  
            


        csv_files = self.get_data_files()
        read_csv_and_display_daily_data(csv_files[1])
        



dotenv.load_dotenv()
DataLoader = Dataloader("data",Coordinates(float(os.environ["Long"]),float(os.environ["Lat"])))
DataLoader.load()
