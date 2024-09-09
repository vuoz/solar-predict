import os;
import dill as pickle;
import matplotlib.pyplot as plt;
import polars as pl;
import requests;
from dataclasses import dataclass;
from datetime import datetime
import torch;
from torch import Tensor;
from customTypes import WeatherData;
import dotenv;

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
@dataclass
class DataframeWithWeatherAsDict():
    df: pl.DataFrame
    weather: dict
    def weather_to_feature_vec(self)->Tensor:
        weather_dict = []

        # calculating day of the season
        date_object = datetime.strptime(self.weather["daily"]["time"][0], "%Y-%m-%d")
        day_of_season = 0
        if date_object.month in [12,1,2]:
            if date_object.month == 12:
                day_of_season = date_object.day
            elif date_object.month == 1:
                day_of_season = date_object.day + 31 
            elif date_object.month == 2:
                day_of_season = date_object.day + 31 + 31
        elif date_object.month in [3,4,5]:
            if date_object.month == 3:
                day_of_season = date_object.day 
            elif date_object.month == 4:
                day_of_season = date_object.day + 31
            elif date_object.month == 5:
                day_of_season = date_object.day + 31 + 30
        elif date_object.month in [6,7,8]:
            if date_object.month == 6:
                day_of_season = date_object.day
            elif date_object.month == 7:
                day_of_season = date_object.day  + 30
            elif date_object.month == 8:
                day_of_season = date_object.day + 30 + 31

        elif date_object.month in [9,10,11]:
            if date_object.month == 9:
                day_of_season = date_object.day 
            elif date_object.month == 10:
                day_of_season = date_object.day  + 30
            elif date_object.month == 11:
                day_of_season = date_object.day  + 30 + 31
        # removing this for now. lets see what this does
        #day_of_year = date_object.timetuple().tm_yday
        hour = self.weather["hourly"]
        for i in range(0,24):
            temp_2m = hour["temperature_2m"][i]
            percipitation = hour["precipitation"][i]
            cloud_cover = hour["cloud_cover"][i]
            sunshine_duration = hour["sunshine_duration"][i]
            irradiance = hour["global_tilted_irradiance"][i]
            wind_speed = hour["wind_speed_10m"][i]
            humidity = hour["relative_humidity_2m"][i]
            if temp_2m == None or percipitation == None or cloud_cover == None or sunshine_duration == None or irradiance == None or wind_speed == None or humidity == None:
                temp_2m = 0
                percipitation = 0
                cloud_cover = 0
                sunshine_duration = 0
                irradiance = 0
                humidity = 0
                wind_speed= 0
            weather_dict.append([day_of_season,float(temp_2m),float(percipitation),float(cloud_cover),float(sunshine_duration),float(irradiance),float(wind_speed),float(humidity)])
        return torch.Tensor(weather_dict)

   
    def df_to_lable_normalized(self)-> Tensor:
        # using the smoothed curve. since the model most likely wont be able to infer minutly changes based on hourly weather data
        values = self.df.get_column("Stromerzeugung smoothed")

        tensor = values.to_torch()
        # noramlize the size of the tensor
        if tensor.shape[0] != 288:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            new_tensor = torch.nn.functional.interpolate(tensor, size=(288),mode="linear",align_corners=False)
            tensor_interpolated = new_tensor.squeeze(0).squeeze(0)
            return tensor_interpolated
        return tensor
    def df_to_lable(self)-> Tensor:
        values = self.df.get_column("Stromerzeugung [kW]")

        tensor = values.to_torch()
        # noramlize the size of the tensor
        if tensor.shape[0] != 288:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            new_tensor = torch.nn.functional.interpolate(tensor, size=(288),mode="linear",align_corners=False)
            tensor_interpolated = new_tensor.squeeze(0).squeeze(0)
            return tensor_interpolated
        return tensor

    # this is used for the lstm. the lables actually match the lables to the input data. since the 5 time windows might shift and at one point
    # you might start in the middle of a hour instead of the start this will most likely affect the model ability to predict the output
    def to_lable_normalized_hours_accurate(self)-> Tensor:
        df = self.df
        df = df.with_columns(
            pl.col("Timestamp").dt.hour().alias("Hour")
        )
        grouped_df = df.group_by("Hour").agg(
            pl.col("Stromerzeugung smoothed").alias("production_values"),
        )
        grouped_df = grouped_df.sort("Hour")
        def interpolate_to_twelve(values:Tensor)->Tensor:
            tensor = values.unsqueeze(0).unsqueeze(0)
            new_tensor = torch.nn.functional.interpolate(tensor, size=(12),mode="linear",align_corners=False)
            tensor_interpolated = new_tensor.squeeze(0).squeeze(0)
            return tensor_interpolated

        lable_tensor_list= []
        for row in grouped_df.iter_rows():
            _,values = row
            values_tensor = torch.Tensor(values)
            interpolated = interpolate_to_twelve(values_tensor)
            lable_tensor_list.append(interpolated)
        return torch.stack(lable_tensor_list)
        



@dataclass
class DataframesWithWeatherSortedBySeason():
    spring: list[DataframeWithWeatherAsDict]
    summer: list[DataframeWithWeatherAsDict]
    winter: list[DataframeWithWeatherAsDict]
    autumn: list[DataframeWithWeatherAsDict]
def split_dfs_by_season(data:list[DataframeWithWeatherAsDict])->DataframesWithWeatherSortedBySeason: 
    spring = []
    summer = []
    winter = []
    autumn = []
    for day in data:
        # using the date in the weather data for simplicity
        date = datetime.strptime(day.weather["daily"]["time"][0], "%Y-%m-%d")
        month = date.month
        if month == 3 or month == 4 or month == 5:
            spring.append(day)
        elif month == 6 or month == 7 or month == 8:
            summer.append(day)
        elif month == 9 or month == 10 or month == 11:
            autumn.append(day)
        else:
            winter.append(day)
    return DataframesWithWeatherSortedBySeason(spring=spring,summer=summer,winter=winter,autumn=autumn)
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
    def load(self)->list[DataframeWithWeatherAsDict]:
        with open("training_data.pkl", 'rb') as f:
            loaded_dataframes_with_weather = pickle.load(f)
        data_types:list[DataframeWithWeatherAsDict] = []
        for day in loaded_dataframes_with_weather:
            df = pl.DataFrame(day.df)
            data_types.append(DataframeWithWeatherAsDict(df,day.weather))
        return data_types
    def read_csv(self,file_name:str)-> tuple[list[DataframeWithWeatherAsDict]|None,Exception|None]:
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
        sanitized_dfs:list[DataframeWithWeatherAsDict] = []
        for df in dfs_per_day:
            #one day should ideally have 288 data points.
            # every day under 270 data points is disregarded
            if len(df.rows()) < 280:
                continue
            
            date = df.get_column("Date")[0]
            #match solar system data with weather data for day
            # currently we get data for each day individually, but the the future you could do batching
            (weather,err) = self.get_weather_for_date(date,date)
            if err != None:
                continue
            if weather == None:
                continue
            df_smoothed = self.smooth_graph(df)
            sanitized_dfs.append(DataframeWithWeatherAsDict(df=df_smoothed,weather=weather))
        return (sanitized_dfs,None)

    def smooth_graph(self,df :pl.DataFrame)->pl.DataFrame:
        df_moving_mean = df.with_columns(
            pl.col("Stromerzeugung [kW]").rolling_mean(window_size=20).fill_nan(0).fill_null(0).alias("Stromerzeugung smoothed")
        )
        return  df_moving_mean

    def prepare_and_save(self):
        csv_files = self.get_data_files()
        data_days:list[DataframeWithWeatherAsDict] = []
        for file in csv_files:
            print(f"Processing file {file}")
            (days,err)= self.read_csv(file)
            if err != None:
                continue
            if days == None:
                continue
            # can be made much faster
            for day in days:
                data_days.append(day)


        # then aggreagte the date into an pickle file to store for later use
        with open("training_data.pkl","wb") as file:
            pickle.dump(data_days,file)



        
    def get_weather_for_date(self,date_start:str,date_end: str)-> tuple[dict[str,str]|None,Exception|None]:  
        url = "https://archive-api.open-meteo.com/v1/archive";
        params = {
	        "latitude": self.coords.latitude,
	        "longitude": self.coords.longitude,
	        "start_date": date_start,
	        "end_date": date_end,
	        "hourly": ["temperature_2m", "precipitation", "cloud_cover", "sunshine_duration","global_tilted_irradiance","relative_humidity_2m", "wind_speed_10m"],
	        "daily": ["sunrise", "sunset", "sunshine_duration"],
	        "timezone": "Europe/Berlin"
        }
        resp =requests.get(url,params=params)
        if resp.status_code != 200:
            return (None,Exception(f"Failed to get weather data for window {date_start}-{date_end}, error: {resp.status_code}"))
        return (resp.json(),None)


    def visualize(self):

        def read_csv_and_display_daily_data(file_name : str):
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
            sanitized_dfs:list[DataframeWithWeatherAsDict] = []
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
                sanitized_dfs.append(DataframeWithWeatherAsDict(df=df,weather=weather))

            example_date_df = sanitized_dfs[6]
            example_date_df.df.select("Stromerzeugung [kW]")
            
            df_smoothed = self.smooth_graph(example_date_df.df)
            
            df_pandas = example_date_df.df.to_pandas()
            plt.figure(figsize=(5, 3))
            plt.subplot(2, 1, 1)
            plt.scatter(df_pandas['Timestamp'], df_pandas['Stromerzeugung [kW]'], color='blue')
            plt.xlabel('Time')
            plt.ylabel('Electricity Production [kW]')
            plt.title('Electricity Production Over Time (Original)')
            plt.grid(True)
            plt.xticks(rotation=45)


            plt.subplot(2,1,2)
            df_smoothed_pandas = df_smoothed.to_pandas()
            plt.scatter(df_smoothed_pandas['Timestamp'], df_smoothed_pandas['Stromerzeugung smoothed'], color='red')
            plt.xlabel('Time')
            plt.ylabel('Electricity Production [kW]')
            plt.title('Electricity Production Over Time (Smoothed)')
            plt.grid(True)



            plt.tight_layout()
            plt.show()  
            


        csv_files = self.get_data_files()
        read_csv_and_display_daily_data(csv_files[8])
        

if __name__ == "__main__":
    dotenv.load_dotenv()
    DataLoader = Dataloader("data",Coordinates(float(os.environ["Long"]),float(os.environ["Lat"])))
    DataLoader.prepare_and_save()
