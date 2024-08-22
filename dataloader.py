import os; import matplotlib.pyplot as plt;
import polars as pl;
from polars.dependencies import dataclasses
import requests;
from dataclasses import dataclass;
from datetime import datetime;

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



 
def parse_senec_export_type(line: str)-> tuple[SenecExportType|None,Exception|None]:
    date_format = "%d.%m.%Y %H:%M:%S"   

    splits = line.split(";")
    if len(splits) != 7:
        return (None, Exception(f"Invalid line: {line}"))
    date_string  = splits[0]
    time = datetime.strptime(date_string, date_format)
    gridexport = float(splits[1])
    usage = float(splits[2])
    acculevel = float(splits[3])
    accudischarge = float(splits[4])
    production = float(splits[5])
    accuvoltage = float(splits[6])
    accucurrent = float(splits[7])
    return (SenecExportType(time,gridexport,usage,acculevel,accudischarge,production,accuvoltage,accucurrent),None)
   
class Dataloader():
    def __init__(self, path,coordinates:Coordinates):
        self.path = path
        self.coords = coordinates

    def get_data_files(self):
        csv_files = [] 
        for file in os.listdir(self.path):
            if file.endswith(".csv"):
                csv_files.append(self.path+"/"+file)
        return csv_files
        
    def load(self):
        

        def read_csv(file_name:str)-> tuple[list[pl.DataFrame]|None,Exception|None]:
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
            sanitized_dfs = []
            for df in dfs_per_day:
                if len(df.rows()) < 270:
                    continue
                sanitized_dfs.append(df)
            return (sanitized_dfs,None)



        csv_files = self.get_data_files()
        data_days = []
        for file in csv_files:
            (files,err)= read_csv(file)
            if err != None:
                continue
            if files == None:
                continue
            # can be made faster
            for file in files:
                data_days.append(file)

        print(len(data_days))

        # then aggreagte the date into an pickle file to store for later use

        
    def get_weather_for_date(self,date_start:str,date_end: str)-> tuple[int|None,Exception|None]:  
        url = "https://archive-api.open-meteo.com/v1/archive";
        params = {
            "latitude": self.coords.latitude,
	        "longitude":self.coords.longitude,
	        "start_date": date_start,
	        "end_date": date_end,
	        "hourly": "temperature_2m",
        }
        resp =requests.get(url,params=params)
        if resp.status_code != 200:
            return (None,Exception(f"Failed to get weather data for window {date_start}-{date_end}, error: {resp.status_code}"))
        #...
        return (0,None)


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
            sanitized_dfs = []
            for df in dfs_per_day:
                if len(df.rows()) < 270:
                    continue
                sanitized_dfs.append(df)

            example_date_df = sanitized_dfs[5]
            example_date_df.select("Stromerzeugung [kW]")
            total_energy = example_date_df["Energy [kWh]"].sum()
            print(total_energy)
            
            df_pandas = example_date_df.to_pandas()
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
        read_csv_and_display_daily_data(csv_files[0])
        



            
DataLoader = Dataloader("data",Coordinates(0.0,0.0))
DataLoader.visualize()
