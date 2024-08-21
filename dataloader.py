import os; import matplotlib.pyplot as plt;
from pandas import to_datetime
import polars as pl;
from dataclasses import dataclass;
from datetime import datetime;



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
    def __init__(self, path):
        self.path = path

    def load(self):
        
        def get_data_files(path):
            csv_files = [] 
            for file in os.listdir(self.path):
                if file.endswith(".csv"):
                    csv_files.append(self.path+"/"+file)
            return csv_files
        
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
            

            dfs_per_day = [df.filter(pl.col("Date") == date) for date in df.select(pl.col("Date")).unique().to_series()]
            print(len(dfs_per_day))
            

            example_date_df = dfs_per_day[0]
            
            df_pandas = example_date_df.to_pandas()
            plt.figure(figsize=(10, 6))
            plt.scatter(df_pandas['Timestamp'], df_pandas['Stromerzeugung [kW]'], color='blue')
            plt.xlabel('Time')
            plt.ylabel('Electricity Production [kW]')
            plt.title('Electricity Production Over Time')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()  
            


        csv_files = get_data_files(self.path)
        read_csv_and_display_daily_data(csv_files[0])
        

        



            
DataLoader = Dataloader("data")
DataLoader.load()
