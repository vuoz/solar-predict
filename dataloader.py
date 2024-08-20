import os;
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
                    csv_files.append(file)
            return csv_files
        
        csv_files = get_data_files(self.path)

        



            
