import os;
from dataclasses import dataclass;




@dataclass
class SenecExportType():
    time:str
    gridexport: float
    usage: float
    acculevel: float
    accudischarge:float
    production: float
    accuvolatge: float
    accucurrent:float
    

    def parse(self,line: str):
        pass



    
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

        



            
