import torch
from dataloader import Dataloader, Coordinates,split_dfs_by_season
import dotenv
from model import LstmModel
from train_lstm import train_lstm_new
import os
import multiprocessing as mp


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dotenv.load_dotenv()
    data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()

    seasonal_data = split_dfs_by_season(data)
    seasonal_data.normalize_seasons()
    seasonal_data_list = [(seasonal_data.summer,"summer"),(seasonal_data.winter,"winter"),(seasonal_data.spring,"spring"),(seasonal_data.autumn,"autumn")]

    
    mp.set_start_method('spawn')
    processes = []

    for season in seasonal_data_list:
        model = LstmModel()
        model.to(device)
        p = mp.Process(target=train_lstm_new, args=(model,device,season[0],season[1],10000,0.0001))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


