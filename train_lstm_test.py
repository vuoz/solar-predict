import torch
from dataloader import Dataloader, Coordinates,DataframeWithWeatherAsDict,split_dfs_by_season
import dotenv
from model import LstmModel
from train_lstm import train_lstm_new
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dotenv.load_dotenv()
    data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()

    seasonal_data = split_dfs_by_season(data)
    seasonal_data.normalize_seasons()
    model = LstmModel()
    model.to(device)
    train_lstm_new(model,device,seasonal_data.summer,"sdd")

