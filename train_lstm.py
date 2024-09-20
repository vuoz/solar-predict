import dotenv
import torch
from dataloader import Dataloader, Coordinates,DataframeWithWeatherAsDict,split_dfs_by_season
import os
import numpy as np
from model import LstmModel
from torch import nn
import torch.multiprocessing as mp
import queue
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset

# custom class to be able to create a pytorch datatset from the provided list of datat
class CustomDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
#  converting data as list into pytorch dataset
def batch_data(data:list[tuple[torch.Tensor,torch.Tensor]],batch_size):
    custom_dataset = CustomDataset(data)
    dataset = DataLoader(custom_dataset,batch_size,shuffle=True)
    return dataset

def train_lstm_new(model:LstmModel,device, data:list[DataframeWithWeatherAsDict],name:str,epochs=50,lr=0.0001):

    train_size = int(0.9* len(data))
    train_set = data[:train_size]
    test_set = data[train_size:]

    # create the dataset / transform it into the correct format
    train_data_batching_ready = [(day.weather_to_feature_vec(),day.to_lable_normalized_smoothed_and_hours_accurate()) for day in train_set]
    dataset = batch_data(train_data_batching_ready,10)
    test_data_batching_ready = [(day.weather_to_feature_vec(),day.to_lable_normalized_smoothed_and_hours_accurate()) for day in test_set]
    dataset_test = batch_data(test_data_batching_ready,1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    last_test_loss = 0.0
    no_improvement = 0
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0



        test_loss = 0.0
        for x,y in dataset:
            splits = torch.split(x,split_size_or_sections=1,dim=1)
            splits_y = torch.split(y,split_size_or_sections=1,dim=1)

            day_loss = 0.0
            for (i,(split_x,split_y)) in enumerate(zip(splits,splits_y)): 
                optimizer.zero_grad()


                # create the weather window
                weather_window  = torch.tensor([])

                fourth_to_last = splits[i-4]
                third_to_last = splits[i-3]
                second_to_last = splits[i-2]
                previous = splits[i-1]
                if fourth_to_last is not None:
                    weather_window = torch.cat((weather_window,fourth_to_last),dim=1)
                else:
                    weather_window = torch.cat((weather_window,torch.zeros(split_x.shape)),dim=1)
                if third_to_last is not None:
                    weather_window = torch.cat((weather_window,third_to_last),dim=1)
                else:
                    weather_window = torch.cat((weather_window,torch.zeros(split_x.shape)),dim=1)
                if second_to_last is not None:
                    weather_window = torch.cat((weather_window,second_to_last),dim=1)
                else:
                    weather_window = torch.cat((weather_window,torch.zeros(split_x.shape)),dim=1)
                if previous is not None:
                    weather_window = torch.cat((weather_window,previous),dim=1)
                else:
                    weather_window = torch.cat((weather_window,torch.zeros(split_x.shape)),dim=1)
                weather_window = torch.cat((weather_window,split_x),dim=1)

                try:
                    in_advance = splits[i+1].to(device)
                    weather_window = torch.cat((weather_window.to(device),in_advance.to(device)),dim=1)
                except IndexError:
                    weather_window = torch.cat((weather_window.to(device),torch.zeros(split_x.shape).to(device)),dim=1)





                # we create the past output window using the past 3 hours 
                # during training we use the ground truth values 
                # this would be called teacher forcing
                fourth_to_last = splits_y[i-4]
                third_to_last = splits_y[i-3]
                second_to_last = splits_y[i-2]
                past_out = splits_y[i-1]

                past_window = torch.tensor([])
                def reshape(x):
                    return x.view(x.size(0), -1)
                if fourth_to_last is not None:
                    past_window = torch.cat((past_window,reshape(fourth_to_last)),dim=1)
                else:
                    past_window = torch.cat((past_window,reshape(torch.zeros(split_y.shape))),dim =1)
                if third_to_last is not None:
                    past_window = torch.cat((past_window,reshape(third_to_last)),dim=1)
                else:
                    past_window = torch.cat((past_window,reshape(torch.zeros(split_y.shape))),dim =1)
                if second_to_last is not None:
                    past_window = torch.cat((past_window,reshape(second_to_last)),dim=1)
                else:
                    past_window = torch.cat((past_window,reshape(torch.zeros(split_y.shape))),dim =1)
                if past_out is not None:
                    past_window = torch.cat((past_window,reshape(past_out)),dim =1)
                else:
                    past_window = torch.cat((past_window,reshape(torch.zeros(split_y.shape))),dim =1)

                # there is some reshaping to do
                flattened_data = weather_window.view(weather_window.size(0), -1)

                out = model(flattened_data.to(device),past_window.to(device))

                loss = criterion(out,split_y.to(device).view(x.size(0),-1))
                loss.backward()

                optimizer.step()
                day_loss += loss.item()

                # we only want the model to take account he actual day it is processing
                model.reset_lstm_state()
            epoch_loss += day_loss / 24

        with torch.no_grad():
            for x,y in dataset_test:
                splits = torch.split(x,split_size_or_sections=1,dim=1)
                splits_y = torch.split(y,split_size_or_sections=1,dim=1)
                day_loss = 0.0
                for (i,(split_x,split_y)) in enumerate(zip(splits,splits_y)):


                    # create the weather window
                    weather_window  = torch.tensor([])
                    fourth_to_last = splits[i-4]
                    third_to_last = splits[i-3]
                    second_to_last = splits[i-2]
                    previous = splits[i-1]
                    if fourth_to_last is not None:
                        weather_window = torch.cat((weather_window,fourth_to_last),dim=1)
                    else:
                        weather_window = torch.cat((weather_window,torch.zeros(split_x.shape)),dim=1)
                    if third_to_last is not None:
                        weather_window = torch.cat((weather_window,third_to_last),dim=1)
                    else:
                        weather_window = torch.cat((weather_window,torch.zeros(split_x.shape)),dim=1)
                    if second_to_last is not None:
                        weather_window = torch.cat((weather_window,second_to_last),dim=1)
                    else:
                        weather_window = torch.cat((weather_window,torch.zeros(split_x.shape)),dim=1)
                    if previous is not None:
                        weather_window = torch.cat((weather_window,previous),dim=1)
                    else:
                        weather_window = torch.cat((weather_window,torch.zeros(split_x.shape)),dim=1)
                    weather_window = torch.cat((weather_window,split_x),dim=1)

                    try:
                        in_advance = splits[i+1]
                        weather_window = torch.cat((weather_window.to(device),in_advance.to(device)),dim=1)    
                    except IndexError:
                        weather_window = torch.cat((weather_window.to(device),torch.zeros(split_x.shape).to(device)),dim=1)



                    # build the past 3 hour window
                    fourth_to_last = splits_y[i-4]
                    third_to_last = splits_y[i-3]
                    second_to_last = splits_y[i-2]
                    past_out = splits_y[i-1]
                    past_window = torch.tensor([])
                    def reshape(x):
                        return x.view(x.size(0), -1)
                    if fourth_to_last is not None:
                        past_window = torch.cat((past_window,reshape(fourth_to_last)),dim=1)
                    else:
                        past_window = torch.cat((past_window,torch.zeros(split_y.shape)),dim =1)
                    if third_to_last is not None:
                        past_window = torch.cat((past_window,reshape(third_to_last)),dim=1)
                    else:
                        past_window = torch.cat((past_window,reshape(torch.zeros(split_y.shape))),dim =1)
                    if second_to_last is not None:
                        past_window = torch.cat((past_window,reshape(second_to_last)),dim=1)
                    else:
                        past_window = torch.cat((past_window,reshape(torch.zeros(split_y.shape))),dim =1)
                    if past_out is not None:
                        past_window = torch.cat((past_window,reshape(past_out)),dim =1)
                    else:
                        past_window = torch.cat((past_window,reshape(torch.zeros(split_y.shape))),dim =1)

                    # there is some reshaping to do
                    flattened_data = weather_window.view(weather_window.size(0), -1)

                    # run the model
                    out  = model(flattened_data.to(device),past_window.to(device))

                    # calculate loss 
                    loss = criterion(out,split_y.to(device).view(x.size(0),-1))
                    day_loss += loss.item()
                test_loss += day_loss / 24
                model.reset_lstm_state()
                        
            # calculate average test loss 
            test_loss = test_loss / len(dataset_test)

            #get the loss precentage
            test_loss = np.sqrt(test_loss)*100

            if last_test_loss < test_loss and last_test_loss != 0:
                no_improvement += 1
                continue
            if no_improvement > 10:
                print(f"{name} Stopping training due to no further improvement")
                break

        epoch_loss = np.sqrt(epoch_loss/len(dataset))*100
        print(f'{name} Epoch [{epoch+1}/{epochs}], Loss: {np.round(epoch_loss,2)}%, Test Loss {np.round(test_loss,2)}%')
    torch.save(model.state_dict(), f"models/lstm_new_{name}.pth")

    





















def train_lstm(model:LstmModel,device, data:list[DataframeWithWeatherAsDict],name:str,queue,epochs=100,lr=0.0001):
    train_size = int(0.9* len(data))
    train_set = data[:train_size]
    test_set = data[train_size:]

    train_data_batching_ready = [(day.weather_to_feature_vec(),day.to_lable_normalized_smoothed_and_hours_accurate()) for day in train_set]
    dataset = batch_data(train_data_batching_ready,10)
    test_data_batching_ready = [(day.weather_to_feature_vec(),day.to_lable_normalized_smoothed_and_hours_accurate()) for day in test_set]
    dataset_test = batch_data(test_data_batching_ready,1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    

    print(f"Starting Training {name} ")
    loss_values = []
    test_loss_values = []
    last_test_loss = 0
    no_improvement = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_test_loss = 0.0
        model.train()
        for ( x,y) in dataset:
            out = model(x.to(device))
            loss = criterion(out,y.to(device))
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        with torch.no_grad():
            for (x,y) in dataset_test:
                out = model(x.to(device))
                loss = criterion(out,y.to(device))
                epoch_test_loss += loss.item()
        train_loss_percent = np.sqrt(epoch_loss/len(dataset))*100
        test_loss_percent = np.sqrt(epoch_test_loss/len(dataset_test))*100
        if test_loss_percent > last_test_loss and  last_test_loss != 0 or test_loss_percent == last_test_loss:
            no_improvement += 1
        if no_improvement > 20:
            break
        last_test_loss = test_loss_percent
        test_loss_values.append(test_loss_percent)
        loss_values.append(train_loss_percent)


        print(f'{name} Epoch [{epoch+1}/{epochs}], Train Loss: {np.round(train_loss_percent,2)}% ,Test Loss: {np.round(test_loss_percent,2)}%')
    print(f"done with training {name}")
    torch.save(model.state_dict(), f"models/lstm_{name}.pth")
    queue.put((name,loss_values,test_loss_values))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dotenv.load_dotenv()
    data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()

    seasonal_data = split_dfs_by_season(data)
    seasonal_data.normalize_seasons()
    print("winter dataset length: ",len(seasonal_data.winter))
    print("summer dataset length:",len(seasonal_data.summer))
    print("spring dataset length:",len(seasonal_data.spring))
    print("autumn dataset length:",len(seasonal_data.autumn))

    seasonal_data_list = [(seasonal_data.summer,"summer"),(seasonal_data.winter,"winter"),(seasonal_data.spring,"spring"),(seasonal_data.autumn,"autumn")]

    mp.set_start_method('spawn')
    processes = []
    res_queue = mp.Queue()

    for season in seasonal_data_list:
        model = LstmModel()
        model.to(device)
        p = mp.Process(target=train_lstm, args=(model,device,season[0],season[1],res_queue,2000,0.0001))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not res_queue.empty():
        try:
            results.append(res_queue.get())
        except queue.Empty:
            break
    print("collected results")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns
    for (ax,data) in zip(axes.flat,results):
        ax.plot(data[1], label='Train Loss')
        ax.plot(data[2], label='Test Loss')
        ax.set_title(f'{data[0]}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss %')
        ax.legend()

    plt.tight_layout()
    plt.show()


