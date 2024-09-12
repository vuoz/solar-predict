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
import random
from torch.utils.data import DataLoader,Dataset

class CustomDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

def batch_data(data:list[tuple[torch.Tensor,torch.Tensor]],batch_size):
    custom_dataset = CustomDataset(data)
    dataset = DataLoader(custom_dataset,batch_size,shuffle=True)
    return dataset
def train_lstm(model:LstmModel,device, data:list[DataframeWithWeatherAsDict],name:str,queue,epochs=100,lr=0.0001):
    train_size = int(0.9* len(data))
    train_set = data[:train_size]
    test_set = data[train_size:]
    # trying to do batching to make training more efficient
    '''
    also want to add a bigger window size that the model can use to predict the next timestep
    train_data_batching_ready = [(day.weather_to_feature_vec(),day.to_lable_normalized_smoothed_and_hours_accurate()) for day in data]
    dataset = batch_data(train_data_batching_ready,2)
    '''

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduled_sampling_prob:float = 5/10
    

    print(f"Starting Training {name} ")
    loss_values = []
    test_loss_values = []
    last_test_loss = 0.0
    no_improvement = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        

        for day in train_set:
            day_loss = 0.0
            inputs = day.weather_to_feature_vec().to(device)
            labels = day.to_lable_normalized_smoothed_and_hours_accurate().to(device)
            prev_out = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0])
            for (input,lable) in zip(inputs,labels):
                out = model(input.to(device).float(),prev_out.to(device).float())

                # trying out using the models output as next input to make it more independent and actually reliable in inference mode when there is no true data to hand
                # when in teacher mode prev_out would be set to label
                if random.random() < scheduled_sampling_prob:
                    prev_out = lable
                else:
                    prev_out = out.detach()

                loss = criterion(out,lable.float())
                optimizer.zero_grad()    
                loss.backward()
                optimizer.step()
                day_loss += loss.item() 
            #model.reset_state_lstm()

            epoch_loss += day_loss / 12 


        test_loss = 0.0

        with torch.no_grad():
            for day in test_set:
                inputs = day.weather_to_feature_vec().to(device)
                label = day.to_lable_normalized_smoothed_and_hours_accurate().to(device)

                day_loss = 0.0
                prev_out = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0])
                for (input,label) in zip(inputs,label):

                    out = model(input.to(device).float(),prev_out.to(device).float())
                    prev_out = out                     
                    loss = criterion(out,label.float())
                    day_loss += loss.item()
                test_loss += day_loss / 12
        if test_loss > last_test_loss or last_test_loss ==test_loss :
            no_improvement += 1
        if no_improvement > 5:
            break
        last_test_loss = test_loss
        train_loss_percent = np.sqrt(epoch_loss / len(train_set)) * 100
        test_loss_percent = np.sqrt(test_loss / len(test_set)) * 100

        loss_values.append(train_loss_percent)
        test_loss_values.append(test_loss_percent)




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
        p = mp.Process(target=train_lstm, args=(model,device,season[0],season[1],res_queue,20000,0.001))
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


