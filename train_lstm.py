import dotenv
import torch
from dataloader import Dataloader, Coordinates,DataframeWithWeatherAsDict,split_dfs_by_season
import os
from model import LstmModel
from torch import nn
import torch.multiprocessing as mp
import queue
import matplotlib.pyplot as plt
import random

def train_lstm(model:LstmModel,device, data:list[DataframeWithWeatherAsDict],name:str,queue,epochs=100,lr=0.0001):
    train_size = int(0.9* len(data))
    train_set = data[:train_size]
    test_set = data[train_size:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduled_sampling_prob:float = 8/10
    

    print(f"Starting Training {name} ")
    loss_values = []
    last_test_loss = 0.0
    no_improvement = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for day in train_set:
            day_loss = 0.0
            inputs = day.weather_to_feature_vec().to(device)
            #lables = day.to_lable_normalized_hours_accurate().to(device)
            labels = day.to_lable_normalized_smoothed_and_hours_accurate().to(device)
            prev_out = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0])
            for (input,lable) in zip(inputs,labels):
                out = model(input.to(device).float(),prev_out.to(device).float())

                # trying out using the models output as next input to make it more independent and actually relyable in inference mode when there is no true data to hand
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

            epoch_loss += day_loss / 12  # Average loss over the 12 time steps
            loss_values.append(day_loss / 12)


        test_loss = 0.0

        with torch.no_grad():
            for day in test_set:
                inputs = day.weather_to_feature_vec().to(device)
                #l lable = day.to_lable_normalized_hours_accurate().to(device)
                label = day.to_lable_normalized_smoothed_and_hours_accurate().to(device)

                day_loss = 0.0
                prev_out = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0])
                for (input,label) in zip(inputs,label):

                    out = model(input.to(device).float(),prev_out.to(device).float())
                    prev_out = out                     
                    loss = criterion(out,label.float())
                    day_loss += loss.item()
                test_loss += day_loss / 12
        if last_test_loss > last_test_loss:
            no_improvement += 1
        if no_improvement > 10:
            break
        last_test_loss = test_loss



        print(f'{name} Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss / 100:.4f} ,Test Loss: {test_loss/ 100:.4f}')
    print(f"done with training {name}")
    torch.save(model.state_dict(), f"models/lstm_{name}.pth")
    queue.put((name,loss_values))

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
        p = mp.Process(target=train_lstm, args=(model,device,season[0],season[1],res_queue,2000,0.001))
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
        ax.set_title(f'{data[0]}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

    plt.tight_layout()
    plt.show()


