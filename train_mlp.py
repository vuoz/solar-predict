import dotenv
from dataloader import Dataloader, Coordinates, split_dfs_by_season
import torch
from torch import nn
from model import  Model
from dataloader import DataframeWithWeatherAsDict
import os
import torch.multiprocessing as mp
import queue
import numpy as np
import matplotlib.pyplot as plt

# very simple training loop with a train and test split
def train(model:Model,device, data:list[DataframeWithWeatherAsDict],name:str,queue,epochs=100,lr=0.001):
    train_size = int(0.95 * len(data))
    data_train = data[:train_size]
    data_test = data[train_size:]

    
    loss_train = []
    loss_test = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    criterion = nn.MSELoss()
    last_test_loss = 0
    no_improvement = 0
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        for day in data_train:
            
            inputs = day.weather_to_feature_vec().to(device)

            lable = day.to_lable_normalized_and_smoothed().to(device)
            # this input flattening might cause potential accuracy loss
            # in the future it might be necessary to make the model input 2d, which might increase the models ability to treat the hours independently and therefore associate them with specific parts of the output, which might intern improve total accuracy
            outputs = model(inputs.flatten())

            
            loss = criterion(outputs,lable.float())


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for day in data_test:
                inputs = day.weather_to_feature_vec().to(device)
                #lable = day.df_to_lable_normalized().to(device)
                lable = day.to_lable_normalized_and_smoothed().to(device)
                outputs = model(inputs.flatten())
                 
                loss = criterion(outputs,lable.float())
                test_loss += loss.item()

        avg_test_loss = test_loss / len(data_test)

        if avg_test_loss > last_test_loss or avg_test_loss == last_test_loss:
            no_improvement += 1
        if no_improvement > 20:
            break
        last_test_loss = avg_test_loss
        train_loss_percent = np.sqrt(running_loss / len(data)) * 100
        test_loss_percent = np.sqrt(avg_test_loss) * 100


        loss_train.append(train_loss_percent)
        loss_test.append(test_loss_percent)
        print(f'Name:{name} Epoch {epoch+1}, Train Loss : {np.round(train_loss_percent,2)}%, Test Loss : {np.round(test_loss_percent,2)}%.')

    torch.save(model.state_dict(), f"{name}.pth")
    queue.put((name,loss_train,loss_test))

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
        model = Model(24*13)
        model.to(device)
        p = mp.Process(target=train, args=(model,device,season[0],f"models/{season[1]}",res_queue,150,0.00001))
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns
    for (ax,data) in zip(axes.flat,results):
        ax.plot(data[1], label='Train Loss %')
        ax.plot(data[2], label='Test Loss %')
        ax.set_title(f'{data[0]}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

    plt.tight_layout()
    plt.show()

