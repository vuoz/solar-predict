import dotenv
from dataloader import Dataloader, Coordinates
import torch
from torch import nn
from model import Model
from dataloader import DataframeWithWeatherAsDict
import os

# very simple training loop with a train and test split
def train(model:Model,device, data:list[DataframeWithWeatherAsDict],epochs=100,lr=0.001):
    train_size = int(0.8 * len(data))
    data_train = data[:train_size]
    data_test = data[train_size:]

    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        for day in data_train:
            
            inputs = day.wether_to_feature_vec().to(device)
            lable = day.df_to_lable().to(device)
            # this input flattening might cause potential accuracy loss
            # in the future it might be necessary to make the model input 2d, which might increase the models
            # ability to treat the hours independently and therefore associate them with specific parts of the output,
            # which might intern improve total accuracy
            outputs = model(inputs.flatten())

            
            loss = criterion(outputs,lable.float())


            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for day in data_test:
                inputs = day.wether_to_feature_vec().to(device)
                lable = day.df_to_lable().to(device)
                outputs = model(inputs.flatten())
                 
                loss = criterion(outputs,lable.float())
                test_loss += loss.item()

        avg_test_loss = test_loss / len(data_test)

        print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(data)}, Test Loss: {avg_test_loss}')

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dotenv.load_dotenv()
    data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()
    model = Model(24*6)
    model.to(device)
    model = train(model,device,data,epochs=150,lr=0.0001)
    torch.save(model.state_dict(), "model.pth")
