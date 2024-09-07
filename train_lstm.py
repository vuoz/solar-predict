import dotenv
import torch
from dataloader import Dataloader, Coordinates,DataframeWithWeatherAsDict
import os
from model import LstmModel
from torch import nn

def train_lstm(model:LstmModel,device, data:list[DataframeWithWeatherAsDict],epochs=100,lr=0.0001):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Starting Training ")
    loss_values = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for day in data:
            day_loss = 0.0
            inputs = day.weather_to_feature_vec().to(device)
            lables = day.to_lable_normalized_hours_accurate().to(device)
            prev_out = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0])
            for (input,lable) in zip(inputs,lables):
                out = model(input.to(device).float(),prev_out.to(device).float())

                loss = criterion(out,lable.float())
                optimizer.zero_grad()    
                loss.backward()
                optimizer.step()
                day_loss += loss.item() 
                prev_out = lable
            epoch_loss += day_loss / 12  # Average loss over the 12 time steps
            loss_values.append(day_loss / 12)


        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / 100:.4f}')
    return model,loss_values


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dotenv.load_dotenv()
    data = Dataloader("/data",Coordinates(float(os.environ["Lat"]),float(os.environ["Long"]))).load()

    model = LstmModel()
    model.to(device)
    model,loss_progression = train_lstm(model,device,data,epochs=300,lr=0.0001)
    torch.save(model.state_dict(), "model_lstm.pth")




