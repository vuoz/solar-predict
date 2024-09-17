import torch.nn as nn
import torch
import math



#very basic mlp based model. i flatten the input from the intital 24 * 5 to a 1d tensor for simplicity
#this might be a cause for accuracy loss
class Model(nn.Module):
    def __init__(self,input_size):
        super(Model, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,288),
            nn.Softplus()
        ) 

    def forward(self,x):
        return self.seq(x) 

class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel,self).__init__()
        self.lstm_1 = nn.LSTM(input_size=312,num_layers=5,hidden_size=512, batch_first=True)
        self.lstm_2  = nn.LSTM(input_size=550,num_layers=5,hidden_size=1024, batch_first=True)

        self.seq = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.Softplus()
        )
        return
    def forward(self,weather_inputs,last_yield):
        lstm_out,self.hidden = self.lstm_1(weather_inputs)
        concat = torch.cat((lstm_out,last_yield),dim=1)
        lstm_out, self.hidden_2 = self.lstm_2(concat)
        return self.seq(lstm_out)
    def reset_lstm_state(self):
        self.hidden =None
        self.hidden_2 = None


