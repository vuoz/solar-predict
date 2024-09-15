import torch.nn as nn



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
        self.lstm_1 = nn.LSTM(input_size=13,num_layers=8,hidden_size=512, batch_first=True)

        self.seq = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU()
        )
        return
    def forward(self, inputs):
        lstm_out,self.hidden = self.lstm_1(inputs)
        return self.seq(lstm_out)
    def reset_lstm_state(self):
        self.hidden = None
