import torch
import torch.nn as nn


#very basic mlp based model. i flatten the input from the intital 24 * 5 to a 1d tensor for simplicity
#this might be a cause for accuracy loss
class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 512)  #
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
       
        self.fc5 = nn.Linear(512, 512)
        self.relu5 = nn.ReLU()
        
        self.fc6 = nn.Linear(512, 512)
        self.relu6 = nn.ReLU()
        
        self.fc7 = nn.Linear(512, 512)
        self.relu7 = nn.ReLU()

        self.fc7_2 = nn.Linear(512, 512)
        self.relu7_2 = nn.ReLU()
        
        self.fc8 = nn.Linear(512, 256)
        self.relu8 = nn.ReLU()
        
        self.fc9 = nn.Linear(256, 128)
        self.relu9 = nn.ReLU()
        
        self.fc10 = nn.Linear(128, 64)
        self.relu10 = nn.ReLU()
        
        self.fc11 = nn.Linear(64, 288)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc7_2(x)
        x = self.relu7_2(x)
        x = self.fc8(x)
        x = self.relu8(x)
        x = self.fc9(x)
        x = self.relu9(x)
        x = self.fc10(x)
        x = self.relu10(x)
        x = self.fc11(x)
        x = self.softplus(x)
        return x

class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel,self).__init__()
        self.lstm_1 = nn.LSTM(input_size=25,hidden_size=128, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128,hidden_size=256, batch_first=True)
        self.fc_1 = nn.Linear(256, 128)
        self.relu_1 = nn.ReLU()

        self.fc_2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU()

        self.fc_3 = nn.Linear(64, 12)
        self.relu_3 = nn.ReLU()

        pass
    def forward(self, weather_input, prev_output):
        weather_input = weather_input.unsqueeze(0)
        prev_output = prev_output.unsqueeze(0)
        combined = torch.cat((weather_input,prev_output),dim=1)
        combined = combined.unsqueeze(0)
        lstm_1_out,(_,_) = self.lstm_1(combined)
        lstm_2_out,(_,_) = self.lstm_2(lstm_1_out)
        fc_out = self.relu_1(self.fc_1(lstm_2_out))
        fc_out = self.relu_2(self.fc_2(fc_out))         
        output = self.relu_3(self.fc_3(fc_out))

        # since lables are of dim [12] we need to remove the other dimensions
        output = output.squeeze(0)
        output = output.squeeze(0)
        return output   

