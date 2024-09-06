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


