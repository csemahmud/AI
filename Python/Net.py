# torch nn class model definition function
import torch
import torch.nn as nn
import torch.nn.functional as F
# construct neural network as class
class Net(nn.Module):  # define functions using nn.module
    def __init__(self):  # define intial functions
        super().__init__()
     # define input layer as 1, filter as 32(n1), kernel size as 5   
        self.conv1 = nn.Conv2d(1, 32, 5)  # input layer
     # define 2 x 2 MaxPooling size        
        self.pool = nn.MaxPool2d(2) 
     # define input as 32, filter as 64(n2), kernel size as 5   
        self.conv2 = nn.Conv2d(32, 64, 5)
     # define activation function in convolution layer   
        self.relu = nn.ReLU()
     # define dropout   
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
     # define fc1 flattened final size 4 x 4 x 64 to 128(n3) vector    
        self.fc1 = nn.Linear(4 * 4 * 64, 128)
     # define output layer 128 vectors to 10 classes   
        self.fc2 = nn.Linear(128, 10)
# define forward propagation convolutional steps        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # matrix to vector conversion
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # activation is Softmax
        return output
if __name__ == '__main__':
    Net()    
