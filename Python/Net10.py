# torch nn class model definition function for cifar10
import torch
import torch.nn as nn
import torch.nn.functional as F
# construct neural network as class for cifar10 data
class Net10(nn.Module):  # define functions using nn.module
    def __init__(self):  # define intial functions
        super().__init__()
     # define input layer as 3, filter as 32, kernel size as 3   
        self.conv1 = nn.Conv2d(3, 32, 3)  # input layer
     # define 2 x 2 MaxPooling size        
        self.pool = nn.MaxPool2d(2) 
        # define input as 32, filter as 32, kernel size as 3   
        self.conv2 = nn.Conv2d(32, 32, 3)
     # define activation function in convolution layer   
        self.relu = nn.ReLU()
     # define input as 32, filter as 64, kernel size as 3   
        self.conv3 = nn.Conv2d(32, 64, 3)
     # define input as 64, filter as 64, kernel size as 3   
        self.conv4 = nn.Conv2d(64, 64, 3)
     # define dropout   
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
     # define fc1 flattened final size 4 x 4 x 64 to 128 vector    
        self.fc1 = nn.Linear(1024, 128)
     # define output layer 128 vectors to 10 classes   
        self.fc2 = nn.Linear(128, 10)
# define forward propagation convolutional steps        
    def forward(self, x):
        x = self.conv1(x)   # image size 32
        x = self.relu(x)
        x = self.conv2(x)   # 30 after convolution (32-3+1)   
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.conv3(x)   # 28 after convolution (30-3+1)   
        x = self.relu(x)
        x = self.conv4(x)   # 26 after convolution (28-3+1)  
        x = self.relu(x)
        x = self.pool(x)    # 13 after pooling (26/2)    
        x = self.dropout1(x)
        x = self.conv4(x)   # 11 after convolution (13-3+1)    
        x = self.relu(x)
        x = self.conv4(x)   # 9  after convolution (11-3+1)   
        x = self.relu(x)
        x = self.pool(x)    # 4  after pooling (9/2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # matrix to vector conversion 256
        x = self.fc1(x)     #  (4 x 4 x 64, 128)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)     # (128, 10)
        output = F.log_softmax(x, dim=1)  # activation is Softmax
        return output
if __name__ == '__main__':
    Net10()    
