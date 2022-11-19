# -*- coding: utf-8 -*-
"""
Week 8 assignment for CNN study
assignment outline:
    
Image dataset: your fruit data created(week 7 assignment)
              (image size 32 x 32)

1) set up libraries required

"""

# CNN4 version 2
# PYR102 Pytorch Deep learning Sample using cnn4 dataset
# 1) Library set-up
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader 
#from torch.utils.data import Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # set to avoid error #15
#import ssl   # avoid SSL certification error
#ssl._create_default_https_context = ssl._create_unverified_context
#

"""

2) data preparation for Deep Learning
 a) customize cnn data for data loader
    using MYdataset function
 b) define image data directory and lable csv file
    annotations_file = 'file name.csv'
    img_dir = "image data file pass"
 c) calcurate mean and std for normalization for dataset 
 d) define transformation for x data (image data)
 e) get dataset with normalization transform
 f) constract data loader for train and test

"""

# 2) data preparation for Deep Learning
#   
# customize cnn4 data for data loader
from Mydataset import Mydataset
# read csv list of image data names and labels
annotations_file = 'cnn_fruits.csv'
# define the path for image data
img_dir = "./image_work/fruits_out/data_resized/"
# define dataset for mean std calculation
transform=transforms.ToTensor()
cnn4 = Mydataset(annotations_file, img_dir, transform)
#
# calcurate mean and std for normalization for dataset cnn
from get_mean_and_std import get_mean_and_std  # get function
dsloader = DataLoader(cnn4, batch_size=32, shuffle=False)
mean, std =get_mean_and_std(dsloader)
print(mean)
print(std)
# define transformation with normalization
transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=[0.5665, 0.4496, 0.3150],std=[0.2504, 0.2398, 0.2467]),])
# define dataset with normalization transform
cnn4t = Mydataset(annotations_file, img_dir, transform)
# constract data loader
batch_size = 32  # define batch size
train_size = int(len(cnn4t) * 0.8)
valid_size = len(cnn4t) - train_size
train_set, valid_set = random_split(cnn4t, (train_size, valid_size))
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
#

"""

3) Define class model, optimizer, and loss function
net = Net3()   # get class nn model for cifar10
criterion = torch.nn.CrossEntropyLoss()  # loss function 
# define algorism
optimizer = optim.SGD(net.parameters(),lr=0.1) # optimizer learning rate 0.05
EPOCHS = 50  # epoch size

"""

#
# 3) Define class model, optimizer, and loss function
#
from Net3 import Net3  # import Net function
net = Net3()   # get class nn model for cifar10

# if gpu available, use multiple gpus
# 4) GPU/CPU set-up
# select either GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss()  # loss function 
# define algorism
optimizer = optim.SGD(net.parameters(),lr=0.1) # optimizer learning rate 0.05
EPOCHS = 50  # epoch size


"""

4) Perform training and test validation
   using do_train_and_validate1 function

"""

from do_train_and_validate1 import do_train_and_validate1
# 5) Perform training and test validation
history = do_train_and_validate1(net, trainloader, validloader, criterion, optimizer, EPOCHS)
# print Finished if training and validation done
print('Finished')


"""

5) define train and validate accuracy/loss
t_losses = history['train_loss_values']
t_accus = history['train_accuracy_values']
v_losses = history['valid_loss_values']
v_accus = history['valid_accuracy_values']

"""

# 6) graphs for accuracy and vlidate
# define train and validate accuracy/loss
t_losses = history['train_loss_values']
t_accus = history['train_accuracy_values']
v_losses = history['valid_loss_values']
v_accus = history['valid_accuracy_values']


"""

6) plot loss function for train and validate
   using plot_graph function
    
"""

# 7) plot loss function for train and validate
from plot_graph import plot_graph  # import plot_graph function
plt.figure()
plot_graph(t_losses, v_losses, EPOCHS, 'loss(train)', 'loss(validate)')
plot_graph(t_accus, v_accus, EPOCHS, 'Acc(train)', 'Acc(validate)')    
plt.show()
print(net)   # print nn model class structure
# end

