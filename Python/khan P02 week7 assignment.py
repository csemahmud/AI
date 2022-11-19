# -*- coding: utf-8 -*-
"""

@author: KHAN MAHMUDUL HASAN

Created on Wed Sep 14 10:21:52 2022
PYR102 week 7 assignment:
 1. create fruit image data files cnn
   - rename all grape, lemon, and strawberry
   - with numbering (each image 300 samples)

"""

#
import os
import glob
import pandas as pd
from PIL import Image

#
#
path1 = './image_work/grape/'   # define dir path for input
files = glob.glob(path1+'*')  # get jpg data
path2 = './image_work/fruits_out/'   # define dir path for output

i = 0
 
for i, f in enumerate(files, 1):  # start from 1 numbering
    fname = 'grape_' + str(i) + '.jpg'  # define renamed string
    os.rename(f, path2 + fname)   # rename and numbering all
#
#

j = i + 1

path1 = './image_work/lemon/'   # define dir path for input
files = glob.glob(path1+'*')  # get jpg data
path2 = './image_work/fruits_out/'   # define dir path for output
 
for i, f in enumerate(files, j):  # start from 1 numbering
    fname = 'lemon_' + str(i) + '.jpg'  # define renamed string
    os.rename(f, path2 + fname)   # rename and numbering all
#
#

j = i + 1

path1 = './image_work/strawberry/'   # define dir path for input
files = glob.glob(path1+'*')  # get jpg data
path2 = './image_work/fruits_out/'   # define dir path for output
 
for i, f in enumerate(files, j):  # start from 1 numbering
    fname = 'strawberry_' + str(i) + '.jpg'  # define renamed string
    os.rename(f, path2 + fname)   # rename and numbering all
#
# get only image file name created
files = os.listdir(path2)
files_file = [f for f in files if os.path.isfile(os.path.join(path2, f))]
print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']
#
# create dataframe of image file name
df_fruits = pd.DataFrame(files_file)  # list to dataframe
# assign column name as image
df_fruits.columns = ["image"]
#
# add label column for fruits as 1
df_fruits["label"] = 1

"""

 2. resize all image as 32 x 32 

"""

# define image size for resizing
w = 32
h = 32
dst_dir = 'image_work/fruits_out/data_resized'  # define save file holder
os.makedirs(dst_dir, exist_ok=True)

files = glob.glob('./image_work/fruits_out/*.jpg')

for f in files:
    img = Image.open(f)
    img_resize = img.resize((w, h))
    root, ext = os.path.splitext(f)
    basename = os.path.basename(root)
    img_resize.save(os.path.join(dst_dir, basename + ext))

"""

 3. create annotation csv file with image name
    and label
   

"""
#
# save created annotation csv list in working directory
# save without index
df_fruits.to_csv("cnn_fruits.csv", index = False)
#
