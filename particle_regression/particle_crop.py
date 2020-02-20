# Copyright 2020 TU Ilmenau. All Rights Reserved.
#
# Code for the Particle regression crop for the
# On the use of a cascaded convolutional neural network for three-dimensional flow measurements using astigmatic PTV task.

# ==============================================================================

import csv
from PIL import Image
import os
import pandas as pd


TBL_IMG_ID='id'
TBL_CLASS='class'
TBL_BBOX_RB_X='xmax'
TBL_BBOX_LT_X='xmin'
TBL_BBOX_RB_Y='ymax'
TBL_BBOX_LT_Y='ymin'
TBL_POS_X='pos_x'
TBL_POS_Y='pos_y'
TBL_POS_Z='pos_z'
TBL_FILENAME='filename'
TBL_CROP_LT_X='cropxmin'
TBL_CROP_LT_Y='cropymin'
TBL_CROP_RB_X='cropxmax'
TBL_CROP_RB_Y='cropymax'

TBL_HEADER=[TBL_IMG_ID,TBL_CLASS,TBL_BBOX_RB_X,TBL_BBOX_LT_X,TBL_BBOX_RB_Y,TBL_BBOX_LT_Y,TBL_FILENAME,TBL_POS_X,TBL_POS_Y,TBL_POS_Z]

path = ''
new_csv = ''

df = pd.read_csv('',
                 sep=',', names=TBL_HEADER)

filenames = [os.path.join(path, file) for file in os.listdir(path) if ".png" in file]

df_new = pd.DataFrame()
fname_list=[]
imageid_list=[]
width_list=[]
height_list=[]
class_list=[]
pos_x_list=[]
pos_y_list=[]
posz_list=[]
xmin_list=[]
ymin_list=[]
xmax_list=[]
ymax_list=[]



for filename in sorted(filenames):

    img_id = int(filename.split('/')[-1].split('.')[0][-4:])
    print(img_id)
    filename_id= filename.split('/')[-1].split('.')[0]
    filename_name=filename.split('/')[-1]
    df_filt = df[df[TBL_IMG_ID] == img_id]


    img = Image.open(filename)
    idx=0
    for _, row in df_filt.iterrows():
        row[TBL_POS_X]=row[TBL_BBOX_LT_X]+(row[TBL_BBOX_RB_X]-row[TBL_BBOX_LT_X])/2
        row[TBL_POS_Y]=row[TBL_BBOX_LT_Y]+(row[TBL_BBOX_RB_Y]-row[TBL_BBOX_LT_Y])/2


        row[TBL_CROP_LT_X]=row[TBL_POS_X]-90
        row[TBL_CROP_LT_Y]=row[TBL_POS_Y]-90
        row[TBL_CROP_RB_X]=row[TBL_POS_X]+90
        row[TBL_CROP_RB_Y]=row[TBL_POS_Y]+90
        area = (row[TBL_CROP_LT_X],
                row[TBL_CROP_LT_Y],
                row[TBL_CROP_RB_X],
                row[TBL_CROP_RB_Y])


        cropped_img = img.crop(area)


        fname = "{0}_{1}_{2}.png".format(filename_id,img_id,idx)
        fname_list.append(fname)
        imageid_list.append(int(img_id))
        width_list.append(210)
        height_list.append(210)
        class_list.append(row[TBL_CLASS])
        posz_list.append(0)
        pos_x_list.append(row[TBL_POS_X])
        pos_y_list.append(row[TBL_POS_Y])
        xmin_list.append(row[TBL_BBOX_LT_X])
        ymin_list.append(row[TBL_BBOX_LT_Y])
        xmax_list.append(row[TBL_BBOX_RB_X])
        ymax_list.append(row[TBL_BBOX_RB_Y])

        cropped_img.save(fname)
        idx += 1

df_new = pd.DataFrame(data={
        'filename':fname_list,
        'imageid':imageid_list,
        'width':width_list,
        'height':height_list,
        'class':class_list,
        'pos_z':posz_list,
        'posx':pos_x_list,
        'posy':pos_y_list,
        'xmin':xmin_list,
        'ymin':ymin_list,
        'xmax':xmax_list,
        'ymax':ymax_list,
        })

df_new.to_csv(new_csv, sep=',', header= True, index=False)
