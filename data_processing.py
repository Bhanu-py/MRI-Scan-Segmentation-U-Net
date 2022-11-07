# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:49:28 2022

@author: Bhanu
"""
from functions import *
import pandas as pd
import os
# from cv2 import imshow, imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import re


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)


files = next(os.walk('data'))
traindf = pd.read_csv('data/train.csv')
# print(traindf.head(50))

classes = traindf['class'].unique().tolist()

# print(traindf.head())

train_df_grouped = traindf.copy()
print(train_df_grouped.head())
train_df_grouped.set_index('id', inplace=True)

seg_list = []
for cl in classes:
    seg = train_df_grouped[train_df_grouped['class'] == cl]['segmentation']
    seg.name = cl
    seg_list.append(seg)

train_df_grouped = pd.concat(seg_list, axis=1).reset_index()

train_df_grouped['case_id'] = train_df_grouped['id'].str.split('_').str[0].str.extract('(\d+)')
train_df_grouped['day']     = train_df_grouped['id'].str.split('_').str[1].str.extract('(\d+)')
train_df_grouped['slice']   = train_df_grouped['id'].str.split('_').str[3]
print(train_df_grouped.head())


def get_case_day_slice(x):

    case = re.search("case[0-9]+", x).group()[len("case"):]
    day = re.search("day[0-9]+", x).group()[len("day"):]
    slice_ = re.search("slice_[0-9]+", x).group()[len("slice_"):]
    return case, day, slice_


def process_df(df, path):

    all_images = glob(os.path.join(path, '**', '*png'), recursive=True)
    img_df = pd.DataFrame(all_images, columns=['full_path'])
    img_df.loc[:, ["case_id", 'day', 'slice']] = img_df['full_path'].apply(get_case_day_slice).to_list()
    return df.merge(img_df, on=["case_id", 'day', 'slice'], how='left')


train_df_grouped = process_df(train_df_grouped, 'data/train')

print(train_df_grouped)
train_df_grouped.fillna('', inplace=True)
print(train_df_grouped.head())

num = 5
segmentation_df_example = train_df_grouped[train_df_grouped.large_bowel != ''].sample(num).reset_index()

fig, ax = plt.subplots(num, 3, figsize=(18, 8*num))
for ind, row in segmentation_df_example.iterrows():
    print(ind)
    img = mpimg.imread(row['full_path'], format='png')
    ax[ind, 0].imshow(img)
    ax[ind, 0].set_title(row['id'])

    mask = np.zeros(img.shape)
    for j, cl in enumerate(classes):
        mask += rle_decode(row[cl], img.shape) * (j + 1) / 4 * np.max(img)
    ax[ind, 1].imshow(mask)

    ax[ind, 2].imshow(img + mask)
plt.show()




# print(segmentation_df_example)
#
# print(traindf[traindf['segmentation'].notnull()])
#
# num = 7
# segmentation_df_example = train_df_grouped[train_df_grouped.large_bowel != ''].sample(num)
#
# fig, ax = plt.subplots(num, 3, figsize=(18, 8 * num))
# for i in range(num):
#     record = segmentation_df_example.iloc[i, :]
#
#     img = mpimg.imread(record.full_path, format='png')
#     ax[i, 0].imshow(img)
#     ax[i, 0].set_title(record.id)
#
#     mask = np.zeros(img.shape)
#     for j, cl in enumerate(classes):
#         mask += rle_decode(record[cl], img.shape) * (j + 1) / 4 * np.max(img)
#     ax[i, 1].imshow(mask)
#
#     ax[i, 2].imshow(img + mask)