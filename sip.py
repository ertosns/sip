import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
from tqdm import tqdm
import os

mask_input = input('mask in range (0,1), (default 0.6): ')
mask_input = 0.6 if mask_input == '' else mask_input
mask = float(mask_input)
assert mask > 0 and mask <1
value_mask_input = input('value mask in range (0,1), (default 0.9): ')
value_mask_input = 0.9 if value_mask_input == '' else value_mask_input
value_mask = float(value_mask_input)
assert value_mask > 0 and value_mask <1


dir = 'img'
paths = [path for path in os.listdir(dir)]
imgs = filter(lambda x: True if x[-4:]=='.jpg' or x[-4:]=='.png' else False, paths)

def threshold_checker(image):
    thresholds =  np.arange(0.1,1.1,0.1)
    tree_gray = rgb2gray(image)
    fig, ax = plt.subplots(2, 5, figsize=(17, 10))
    for n, ax in enumerate(ax.flatten()):
        ax.set_title(f'Threshold  : {round(thresholds[n],2)}',
                       fontsize = 16)
        threshold_tree = tree_gray < thresholds[n]
        ax.imshow(threshold_tree);
        ax.axis('off')
    fig.tight_layout()


def preprocessing(img_path):
    global mask
    global value_mask
    tree = imread(dir+os.sep+img_path)[:,:,:3]
    tree_gray = rgb2gray(tree)
    otsu_thresh = threshold_otsu(tree_gray)
    tree_binary = tree_gray < otsu_thresh
    tree_hsv = rgb2hsv(tree[:,:,:])
    lower_mask = tree_hsv [:,:,0] > mask
    upper_mask = tree_hsv [:,:,0] <= 1.00
    value_mask = tree_hsv [:,:,2] < value_mask
    mask = median_filter(upper_mask*lower_mask*value_mask, 10)
    red = tree[:,:,0] * mask
    green = tree[:,:,1] * mask
    blue = tree[:,:,2] * mask
    tree_mask = np.dstack((red, green, blue))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    tree_blobs = label(rgb2gray(tree_mask) > 0)
    properties =['area','bbox','convex_area','bbox_area',
             'major_axis_length', 'minor_axis_length',
             'eccentricity']
    df = pd.DataFrame(regionprops_table(tree_blobs, properties = properties))

    blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] )for
                    index, row in df.iterrows()]
    fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi = 80)
    for blob in tqdm(blob_coordinates):
        width = blob[3] - blob[1]
        height = blob[2] - blob[0]
        patch = Rectangle((blob[1],blob[0]), width, height,
                       edgecolor='r', facecolor='none')
        ax.add_patch(patch)
    ax.imshow(tree);
    ax.set_axis_off()
    plt.savefig('sip_'+str(len(df.index))+"_"+img_path)
    df = df[df['eccentricity'] < df['eccentricity'].max()]
    fig, ax = plt.subplots(1, len(blob_coordinates), figsize=(15,5))
    for n, axis in enumerate(ax.flatten()):
        axis.imshow(tree[int(blob_coordinates[n][0]):
                         int(blob_coordinates[n][2]),
                         int(blob_coordinates[n][1]):
                         int(blob_coordinates[n][3])]);

    plt.close()

for img  in imgs:
    preprocessing(img)
