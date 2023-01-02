
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: ../trajectory_classification.ipynb

# %matplotlib inline
# %matplotlib widget
import scipy.io as sio
import math
import numpy as np
import time
import os
import zipfile
import pickle
import urllib.request
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import read_skeleton
import helpers.skeleton_helpers
from helpers.skeleton_helpers import SkeletonAnimation
# import visdom
import pyts
from pyts.multivariate.image import JointRecurrencePlot

matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.bottom'] = False



# matplotlib.use('nbagg')

def filter_skeleton(skeleton_data):
    # filter foot # no infomration worth there and the estimates are very noisy
    filtered_skeleton = skeleton_data
    filtered_skeleton[15] = filtered_skeleton[14]
    filtered_skeleton[19] = filtered_skeleton[18]
    return filtered_skeleton

def filter_attention(skeleton_data, filter_percentage):
    dtype = [('index', int), ('var', float), ('std', float)]
    filtered_skeleton = skeleton_data
    values = []
    for i, col in enumerate(skeleton_data):
        values.append((i, col.var(), col.std()))
    a = np.array(values, dtype=dtype)
    a = np.sort(a, order='std')
#     print("Sorted by std deviation")
#     print(a)
    a = np.sort(a, order='std')
#     print("Sorted by variance")
#     reverse_array = a[::-1]
#     print(reverse_array)

#     print("filtered by importance (low std_deviation)")
    to_remove = []
    for i, dic in enumerate(a[::-1]):
#         print(i, dic)
        if i > len(a) * (1-filter_percentage):
            break
#         print(dic[0])
        to_remove.append(dic[0])
#     print(to_remove)
#     print(filtered_skeleton.shape)
#     return np.delete(filtered_skeleton, to_remove, 0) # this is not good, same signals can result in differnt color encoding
#     filtered = a.copy()
    for x in to_remove:
        filtered_skeleton[x] = np.zeros(filtered_skeleton[x].shape)
#     print(filtered_skeleton)
    return filtered_skeleton




def plot_skeleton_data(skeleton_data, size=5, offset=False, approach_id=1, alpha=0.2, attention=False, filter_percentage=0.2):
    fig = plt.figure(figsize=(5,5))
#   fig.title(filename)
    ax = fig.add_subplot(frameon=False)
#     ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
#     ax = plt.figure(frameon=False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(0.0)
#     ax.spines['left'].set_linewidth(0.0)
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)



    if approach_id == 3:
        skeleton_data = np.sort(skeleton_data, axis=0)

    if approach_id == 8:
        skeleton_data = filter_skeleton(skeleton_data)

    if approach_id == 9:
        print("before filter, ",skeleton_data.shape)
        skeleton_data = filter_attention(skeleton_data)
        print("after filter, ",skeleton_data.shape)

    if attention:
        print("before filter, ",skeleton_data.shape)
        skeleton_data = filter_attention(skeleton_data, filter_percentage)
        print("after filter, ",skeleton_data.shape)

    for joint_index, joint_data in enumerate(skeleton_data):
        # line currently the best results
        if approach_id ==1:
            of1 = 0
            of2 = 2
            of3 = 3
        if approach_id ==2:
            of1=joint_index/2+offset
            of2=joint_index/2+offset
            of3=joint_index/2+offset
        if approach_id ==3:
            of1=joint_index/2+offset
            of2=joint_index/2+offset
            of3=joint_index/2+offset

            if joint_index > len(skeleton_data) / 6:
                continue
        if approach_id == 4:
            of1 = joint_index/2
            of2 = joint_index/2
            of3 = joint_index/2
            if joint_data.var() > 1:
                continue
        if approach_id == 5:
            import pyts
            from pyts.image import GramianAngularField
            transformer = GramianAngularField(image_size=len(joint_data))
            image_data = transformer.fit_transform(joint_data)
            plt.figure(figsize=(10,10))
            plt.imshow(image_data)
            plt.savefig("gaf.png")
            continue
        if approach_id == 6:
            of1 = -np.average(joint_data[0])
            of2 = -np.average(joint_data[1])
            of3 = -np.average(joint_data[2])
        if approach_id == 7:
            of1 = - np.max(joint_data[0])
            of2 = - np.max(joint_data[1])
            of3 = - np.max(joint_data[2])
        if approach_id == 8:
            of1 = - np.max(joint_data[0])
            of2 = - np.max(joint_data[1])
            of3 = - np.max(joint_data[2])
        if approach_id == 9:
            of1 = - np.max(joint_data[0])
            of2 = - np.max(joint_data[1])
            of3 = - np.max(joint_data[2])
        if approach_id == 10:
            of1 = -np.average(joint_data[0])
            of2 = -np.average(joint_data[1])+0.5
            of3 = -np.average(joint_data[2])+1
        if approach_id == 11: # like 10 but use alpha for temporal
            of1 = -np.average(joint_data[0])
            of2 = -np.average(joint_data[1])+0.5
            of3 = -np.average(joint_data[2])+1
            n = len(joint_data[0])
            s = 5 # Segment length
            alpha = 0.1
            for i in range(0,n-s,s):

                alpha = min(alpha+ (s / n), 1.0)
#                 print(alpha)
                ax.plot(range(i,i+s+1), of1+joint_data[0][i:i+s+1],  "-", linewidth=size, c=color_map[joint_index], alpha=alpha);
                ax.plot(range(i,i+s+1), of2+joint_data[1][i:i+s+1], "-", linewidth=size, c=color_map[25+joint_index], alpha=alpha);
                ax.plot(range(i,i+s+1), of3+joint_data[2][i:i+s+1], "-", linewidth=size, c=color_map[50+joint_index], alpha=alpha);
        else:
            ax.plot(range(len(joint_data[0])), of1+joint_data[0],  "-", linewidth=size, c=color_map[joint_index], alpha=alpha);
            ax.plot(range(len(joint_data[1])), of2+joint_data[1], "-", linewidth=size, c=color_map[25+joint_index], alpha=alpha);
            ax.plot(range(len(joint_data[2])), of3+joint_data[2], "-", linewidth=size, c=color_map[50+joint_index], alpha=alpha);
#         ax.view_init(0, 0)

    return fig

color_map = helpers.skeleton_helpers.generate_color_map(50*3)


def get_experiment_string(skeleton_data, size=5, offset=False, approach_id=1, alpha=0.2, attention=False, filter_percentage=0.2):
    if approach_id == 11:
        experiment_string = "approach_id"+str(approach_id).zfill(3)+"_gradient_alpha_size_"+str(size)+("_attention" if attention else "_no_attention")+"_filter_"+str(filter_percentage).zfill(2)
    else:
        experiment_string = "approach_id"+str(approach_id).zfill(3)+"_"+str(alpha)+"_size_"+str(size)+("_attention" if attention else "_no_attention")+"_filter_"+str(filter_percentage).zfill(2)
    return experiment_string


def extract_ntu(input_dataset_folder, dataset_folder, max_show=99999999, overwrite_if_existing=True):
    train_subjects = helpers.skeleton_helpers.get_ntu_train_subjects()
    files = helpers.skeleton_helpers.get_sorted_files_in_folder(input_dataset_folder)

    for i, file in enumerate(files):
        filename = os.fsdecode(file)
        print(str(i)+"/"+str(len(files)), filename[:1], filename)
        if filename.endswith(".skeleton"):
    #         try:
                skeletons = helpers.skeleton_helpers.load_skeletons(input_dataset_folder+"/"+filename)
                if len(skeletons) == 0:
                    print("Warining: No skeletons found")
                    continue
                if len(skeletons) > 1:
                    skeleton_data = np.concatenate((skeletons[0], skeletons[1]), axis=0)
                else:
                    skeleton_data = skeletons[0]
                experiment_string = get_experiment_string(skeleton_data, offset=skeleton_data.var(), approach_id=11, size=1, attention=True, filter_percentage=0.5)

                subject = filename[:4]
                class_name = filename[16:16+4]

                # sort into respective folder
                set_cat = "test"
                if subject in train_subjects:
                    set_cat = "train"
                dest_folder = dataset_folder+experiment_string+"/"+set_cat+"/"+class_name
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder, exist_ok=True)

                print(class_name, subject,set_cat, dest_folder)
                dest_filename = dest_folder+"/"+filename+".png"
                if not os.path.isfile(dest_filename) or overwrite_if_existing:
                    fig = plot_skeleton_data(skeleton_data, offset=skeleton_data.var(), approach_id=11, size=1, attention=True, filter_percentage=0.5)
                    fig.savefig(dest_filename)
                else:
                    print("File existing, skipping to save time, set overwrite_if_existing to True if you want to regenerate")
             #   fig.savefig("skeleton_representation.svg")
    #             fig.savefig(filename+".png")
                if max_show > 60:
                    plt.close()
    #         except Exception as e:
    #             print("Error", e)
        if i > max_show:
            break