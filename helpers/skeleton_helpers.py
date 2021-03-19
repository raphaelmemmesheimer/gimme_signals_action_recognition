import os
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib
import math
from matplotlib import animation
from IPython.display import HTML
import numpy as np
from . import read_skeleton
import pandas as pd
import shutil
from tqdm.auto import tqdm

# def generate_color_map(n, name='hsv'):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     return plt.cm.get_cmap(name, n)

def get_sorted_files_in_folder(folder):
    """This function will return a sorted list of files. This is beneficial as
    for iterating over skeletons and files in general as subjects will be generated
    after each other instead just random. That allows inspection and visual
    comparion throughout the running.

    :folder: Input folder
    :returns: sorted list of files

    """
    directory = os.fsencode(folder)
    files = sorted(os.listdir(directory))
    return files

# https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours


def get_ntu_cross_subject_train_subjects():
    """
    This function will return the train subjects, all subjects not listed can be
    expected to be test subjects. This is following the test/train protocol from
    the NTU 120 dataset paper and described as cross subject protocoll.
    The paper can be found here: https://arxiv.org/pdf/1905.04757.pdf
    """
    # based on cross subject protocol ()
    train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                      31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56,
                      57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89,
                      91, 92, 93, 94, 95, 97, 98, 100, 103]
    for i, s in enumerate(train_subjects):
        train_subjects[i] = "P"+str(s).zfill(3)
    return train_subjects

def get_ntu_cross_subject_validation_subjects():
    """
    This function will return the subjects subjects. No validation set is defined in the original
    NTU paper, we therefore consider every second subject from the train subjects as validation
    subject.
    """
    # based on cross subject protocol ()
    validation_subjects = [1, 4,  8,  13,  15,  17, 19,  27, 31, 35, 45, 47, 50, 53, 55, 
                      57, 59, 74, 80, 82, 84, 86, 
                      91, 93, 95, 98,  103]
    for i, s in enumerate(validation_subjects):
        validation_subjects[i] = "P"+str(s).zfill(3)
    return validation_subjects


def get_ntu_train_setups_cross_setup():
    """
    This function returns the train setups as defined in the cross view / cross setup 
    protcol.
    """
    # based on cross subject protocol ()
    train_setups = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
                    30, 32]
    for i, s in enumerate(train_setups):
        train_setups[i] = "S"+str(s).zfill(3)
    return train_setups

def get_ntu_validation_setups_cross_setup():
    """
    This function returns the validation setups as defined in the cross view / cross setup 
    protcol.
    """
    # based on cross subject protocol ()
    train_setups = [2, 6, 10, 14, 18, 22, 26, 30]
    for i, s in enumerate(train_setups):
        train_setups[i] = "S"+str(s).zfill(3)
    return train_setups

def get_ntu_test_actions_one_shot():
    """
    This function returns the train action classes. The NTU RGB+D dataset authors defined the 
    one-shot protocoll as using 100 actions for training (auxilary) and 20 action classes as 
    evaluation set. The evaluation classes are taken from the github repository 
    (https://github.com/shahroudy/NTURGB-D)
    """
    test_actions = ["A001", "A007", "A013", "A019", "A025", "A031", "A037", "A043",
                    "A049", "A055", "A061", "A067", "A073", "A079", "A085", "A091",
                    "A097", "A103", "A109", "A115"]
    return test_actions

def get_ntu_val_actions_one_shot(start=2,increment=6):
    """
    (https://github.com/shahroudy/NTURGB-D)
    """
    if start == 1 and incement == 6:
        raise ValueError("start == 1 and increment == 5 is already used for the one-shot test set")
    val_actions = []
    val_samples = []
    for i in range(start,120,increment):
        action = "A"+str(i).zfill(3)
        val_actions.append(action) 
        sample_file = "S001C003P008R001%s"%(action) if i < 60 else "S018C003P008R001%s"%(action)
        val_samples.append(sample_file)
    return val_actions, val_samples


def get_ntu_one_shot_sample_sequences():
    """
    One shot sample actions as protocolled here: https://github.com/shahroudy/NTURGB-D#evaluation-protocol-of-one-shot-action-recognition-on-ntu-rgbd-120 
    """                                            
    sample_actions = [                             
          "S001C003P008R001A001",
          "S001C003P008R001A007",
          "S001C003P008R001A013",
          "S001C003P008R001A019",
          "S001C003P008R001A025",
          "S001C003P008R001A031",
          "S001C003P008R001A037",
          "S001C003P008R001A043",
          "S001C003P008R001A049",
          "S001C003P008R001A055",
          "S018C003P008R001A061",
          "S018C003P008R001A067",
          "S018C003P008R001A073",
          "S018C003P008R001A079",
          "S018C003P008R001A085",
          "S018C003P008R001A091",
          "S018C003P008R001A097",
          "S018C003P008R001A103",
          "S018C003P008R001A109",
          "S018C003P008R001A115"]
    return sample_actions




def get_test_train_ntu(filename, split="cross_subject"):
    """
    sorts ntu file name depending on the split. 
    Parameters
    ==========
    `filename`: Filename to sort
    `split`: Either be 'cross_subject', 'cross_setup', `one_shot`

    Returns
    =======
    `cat` : Either `train`, `test`, `val` or `samples`, depending on split and filename
    """
    cat = "test"
    if split == "cross_subject":
        subject = filename[8:12]
        if subject in get_ntu_cross_subject_train_subjects():
            cat = "train"
        if subject in get_ntu_cross_subject_validation_subjects():
            cat = "val"
    elif split == "cross_setup":
        setup = filename[:4]
        if setup in get_ntu_train_setups_cross_setup():
            cat = "train"
        if setup in get_ntu_validation_setups_cross_setup():
            cat = "val"
    elif split == "one_shot":
        action = filename[16:20]
        val_actions, val_samples = get_ntu_val_actions_one_shot()
        if action not in get_ntu_test_actions_one_shot() and action not in val_actions:
            cat = "train"
        if action in val_actions:
            cat = "val"
        if filename[0:20] in val_samples:
            cat = "val_samples"
        if filename[0:20] in get_ntu_one_shot_sample_sequences():
            cat = "samples"
    return cat                                                                                                                                                                                                                 

def get_test_train_utdmhad(filename, split="one_shot", aux_size=23, static_val=True):
    """
    sorts the UTDMHAD files into test, train samples. If split is not one_shot the standard protocoll
    as descibed in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7350781 will be used.

    For one_shot we define a custom protocol.

    Parameters
    ==========
    `filename`: Filename to sort
    `split`: Either be 'cross_subject', `one_shot`

    Returns
    =======
    `cat` : Either `train`, `test`, `val` or `samples`, depending on split and filename
    """
    cat = "test"
    if split != "one_shot":
        filename_parts = filename.split("_")
        subject = filename_parts[1]
        if subject not in ["s1", "s3", "s5", "s7"]:
            cat = "train"
    elif split == "one_shot":
        filename_parts = filename.split("_")
        action = filename_parts[0]
        trial = filename_parts[2]
        subject = filename_parts[1]
        #val_actions, val_samples = get_utdmhad_val_actions_one_shot()
        test_action_ids = [*range(aux_size+1, 27+1)]
        test_action_ids_formatted= []
        for i in test_action_ids:
            test_action_ids_formatted.append("a"+str(i))
        if action not in test_action_ids_formatted:
            cat = "train"
        #if action in val_actions:
            #cat = "val"
        #if filename[0:20] in val_samples:
            #cat = "val_samples"
        if subject == "s1" and trial == "t1" and cat == "test":
            cat = "samples"
    return cat                                                                                                                                                                                                                 
                             
                                                                                                                                                                                                                               
def mk_dest_dir(dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)
                             
def get_ntu_class_name_from_filename(filename):
    return filename[16:16+4] 
                             
def sort_ntu_to_test_train_split_in_folder(folder, split="cross_subject", dest_folder=None, copy=False):
    files = get_sorted_files_in_folder(folder)
    for file in files:       
        file = os.fsdecode(file)
        if file.endswith(".png"):
            test_train = get_test_train_ntu(file, split)
            print(file, test_train)
            if dest_folder is None:
                dest_folder = folder
            dest = str(dest_folder)+"/"+str(test_train)+"/"+get_ntu_class_name_from_filename(str(file))
            mk_dest_dir(dest)
            target_filename = dest + "/" + str(file)
            print(file, "sorted to ", test_train, target_filename)
            if copy:
                shutil.copy(folder+"/"+str(file), target_filename)
            else:
                os.rename(folder+"/"+str(file), target_filename)
        else:
            print("File", file, "ignored as not ending with .png")
            
            
def sort_ntu_folder_to_test_train_split_in_folder(folder, split="cross_subject", dest_folder=None, copy=False):
    """
    This method split a dataset folder into the given split.

    
    Structure assumed is:
    
    before:
        <dataset_folder>:
         A001
         ....
         A120
         
    after:
        <dest_folder>:
            one_shot:
                train:
                    A001
                    ....
                    A120
                test:
                    A007
                    ...
                    A115
                samples:
                    A007
                    ...
                    A115
    """
    classes = get_sorted_files_in_folder(folder)
    for class_name in tqdm(classes):       
        class_folder = os.fsdecode(folder+"/"+os.fsdecode(class_name))
        print(class_folder)
        files = get_sorted_files_in_folder(class_folder)
        if dest_folder is None:
            dest_folder = folder
            if not os.path.exits(dest_folder):
                os.mk_dir(dest_folder)
        sort_ntu_to_test_train_split_in_folder(class_folder, split=split, dest_folder=dest_folder, copy=copy)


def get_utdmhad_class_name_from_filename(filename):
    return filename.split("_")[0]

def sort_utdmhad_to_test_train_split_in_folder(folder, split="one_shot", aux_size=23):
    files = get_sorted_files_in_folder(folder)
    for file in files:       
        file = os.fsdecode(file)
        if file.endswith(".png"):
            test_train = get_test_train_utdmhad(file, split, aux_size=aux_size)
            print(file, test_train)
            dest = str(folder)+"/"+str(test_train)+"/"+get_utdmhad_class_name_from_filename(str(file))
            mk_dest_dir(dest)
            target_filename = dest + "/" + str(file)
            print(file, "sorted to ", test_train, target_filename)
            os.rename(folder+"/"+str(file), target_filename)
        else:
            print("File", file, "ignored as not ending with .png")

def get_utdmhad_train_subjects():
    """
    This fuction returns test and train subjects for a cross subject test/train 
    split as proposed in the UTD-MHAD paper ()
    Futher information:
        https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
    Paper:
        https://www.utdallas.edu/~cxc123730/ICIP2015-Chen-Final.pdf
    """
    train_subjects = ["s1", "s3", "s5", "s7"]
    # test_subjects = ["s2", "s4", "s6", "s8"]
    return train_subjects#, test_subjects

def generate_color_map(N):
    arr = np.arange(N)/N
    N_up = int(math.ceil(N/7)*7)
    arr.resize(N_up)
    arr = arr.reshape(7, N_up//7).T.reshape(-1)
    ret = matplotlib.cm.hsv(arr)
    n = ret[:, 3].size
    a = n//2
    b = n-a
    for i in range(3):
        ret[0:n//2, i] *= np.arange(0.2, 1, 0.8/a)
    ret[n//2:, 3] *= np.arange(1, 0.1, -0.9/b)
#     print(ret)
    return ret


def get_bone_list(data_source="ntu"):
    if data_source == "ntu":
        bone_list = [
                        [0, 1], [1, 2], [2, 3],  #  BACk
                        [0, 16], [16, 17], [17, 18], [18, 19],  #  left leg
                        [0, 12], [12, 13], [13, 14], [14, 15], # right leg
                        [20, 8], [8, 9], [9, 10], [10, 11], [11, 24], [11, 23], # left arm
                        [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22]
                    ]
    elif data_source == "utd":
        bone_list = [
                        [0, 1], [1, 2], [2, 3], # BACk
                        [3, 12], [12, 13], [13, 14], [14, 15], # left leg
                        [3, 16], [16, 17], [17, 18], [18, 19], # right leg
                        [1, 4], [4, 5], [5, 6], [6, 7],  # left arm
                        [1, 8], [8, 9], [9, 10], [10, 11] # right arm
                ]
    return bone_list


def plot_skeleton(skeleton_data, index, size=4):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    data_source = get_data_source_from_data(skeleton_data)
    bone_list = get_bone_list(data_source)
    color_map = generate_color_map(len(bone_list)*3)
    data = skeleton_data[:, :, index].transpose()
    mean_z = data[2].mean()
    x = data[0]
    y = data[1]
    z = data[2] - mean_z
    for i, bone in enumerate(bone_list):
        # If you get an error here, most likely you have given the wrong
        # data_source
        ax.plot3D([x[bone[0]], x[bone[1]]],
                  [y[bone[0]], y[bone[1]]],
                  [z[bone[0]], z[bone[1]]],  "-",
                  linewidth=size, c=color_map[i*3], zdir="x")
        ax.scatter3D([x[bone[0]], x[bone[1]]],
                     [y[bone[0]], y[bone[1]]],
                     [z[bone[0]], z[bone[1]]], s=10, zdir="x")
    return fig


def get_data_source_from_data(skeleton_data):
    return "utd" if len(skeleton_data) == 20 else "ntu"

def load_skeletons_ntu(filename):
    a = read_skeleton.read_skeleton(filename)
    skeletons = []
    try:
        num_skeletons = 0 
        for f in a["frameInfo"]:
            num_skeletons = max(num_skeletons, f["numBody"])
        #print("Skeletons found", num_skeletons)
        skeleton_data = np.empty([len(a["frameInfo"][0]["bodyInfo"][0]["jointInfo"]), 3, len(a["frameInfo"])])
        for i in range(num_skeletons):
            skeletons.append(skeleton_data.copy())
        for frame_id, frame in enumerate(a["frameInfo"]):
            #print(frame_id)
            num_skeletons_in_frame = len(a["frameInfo"][frame_id]["bodyInfo"])
            for skeleton_id in range(num_skeletons_in_frame):
                for joint_id, joint in enumerate(frame["bodyInfo"][skeleton_id]["jointInfo"]):
                    #aa = skeletons[skeleton_id]
                    #bb = skeletons[skeleton_id][joint_id]
                    skeletons[skeleton_id][joint_id][0][frame_id] = joint["x"]
                    skeletons[skeleton_id][joint_id][1][frame_id] = joint["y"]
                    skeletons[skeleton_id][joint_id][2][frame_id] = joint["z"]
    except Exception as e:
        print ("Error: ", e)
    return skeletons

def load_skeletons_pkummd(filename):
    skeletons = []
    dataframe = pd.read_csv(filename, skiprows=0, sep=' ', header=None)
    num_skeletons = 0
    try:
    	for d in dataframe.values:
    	#     print(d)
    	    num_skeletons_current_frame = 1 if d[75] == 0 else 2
    	    num_skeletons = max(num_skeletons_current_frame, num_skeletons)
    	skeleton_data = np.empty([25, 3, len(dataframe)])
    	for i in range(num_skeletons):
    	    skeletons.append(skeleton_data.copy())
    	for frame_id, frame in enumerate(dataframe.values):
    	    #print(frame_id)
    	    num_skeletons_in_frame = 1 if frame[75] == 0 else 2
    	    for skeleton_id in range(num_skeletons_in_frame):
    	        for joint_id in range(25):
                    start_column = skeleton_id*25+joint_id*3
                    skeletons[skeleton_id][joint_id, 0, frame_id] = frame[start_column]
                    skeletons[skeleton_id][joint_id, 1, frame_id] = frame[start_column+1]
                    skeletons[skeleton_id][joint_id, 2, frame_id] = frame[start_column+2]
    except Exception as e:
        print ("Error: ", e)
    return skeletons


def load_skeletons(filename):
    _, file_extension = os.path.splitext(filename)
    if file_extension == ".skeleton": # handle NTU format
        return load_skeletons_ntu(filename)
    if file_extension == ".txt": # handle PKU-MMD
        return load_skeletons_pkummd(filename)


def animate_all_in_dir(directory, overwrite_if_existing=True):
    files = sorted(os.listdir(directory))
#    try:
    for i, file in enumerate(files):
        filename = os.fsdecode(file)
        dest_filename = filename+".mp4"
        print(dest_filename)
        if os.path.isfile(dest_filename) and not overwrite_if_existing:
            print("Skipping %s as file already existing" % dest_filename)
            continue

        print(str(i)+"/"+str(len(files)), filename[:1], filename)
        if filename.endswith(".skeleton"):
            skeletons = load_skeletons(directory + "/" + filename)
            # print(skeleton_data.shape, len(skeleton_data))
            a = SkeletonAnimation(skeletons, clear_frames=True)
            # a.animate()
            # x = a.animate_html5()
            a.save_as_mp4(dest_filename)
    #except Exception as e:
        #print(e)

class Anim():

    def __init__(self):
        self.fig = plt.figure(figsize=(10, 10))
        pass

    def animate(self):
        pass

    def animate_html5(self):
        return HTML(self.animate().to_html5_video())

    def save_as_mp4(self, filename, dpi=300):
        return self.animate().save(filename, writer=self.writer, dpi=dpi)


class SkeletonSignalAnimation(Anim):

    def __init__(self, skeleton_data, clear_frames=True):
        super().__init__()
        self.skeleton_data = skeleton_data
        self.clear_frames = clear_frames
        self.ax = self.fig.add_subplot(frameon=False)
        self.writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Raphael Memmesheimer'),
                                            bitrate=1800)

        data_source = get_data_source_from_data(skeleton_data)
        bone_list = get_bone_list(data_source)
        self.color_map = generate_color_map(len(bone_list)*3)
        self.data_source = get_data_source_from_data(skeleton_data[0])


    def plot_signals(self, skeleton_data, num):
        for joint_index, joint_data in enumerate(skeleton_data):
            alpha = 0.5
            color_offset = len(joint_data)
            self.ax.plot(range(num), joint_data[0][:num],  "-", linewidth=1, c=self.color_map[joint_index], alpha=alpha)
            self.ax.plot(range(num), joint_data[1][:num], "-", linewidth=1, c=self.color_map[color_offset+joint_index], alpha=alpha)
            self.ax.plot(range(num), joint_data[2][:num], "-", linewidth=1, c=self.color_map[2*color_offset+joint_index], alpha=alpha)

        self.ax.axis("off")
        self.ax.set_xlim(0, len(skeleton_data.transpose()))
        self.ax.set_ylim(np.ndarray.min(skeleton_data), np.ndarray.max(skeleton_data))
        plt.box(False)
        plt.close()


    def update(self, num):
        if self.clear_frames:
            self.ax.clear()
        self.plot_signals(self.skeleton_data, num)


    def animate(self):
        length = len(self.skeleton_data.transpose())
        print("Length", length)
        ani = animation.FuncAnimation(self.fig, self.update,
                                      length,
                                      interval=10000/length, blit=False)
        return ani




class SignalAnimation(Anim):

    def __init__(self, signals, clear_frames=True):
        super().__init__()
        self.signals = signals
        self.clear_frames = clear_frames
        self.ax = self.fig.add_subplot(frameon=False)
        self.writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Raphael Memmesheimer'),
                                            bitrate=1800)
        self.color_map = generate_color_map(len(signals))


    def plot_signals(self, signals, num):
        print(num, signals.shape, len(signals.shape))
        for j in range(len(signals)):
            self.ax.plot(range(0, num), signals[j][:num], "-", linewidth=1, c=self.color_map[j])
        self.ax.axis("off")
        self.ax.set_xlim(0, len(signals[0]))
        self.ax.set_ylim(np.ndarray.min(signals), np.ndarray.max(signals))
        plt.box(False)
        plt.close()


    def update(self, num):
        if self.clear_frames:
            self.ax.clear()
        self.plot_signals(self.signals, num)


    def animate(self):
        length = len(self.signals[0])
        ani = animation.FuncAnimation(self.fig, self.update,
                                      length,
                                      interval=10000/length, blit=False)
        return ani



class SkeletonAnimation(Anim):
    """docstring for SkeletonAnimation"""
    def __init__(self, skeletons, clear_frames=True, alpha_blend=False, skip_every=0, seperate_spatial=False, show_axes=True, color=None):
        super().__init__()
        self.alpha_blend = alpha_blend
        self.skip_every = skip_every
        self.skeletons = skeletons
        self.seperate_spatial = seperate_spatial
        self.show_axes = show_axes
        self.length = len(self.skeletons[0].transpose())
        self.data_source = get_data_source_from_data(skeletons[0])
        self.fig = plt.figure()
        self.clear_frames = clear_frames
        self.ax = p3.Axes3D(self.fig)
        if not self.show_axes:
            self.ax.set_axis_off() 
        self.bone_list = get_bone_list(data_source=self.data_source)
        self.color = color
        if color is None:
            self.color_map = generate_color_map(len(self.bone_list)*3)
        self.writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Raphael Memmesheimer'),
                                            bitrate=1800)

    def plot_skeleton(self, skeleton_data, num, alpha=1.0, offset_x=0.0):
        for bone_index, bone in enumerate(self.bone_list):
            x = skeleton_data[num][0] - skeleton_data[0].mean()
            y = skeleton_data[num][1] - skeleton_data[1].mean() 
            z = skeleton_data[num][2] - skeleton_data[2].mean() + offset_x
            if self.color is None:
                c = self.color_map[bone_index],
            else: 
                #print(self.color)
                c = self.color
                #c=(1,0,0, alpha),
            self.ax.plot3D([x[bone[0]], x[bone[1]]],
                           [y[bone[0]], y[bone[1]]],
                           [z[bone[0]], z[bone[1]]], zdir="x",
                           c=c,
                           alpha=alpha)
        plt.close()

    def update(self, num):
        #print(num, data.shape)
        if (num % self.skip_every):
            return
        if self.clear_frames:
            self.ax.clear()
        self.ax.view_init(35, 130)
    #   self.  ax.ylim ()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        if not self.show_axes:
            self.ax.set_axis_off() 

        self.ax.set_zlim(-0, 1.0)
        self.ax.set_ylim(-0, 1.0)
        self.ax.set_xlim(-0, 1.0)

        for skeleton_id, skeleton in enumerate(self.skeletons):
            skeleton_data = skeleton.transpose()
            offset_x = -num * 2 if self.seperate_spatial else 0.0
            if self.alpha_blend:
                self.plot_skeleton(skeleton_data, num, alpha=num/self.length, offset_x=offset_x)
            else:
                self.plot_skeleton(skeleton_data, num, offset_x=offset_x)
        #self.plot_skeleton(self.skeletons[0].transpose(), num)

    #         ax.scatter3D([x[bone[0]], x[bone[1]]],
    #                      [y[bone[0]], y[bone[1]]],
    #                      [z[bone[0]], z[bone[1]]],s=2, zdir="x")

    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.update,
                                      self.length,
                                      interval=10000/self.length, blit=False)
        return ani


