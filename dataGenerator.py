from tensorflow.keras.utils import Sequence
from .pyskl.vis_heatmap import to_pseudo_heatmap
from mmcv import load, dump
from tqdm import tqdm
import numpy as np
import cv2
import os

class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args: 
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """

    def __init__(self, directory, batch_size=1, shuffle=False, target_heatmap=32, resize=224, mode="both"): 
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.target_heatmap = target_heatmap
        self.mode = mode  # ["only_limbs","only_keypoints", "both"]
        self.resize = resize

        self.X_name, self.Y_dict = self.spread_pkl()
        self.shuffle_index()
        return None

    def spread_pkl(self):
        if not os.path.exists("spread_pkl"):
            os.mkdir("spread_pkl")
        anno = load(self.directory)
        X_name = []
        Y_dict = {}
        for video in anno:
            X_name.append(video["frame_dir"])
            Y_dict.update({video["frame_dir"]:video["label"]})
            dump(video, f"spread_pkl/{video['frame_dir']}.pkl")
        return X_name, Y_dict

    def shuffle_index(self):
        self.indexes = np.arange(len(self.X_name))
        np.random.shuffle(self.indexes)

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_name) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index *
                                    self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_name = [self.X_name[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_name)
        print(batch_x.shape, batch_y.shape)
        return batch_x, batch_y

    def data_generation(self, batch_name):
        # loading X
        batch_limbs = []
        batch_keypoints = []
        if self.mode == "both":
            for x in batch_name:
                lb_data, kps_data = self.load_data(x)
                batch_limbs.append(lb_data)
                batch_keypoints.append(kps_data)
            batch_limbs = np.array(batch_limbs)
            batch_keypoints = np.array(batch_keypoints)
        elif self.mode == "only_limbs":
            for x in batch_name:
                data = self.load_data(x)
                batch_limbs.append(data)
            batch_limbs = np.array(batch_limbs) 
        elif self.mode == "only_keypoints":
            for x in batch_name:
                kps_data = self.load_data(x)
                batch_keypoints.append(kps_data)
            batch_keypoints = np.array(batch_keypoints) 
        # loading Y
        batch_y = [self.Y_dict[x] for x in batch_name]
        batch_y = np.array(batch_y)
        if self.mode == "both":
            return [batch_limbs, batch_keypoints], batch_y
        if self.mode == "only_limbs":
            return [batch_limbs], batch_y            
        if self.mode == "only_keypoints":
            return [batch_keypoints], batch_y
    
    def load_data(self, name):

        if self.mode == "both":
            limbs = True
            keypoints = True
        elif self.mode == "only_limbs":
            limbs = True
            keypoints = False
        elif self.mode == "only_keypoints":
            limbs = False
            keypoints = True

        if limbs:
            video = load(f"spread_pkl/{name}.pkl")
            heatmaps = to_pseudo_heatmap(video, flag="limb")
            heatmaps = heatmaps.transpose(1, 0, 2, 3)
            _ ,_ , h, w = heatmaps.shape
            ratio = self.resize / h
            newh, neww = int(h * ratio), int(w * ratio)
            data_limbs = []
            for hm in heatmaps:
                data_limbs.append([cv2.resize(x, (neww, newh)) for x in hm])
            data_limbs = np.array(data_limbs).transpose(0, 2, 3, 1)

            # uniform_sampling
            indexes = np.arange(len(data_limbs))
            np.random.shuffle(indexes)
            indexes = np.sort(indexes[:self.target_heatmap])
            data_limbs = np.array([data_limbs[i] for i in indexes])

        if keypoints:
            video = load(f"spread_pkl/{name}.pkl")
            heatmaps = to_pseudo_heatmap(video, flag="keypoint")
            heatmaps = heatmaps.transpose(1, 0, 2, 3)
            _ ,_ , h, w = heatmaps.shape
            ratio = self.resize / h
            newh, neww = int(h * ratio), int(w * ratio)
            data_kps = []
            for hm in heatmaps:
                data_kps.append([cv2.resize(x, (neww, newh)) for x in hm])
            data_kps = np.array(data_kps).transpose(0, 2, 3, 1)

            # uniform_sampling
            indexes = np.arange(len(data_kps))
            np.random.shuffle(indexes)
            indexes = np.sort(indexes[:self.target_heatmap])
            data_kps = np.array([data_kps[i] for i in indexes])
            
        if self.mode == "both":
            return data_limbs, data_kps
        elif self.mode == "only_limbs":
            return data_limbs
        elif self.mode == "only_keypoints":
            return data_kps

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)
