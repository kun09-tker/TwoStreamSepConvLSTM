from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import apply_affine_transform
from .pyskl.vis_heatmap import to_heatmap
from mmcv import load, dump
import numpy as np
import os
import random

class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args: 
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """

    def __init__(self, directory, type_part = 'limb',batch_size=1, shuffle=False, target_heatmap=32, resize=224, frame_diff_interval=1, data_augmentation = True, mode="both"): 
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory # địa chỉ của file.pkl (ví dụ file train.pkl)
        self.type_part = type_part # chọn "limb" hay "keypoint"
        self.shuffle = shuffle
        self.data_aug = data_augmentation 
        self.target_heatmap = target_heatmap # số heatmap đầu vào mỗi batch_size (ví dụ là 32)
        self.mode = mode  #['both', 'only_frames', 'only_differences']
        self.resize = resize # chính là H và W (H = W)
        ratio = self.resize / 64
        self.ratio = (ratio, ratio)
        self.frame_diff_interval = frame_diff_interval # sự khác nhau của frame_diff_interval liên tiếp
        self.X_path, self.Y_dict = self.spread_pkl()
        self.shuffle_index()
        return None

    def spread_pkl(self):
        # tạo một folder tên spread_pkl để chứa các file.pkl của một video
        # X_path là mảng chứa địa chỉ của file.pkl theo format spread_pkl/ten_video.pkl
        # Y_dict là danh sách theo format {địa chỉ video:label}
        if not os.path.exists("spread_pkl"):
            os.mkdir("spread_pkl")
        anno = load(self.directory)
        X_path = []
        Y_dict = {}
        for video in anno:
            X_path.append(f"spread_pkl/{video['frame_dir']}.pkl") 
            Y_dict.update({f"spread_pkl/{video['frame_dir']}.pkl":video["label"]})
            dump(video, f"spread_pkl/{video['frame_dir']}.pkl")
        return X_path, Y_dict

    def shuffle_index(self):
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index *
                                    self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)

        return batch_x, batch_y

    def data_generation(self, batch_path):
        # loading X
        batch_frames = []
        batch_differences = []
        if self.mode == "both":
            for x in batch_path:
                lb_data, kps_data = self.load_data(x)
                batch_frames.append(lb_data)
                batch_differences.append(kps_data)
            batch_frames = np.array(batch_frames)
            batch_differences = np.array(batch_differences)
        elif self.mode == "only_frames":
            for x in batch_path:
                data = self.load_data(x)
                batch_frames.append(data)
            batch_frames = np.array(batch_frames) 
        elif self.mode == "only_differences":
            for x in batch_path:
                kps_data = self.load_data(x)
                batch_differences.append(kps_data)
            batch_differences = np.array(batch_differences) 
        # loading Y
        batch_y = [self.Y_dict[x] for x in batch_path]
        batch_y = np.array(batch_y)
        if self.mode == "both":
            return [batch_frames, batch_differences], batch_y
        if self.mode == "only_frames":
            return [batch_frames], batch_y            
        if self.mode == "only_differences":
            return [batch_differences], batch_y
        
    def uniform_sampling(self, data):
        # ví dụ self.target_heatmap = 32 thì sẽ chia video thành 32 khoảng bằng nhau, rồi lấy ngẫu nhiên 1 frame trong từng khoảng đó
        indexes = np.arange(len(data))
        part = int(len(data)/self.target_heatmap)
        indexes_choice = [random.choice(indexes[i*part:(i+1)*part])  for i in range(self.target_heatmap)]
        return np.array([data[idx] for idx in indexes_choice])


    # def uniform_sampling(self, data):
    #     limit = (len(data) - self.target_heatmap*2)
    #     if limit < 0:
    #         random_start = random.choice(range(len(data) - self.target_heatmap + 1))
    #         indexes_choice = [random_start+i for i in range(self.target_heatmap)]
    #     else:
    #         random_start = random.choice(range(limit + 1))
    #         indexes_choice = [random_start+2*i for i in range(self.target_heatmap)]
    #     return np.array([data[idx] for idx in indexes_choice])
        
    
    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video   
    
    def random_rotation(self, video, rg, prob=0.5, row_axis=0, col_axis=1, channel_axis=2,
                        fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video
        theta = np.random.uniform(-rg, rg)
        for i in range(np.shape(video)[0]):
            x = apply_affine_transform(video[i, :, :, :], theta=theta,row_axis=row_axis ,col_axis=col_axis, channel_axis=channel_axis,
                                       fill_mode=fill_mode, cval=cval,
                                       order=interpolation_order)
            video[i] = x

        return video
    
    def downsample(self, video, ratio=0.5):
        nb_return_frame = int(np.floor(ratio * len(video)))
        return_ind = [int(i) for i in np.linspace(1, len(video), num=nb_return_frame)]
        clip = [video[i-1] for i in return_ind]
        return np.concatenate((clip, clip), axis = 0)
    
    def upsample(self, video, ratio=2):
        num_frames = len(video)    
        nb_return_frame = int(np.floor(ratio * len(video)))
        return_ind = [int(i) for i in np.linspace(1, len(video), num=nb_return_frame)]
        clip = [video[i-1] for i in return_ind]
        s = np.random.randint(0,1)
        if s:
            return clip[:num_frames]
        else:
            return clip[num_frames:]

    def upsample_downsample(self, video, prob=0.5):
        s = np.random.rand()
        if s>prob:
            return video
        s = np.random.randint(0,1)
        if s:
            return self.upsample(video)
        else:
            return self.downsample(video)  
    
    def frame_difference(self, video):
        num_frames = len(video)
        k = self.frame_diff_interval
        out = []
        for i in range(num_frames - k):
            out.append(video[i+k] - video[i])
        return np.array(out,dtype=np.float32)
    
    def load_data(self, path):

        if self.mode == "both":
            frames = True
            differences = True
        elif self.mode == "only_frames":
            frames = True
            differences = False
        elif self.mode == "only_differences":
            frames = False
            differences = True
        # load file .pkl của video
        data = load(path)
        # chuyển thành heatmap dạng TxWxH
        data = to_heatmap(data, flag=self.type_part, ratio=self.ratio)
        data = self.uniform_sampling(data)

        if self.data_aug:
            data = self.random_flip(data, prob=0.50)
            data = self.random_rotation(data, rg=25, prob=0.8)
            data = self.upsample_downsample(data,prob=0.5)
        
        if frames:
            data = np.array(data, dtype=np.float32)
            assert (data.shape == (self.target_heatmap,self.resize, self.resize,3)), str(data.shape)

        if differences:
            diff_data = self.frame_difference(data)
            diff_data = np.array(diff_data, dtype=np.float32)
            assert (diff_data.shape == (self.target_heatmap - self.frame_diff_interval, self.resize, self.resize, 3)), str(data.shape)

        if self.mode == "both":
            return data, diff_data
        elif self.mode == "only_frames":
            return data
        elif self.mode == "only_differences":
            return diff_data

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)
