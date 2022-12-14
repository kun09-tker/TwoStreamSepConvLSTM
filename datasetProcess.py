import os
import math
import random
import shutil
import cv2
from tqdm import tqdm
import numpy as np


def train_test_split(dataset_name=None, source=None, test_ratio=.20):
    assert (dataset_name == 'hockey' or dataset_name == 'movies' or dataset_name == 'surv')
    fightVideos = []
    nonFightVideos = [] 
    for filename in os.listdir(source):
        filepath = os.path.join(source, filename)
        if filename.endswith('.avi') or filename.endswith('.mpg') or filename.endswith('.mp4'):
            if dataset_name == 'hockey':
                if filename.startswith('fi'):
                    fightVideos.append(filepath)
                else:
                    nonFightVideos.append(filepath)
            elif dataset_name == 'movies':
                if 'fi' in filename:
                    fightVideos.append(filepath)
                else:
                    nonFightVideos.append(filepath)
    random.seed(0)
    random.shuffle(fightVideos)
    random.shuffle(nonFightVideos)
    fight_len = len(fightVideos)
    split_index = int(fight_len - (fight_len*test_ratio))
    trainFightVideos = fightVideos[:split_index]
    testFightVideos = fightVideos[split_index:]
    trainNonFightVideos = nonFightVideos[:split_index]
    testNonFightVideos = nonFightVideos[split_index:]
    split = trainFightVideos, testFightVideos, trainNonFightVideos, testNonFightVideos
    return split


def move_train_test(dest, data):
    trainFightVideos, testFightVideos, trainNonFightVideos, testNonFightVideos = data
    trainPath = os.path.join(dest, 'train')
    testPath = os.path.join(dest, 'test')
    os.makedirs(trainPath)
    os.makedirs(testPath)
    trainFightPath = os.path.join(trainPath, 'fight')
    trainNonFightPath = os.path.join(trainPath, 'nonFight')
    testFightPath = os.path.join(testPath, 'fight')
    testNonFightPath = os.path.join(testPath, 'nonFight')
    os.makedirs(trainFightPath)
    os.makedirs(trainNonFightPath)
    os.makedirs(testFightPath)
    os.makedirs(testNonFightPath)
    print("moving files...")
    for filepath in trainFightVideos:
        shutil.copy(filepath, trainFightPath)
    print(len(trainFightVideos), 'files have been copied to', trainFightPath)
    for filepath in testFightVideos:
        shutil.copy(filepath, testFightPath)
    print(len(trainNonFightVideos), 'files have been copied to', trainNonFightPath)
    for filepath in trainNonFightVideos:
        shutil.copy(filepath, trainNonFightPath)
    print(len(testFightVideos), 'files have been copied to', testFightPath)
    for filepath in testNonFightVideos:
        shutil.copy(filepath, testNonFightPath)
    print(len(testNonFightVideos), 'files have been copied to', testNonFightPath)


def crop_img_remove_black(img, x_crop, y_crop, y, x):
    x_start = x_crop
    x_end = x - x_crop
    y_start = y_crop
    y_end = y-y_crop
    frame = img[y_start:y_end, x_start:x_end, :]
    # return img[44:244,16:344, :]
    return frame


def uniform_sampling(video, target_frames=64,resize=320,interval=5):
    # get total frames of input video and calculate sampling interval
    len_frames = video.shape[0]
    if interval == 0:
        interval = int(len_frames//target_frames)
    # init empty list for sampled video and
    sampled_video = []
    for i in range(0, len_frames, interval):
        sampled_video.append(video[i])
    # calculate numer of padded frames and fix it
    if (target_frames - len(sampled_video)) > 3:
      return None
    num_pad = target_frames - len(sampled_video)%target_frames
    padding = []
    if num_pad > 0 and num_pad <= 3:
        for i in range(-num_pad, 0):
            try:
                padding.append(video[i])
            except:
                padding.append(video[0])
        sampled_video += padding
    else:
        sampled_video = sampled_video[:target_frames*(len(sampled_video)//target_frames)]
    # get sampled video
    return np.array(sampled_video).reshape(-1, target_frames, resize, resize, 3)


def Video2Npy(file_path, resize=320, crop_x_y=None, target_frames=None,interval=5):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
        crop_x_y: black boundary cropping
        target_frames:
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows 
    """
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    frames = []
    try:
        for i in range(len_frames):
            _, x_ = cap.read()
            if crop_x_y:
                frame = crop_img_remove_black(
                    x_, crop_x_y[0], crop_x_y[1], x_.shape[0], x_.shape[1])
            else:
                frame = x_
            frame = cv2.resize(frame, (resize,resize), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (resize, resize, 3))
            frames.append(frame)
    except Exception as e:
        print("Error: ", file_path, len_frames)
        print(e)
    finally:
        frames = np.array(frames)
        cap.release()
    frames = uniform_sampling(frames, target_frames=target_frames,resize=resize,interval=interval)
    return frames


def Save2Npy(file_dir, save_dir, crop_x_y=None, target_frames=None, frame_size=320, interval=5):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        # Split video name
        video_name = v.split('.')[0]
        # Get src
        video_path = os.path.join(file_dir, v)
        # Load and preprocess video
        datas = Video2Npy(file_path=video_path, resize=frame_size,
                         crop_x_y=crop_x_y, target_frames=target_frames,interval=interval)
        if datas is not None:
          for index,data in enumerate(datas):
              if target_frames:
                  assert (data.shape == (target_frames,
                                      frame_size, frame_size, 3))
              # Get dest
              save_path = os.path.join(save_dir, f'{video_name}_{index}.npy')
              # os.remove(video_path)
              data = np.uint8(data)
              # Save as .npy file
              np.save(save_path, data)
    return None


def convert_dataset_to_npy(src, dest, crop_x_y=None, target_frames=None, frame_size=320,interval=5):
    if not os.path.isdir(dest):
        os.path.makedirs(dest)
    for dir_ in ['train', 'val']:
        for cat_ in ['Fight', 'NonFight']:
            path1 = os.path.join(src, dir_, cat_)
            path2 = os.path.join(dest, dir_, cat_)
            print(path2)
            Save2Npy(file_dir=path1, save_dir=path2, crop_x_y=crop_x_y,
                     target_frames=target_frames, frame_size=frame_size,interval=interval)

def convert_dataset_to_npy_evl(src, dest, crop_x_y=None, target_frames=None, frame_size=320,interval=5):
    if not os.path.isdir(dest):
        os.path.makedirs(dest)
    for cat_ in ['Fight', 'NonFight']:
        path1 = os.path.join(src, "test", cat_)
        path2 = os.path.join(dest, "test", cat_)
        Save2Npy(file_dir=path1, save_dir=path2, crop_x_y=crop_x_y,
                    target_frames=target_frames, frame_size=frame_size, interval=interval)
