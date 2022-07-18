import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

Cross_Subject = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]

import functools

# ntu mean: np.array([0.14869209, 0.51100601, 5.01102056])
# ntu std:  np.array([0.97679044, 0.97010849, 0.82271729])

def pc_norm(data,mean,std):
    return (data-mean)/std

class NTU60Subject(Dataset):
    def __init__(self, root, meta, frames_per_clip=23, step_between_clips=1, num_points=2048, train=True, suffix_file='.npy'):
        super(NTU60Subject, self).__init__()
        self.mean = np.array([0.14869209, 0.51100601, 5.01102056])
        self.std = np.array([1.0, 1.0, 1.0])
        self.videos = []
        self.labels = []
        self.index_map = []
        self.min_nframes = step_between_clips*(frames_per_clip-1)+1
        index = 0

        with open(meta, 'r') as f:
            for line in f:
                name, nframes = line.split()
                if int(nframes) == 0:
                    continue

                subject = int(name[9:12])
                if train:
                    if subject in Cross_Subject:
                        label = int(name[-3:]) - 1
                        nframes = int(nframes)
                        # use all video of train set!! necessary for eval!! not necessary for train!!
                        if nframes < self.min_nframes: # 24
                            nframes = self.min_nframes
                        for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips*2):
                            self.index_map.append((index, t))
                        index += 1
                        self.labels.append(label)
                        self.videos.append(os.path.join(root, name + suffix_file))
                else:
                    if subject not in Cross_Subject:
                        label = int(name[-3:]) - 1
                        nframes = int(nframes)
                        # use all video of val set!! necessary for eval!! not necessary for train!!
                        if nframes < self.min_nframes: # 24
                            nframes = self.min_nframes
                        for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                            self.index_map.append((index, t))
                        index += 1
                        self.labels.append(label)
                        self.videos.append(os.path.join(root, name + suffix_file))

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1


    def __len__(self):
        return len(self.index_map)
    @functools.lru_cache(100)
    def get_video_test(self, video_path):
        # return np.load(video_path, allow_pickle=True)['data'] * 100
        video = np.load(video_path, allow_pickle=True) * 100
        # use all video of val set!! necessary for eval!! not necessary for train!!
        if video.shape[0] < self.min_nframes:
            video_copy = []
            for i in range(int(video.shape[0])):
                video_copy.append(video[i])
            for i in range(self.min_nframes - int(video.shape[0])):
                video_copy.append(np.zeros_like(video[-1]))
            video = np.asarray(video_copy, dtype=object)
        return video

    @functools.lru_cache(100)
    def get_video(self, video_path):
        # return np.load(video_path, allow_pickle=True)['data'] * 100
        video = np.load(video_path, allow_pickle=True) * 100
        # use all video of train set!! necessary for eval!! not necessary for train!!
        if video.shape[0] < self.min_nframes:
            video_copy = []
            for i in range(int(video.shape[0])):
                video_copy.append(video[i])
            for i in range(self.min_nframes - int(video.shape[0])):
                video_copy.append(np.zeros_like(video[-1]))
            video = np.asarray(video_copy, dtype=object)
        return video

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video_path = self.videos[index]
        if self.train:
            video = self.get_video(video_path)
        else:
            video = self.get_video_test(video_path)
        label = self.labels[index]

        clip = [video[t+i*self.step_between_clips] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)
        # clip = pc_norm(clip, self.mean, self.std)
        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales
            shifts = np.random.uniform(-0.1, 0.1, size=3)
            clip += shifts
            clip = self.point_transform(clip)

        return clip.astype(np.float32), label, index

    def point_transform(self, points_xyz):

        # input: temporal_num*2048*3

        y = (np.random.rand()*2-1)*np.pi/3 # -60 ~ 60
        anglesX = (np.random.uniform() - 0.5) * (1 / 9) * np.pi # -10 ~ 10
        R_y = np.array([[[np.cos(y), 0.0, np.sin(y)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(y), 0.0, np.cos(y)]]])
        R_x = np.array([[[1, 0, 0],
                         [0, np.cos(anglesX), -np.sin(anglesX)],
                         [0, np.sin(anglesX), np.cos(anglesX)]]])
        R = np.matmul(R_y, R_x) #3x3

        points_xyz = np.matmul(points_xyz, R) #temporal_num*2048*3
        return points_xyz

if __name__ == '__main__':
    dataset = NTU60Subject(root='../data/ntu/npy_faster/point_reduce_without_sample', meta='./ntu/ntu60.list', frames_per_clip=16)
    print(len(dataset.videos))
    dataset = NTU60Subject(root='../data/ntu/npy_faster/point_reduce_without_sample', meta='./ntu/ntu60.list', frames_per_clip=16, train=False)
    print(len(dataset.videos))
    clip, label, video_idx = dataset[0]
    data = clip[0]
    print(data[:,0].max()-data[:,0].min())
    print(data[:,1].max()-data[:,1].min())
    print(data[:,2].max()-data[:,2].min())
    #print(clip)
    print(label)
    print(video_idx)
    print(dataset.num_classes)
