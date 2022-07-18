import os
import sys
import numpy as np
from torch.utils.data import Dataset

import functools
# mean: np.array([-0.12453246,  0.10527167,  1.68724235])
# std:  np.array([0.10234013, 0.281423  , 0.09402914])

def pc_norm(data,mean,std):
    return (data-mean)/std

class MSRAction3D(Dataset):
    def __init__(self, root, frames_per_clip=16, step_between_clips=1, num_points=2048, train=True, meta='./datasets/msr/msr_all.list', suffix_file='.npz'):
        super(MSRAction3D, self).__init__()
        self.mean = np.array([-0.12513872,  0.10878776,  1.68986231])
        self.std = np.array([1.0, 1.0, 1.0])
        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0
        with open(meta, 'r') as f:
            for line in f:
                video_name, nframes = line.split()

                video_name = video_name + suffix_file
            #for video_name in os.listdir(root):
                if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                    video = self.get_video(root, video_name)
                    self.videos.append(video)
                    label = int(video_name.split('_')[0][1:])-1
                    self.labels.append(label)
                    assert int(nframes) == video.shape[0]
                    nframes = video.shape[0]
                    for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                        self.index_map.append((index, t))
                    index += 1

                if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
                    video = self.get_video(root, video_name)
                    #use all video of val set!! necessary for eval!! not necessary for train!!
                    need_min_frames = step_between_clips*(frames_per_clip-1)+1 # 24
                    if int(nframes)<need_min_frames:
                        video_copy = []
                        for i in range(int(nframes)):
                            video_copy.append(video[i])
                        for i in range(need_min_frames-int(nframes)):
                            video_copy.append(video[-1])
                        video = np.asarray(video_copy, dtype=object)
                        nframes = str(need_min_frames)

                    self.videos.append(video)
                    label = int(video_name.split('_')[0][1:])-1
                    self.labels.append(label)
                    assert int(nframes) == video.shape[0]
                    nframes = video.shape[0]
                    for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                        self.index_map.append((index, t))
                    index += 1

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1

    @functools.lru_cache(100)
    def get_video(self, root, video_name):
        return np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
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
        clip = clip / 300
        # clip = pc_norm(clip, self.mean, self.std)
        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales
            shifts = np.random.uniform(-0.05, 0.05, size=3)
            clip += shifts



        return clip.astype(np.float32), label, index

if __name__ == '__main__':
    dataset = MSRAction3D(root='../data/MSRAction3D/point', frames_per_clip=16)
    clip, label, video_idx = dataset[0]
    print(clip)
    print(label)
    print(video_idx)
    print(dataset.num_classes)
