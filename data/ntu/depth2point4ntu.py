import os
import numpy as np
import argparse
from matplotlib.image import imread
from glob import glob
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Depth to Point Cloud')

parser.add_argument('--input', default='./nturgb+d_depth_masked', type=str)
parser.add_argument('--output', default='./npy_faster/point_reduce_without_sample', type=str)
parser.add_argument('-n', '--action', type=int)
parser.add_argument('--step', default=2, type=int)
args = parser.parse_args()

fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347

W = 512
H = 424

'''
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

mkdir(args.output)
'''

xx, yy = np.meshgrid(np.arange(W), np.arange(H))
# focal = 280
# print('%s/*A%03d'%(args.input, args.action), len(glob('%s/*A%03d'%(args.input, args.action))))
# print('Action %02d begin!'%args.action)
for video_path in sorted(glob('%s/*A%03d'%(args.input, args.action))):
    video_name = video_path.split('/')[-1]

    point_clouds = []
    for img_name in sorted(os.listdir(video_path))[::args.step]:
        img_path = os.path.join(video_path, img_name)
        img = imread(img_path) # (H, W)

        depth_min = img[img > 0].min()
        depth_map = img

        x = xx[depth_map > 0]
        y = yy[depth_map > 0]
        z = depth_map[depth_map > 0]
        x = (x - cx) / fx * z
        y = (y - cy) / fy * z

        points = np.stack([x, y, z], axis=-1).astype(np.float16)
        point_clouds.append(points)

    np.save(os.path.join(args.output, video_name + '.npy'), np.array(point_clouds, dtype=object))
print('Action %02d finished!'%args.action)
