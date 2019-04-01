#encoding=utf8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import PIL.Image as Image
import cv2
import numpy as np
import sys
sys.path.append("../")
from dataProcess import clip_process
from tqdm import tqdm
import argparse

'''
保存成npy均值文件
'''

# 获取参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="jpl", help="[jpl/dog]")
parser.add_argument("--structureType", type=str, default="local", help="[global/local]")
parser.add_argument("--net", type=str, default="i3d", help="[i3d/c3d]")
parser.add_argument("--clip_length", type=int, default=16, help="[clip_length]")
parser.add_argument("--channel", type=int, default=3, help="[channel]")
args = parser.parse_args()
dataset = args.dataset
structureType = args.structureType
net=args.net
clip_length = args.clip_length
channel = args.channel

# 变量定义
# sample_rate默认为1
# stride默认为clip_length
# CROP_SIZE
if net =="c3d":
    CROP_SIZE = 112         #frame_size
elif net == "i3d": 
    CROP_SIZE = 224
else: 
    CROP_SIZE = 240
HOME = r"/mnt/151/sch/jhy/3dNet/"
dir =r"{}/frames/{}/{}/{}/train_frame/".format(HOME,dataset,structureType,evalType) #帧路径
mean_dir = "{}_{}_clipLength{}_cropSize{}_mean.npy".format(net,dataset,clip_length,CROP_SIZE) #均值文件

#均值计算并写入均值文件
def getMean():
    videos_mean = []
    # 设置进度条
    videoslist = os.listdir(dir) #视频文件名:4_7
    pbar = tqdm(sorted(videoslist)) # 存储的每个视频中的帧
    for video in pbar:
        pbar.set_description("Processing %s " % video)  # 4_7
        videopath = os.path.join(dir, video) #视频全路径:/jpl/input/global/train_frame/4_7
        # 将一个视频中的每个帧全路径添加进list中
        # 帧全路径/jpl/input/global/train_frame/4_7/000001.jpg
        frameslist = [os.path.join(videopath, frame) for frame in sorted(os.listdir(videopath))] 
        framesNum=len(frameslist)
        video_clips = []
        # 不重复帧的clip
        for idx in range(0,framesNum,clip_length): #包左不包右
            # 不足clip_length帧
            if idx + clip_length > framesNum:
                clip = frameslist[idx:]
                [clip.append(clip[-1]) for i in range(clip_length-len(clip))] #用最后一帧进行添补
            else:
                clip = frameslist[idx:idx+clip_length]
            processed_clip = clip_process(clip, CROP_SIZE, channel, crop_type="center")
            video_clips.append(processed_clip)
        
        # 一个视频中总clip
        video_clips = np.array(video_clips)  # (clipNum帧数,clip_length=16,CROP_SIZE,CROP_SIZE,channel=3) 
        print ("video_clips.shape:", video_clips.shape)
        # 一个视频的均值
        video_mean = np.sum(video_clips,0) / len(video_clips) #除以视频的所有clip数
        print ("video_mean.shape:", video_mean.shape)
        # videos_mean:将每个视频均值添加到列表
        videos_mean.append(video_mean)
    videos_mean = np.array(videos_mean) # (num=70,clip_length,CROP_SIZE,CROP_SIZE,3)
    print ("len(videos_mean):", len(videos_mean)) 
    # 最终均值
    mean = np.sum(videos_mean,0) / len(videos_mean) #除以视频个数
    np.save(mean_dir,mean)
    print ("result:",mean.shape)
    print ("----------------------- All Done ---------------------")

if __name__ == '__main__':
    getMean()
