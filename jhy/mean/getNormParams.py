#encoding=utf-8
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import pandas as pd
import sys
sys.path.append('../')
from common import *
from dataProcess import scale_frame,random_crop 

'''
获取标准化参数
两种常用的归一化方法:
1.min-max标准化方法
2.Z-score标准化方法
'''

# 获取参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='jpl', help='[dog/jpl]')
parser.add_argument('--cropSize', type=int, default=112, help='[crop size]')
parser.add_argument('--structureType', type=str, default='local', help='[global/local]')
parser.add_argument('--evalType', type=str, default='flow', help='[rgb/flow]')
args = parser.parse_args()
dataset = args.dataset
cropSize = args.cropSize
structureType = args.structureType
evalType = args.evalType

# 获取光流类型
def getFlowType():
    flowLists = sorted(['dense','warp'])
    for idx in range(len(flowLists)):
        print("{}. {}".format(idx,flowLists[idx]))
    print('********************************************')
    try:
        while True:
            idx=eval(raw_input("Select flow type:"))
            if idx in range(len(flowLists)):
                break
    except:
        print ("[Info]Error: Invalid Input!")
        exit()
    else: 
        print("[Info]Using {} flow!".format(flowLists[idx]))
    return flowLists[idx]

# 计算BGR
def getBGR(videosList):
    B, G, R = [], [], []    # 存储列表中所有帧通道值
    for video in tqdm(videosList):
        frameslist = sorted([os.path.join(video, frm) for frm in os.listdir(video)])
        for framePath in frameslist:
            if evalType == 'rgb':   # 三通道
                frame = Image.open(framePath)          # Image.open通道顺序012对应RGB,cv2.imread通道顺序012对应BGR
                # 1. 缩放
                frame = scale_frame(frame, cropSize)   # 执行缩放
                # 2.裁剪
                frame = random_crop(frame, cropSize)
                # 3.改变通道顺序为BGR并存储
                (b, g, r) = cv2.split(np.array(frame)[:,:, (2,1,0)])    
            elif evalType == 'flow':  # 两通道
                flow_x_frame = Image.open(framePath)
                flow_y_frame = Image.open(framePath.replace('flow_x', 'flow_y'))
                # 1.缩放
                flow_x_frame = scale_frame(flow_x_frame, cropSize)
                flow_y_frame = scale_frame(flow_y_frame, cropSize)
                # 2.裁剪
                flow_x_frame = random_crop(flow_x_frame, cropSize)
                flow_y_frame = random_crop(flow_y_frame, cropSize)
                # 3.改变通道顺序: 只有两个通道将r通道设置为np.nan
                (b, g, r) = (flow_x_frame, flow_y_frame, np.nan)    
            B.append(b)
            G.append(g)
            R.append(r)
    return B,G,R

def getChannelMin(B, G, R):
    return np.min(B), np.min(G), np.min(R)

def getChannelMax(B, G, R):
    return np.max(B), np.max(G), np.max(R)

def getChannelMean(B, G, R):
    return np.mean(B), np.mean(G), np.mean(R)

def getChannelStd(B, G, R):
    return np.std(B, ddof=1), np.std(G, ddof=1), np.std(R, ddof=1)

def getPixelMean(B, G, R):
    return np.mean(B, axis=0), np.mean(G, axis=0), np.mean(R, axis=0)

def getPixelStd(B, G, R):
    return np.std(B, axis=0, ddof=1), np.std(G, axis=0, ddof=1), np.std(R, axis=0, ddof=1)

if __name__ == '__main__':
    # 目录
    HOME = os.path.dirname(HOME)
    if evalType == 'rgb':
        framesDir = '{}/frames/{}/{}/{}/train_frame/' \
                         .format(HOME,dataset,structureType,evalType)
        videosList = sorted([os.path.join(framesDir, video) for video in os.listdir(framesDir)])
    elif evalType == 'flow':
        flowType = getFlowType()
        framesDir = '{}/frames/{}/{}/{}/{}/train_frame/flow_x/' \
                         .format(HOME,dataset, structureType,evalType,flowType)
        videosList = sorted([os.path.join(framesDir, video) for video in os.listdir(framesDir)])
    
    B,G,R = getBGR(videosList)
    B, G, R = np.array(B), np.array(G), np.array(R)
    
    ###############1.npy均值文件存储的均值格式###############
    # 获取均值(按照像素) B_mean的shape为帧的尺寸
    B_mean, G_mean, R_mean = getPixelMean(B, G, R)
    # 获取标准差(按照像素)
    B_std, G_std, R_std = getPixelStd(B, G, R)

    ######################2.单数值均值#######################
    # 获取均值(按照通道) B_mean为一个数值
    B_mean, G_mean, R_mean = getChannelMean(B, G, R)
    # 获取标准差(按照通道)
    B_std, G_std, R_std = getChannelStd(B, G, R)
    # 获取最大值(按照通道)
    B_max, G_max, R_max = getChannelMax(B, G, R)
    # 获取最小值(按照通道)
    B_min, G_min, R_min = getChannelMin(B, G, R)
    print('Mean(BGR):', B_mean, G_mean, R_mean)
    print('Std(BGR):', B_std, G_std, R_std)
    print('Max(BGR):', B_max, G_max, R_max)
    print('Min(BGR):', B_min, G_min, R_min)
    
    '''
    将0均值标准差写入文件
    format:
           b  g  r
      mean
      std
    '''
    table = pd.DataFrame(columns=['b','g','r'],
                         index=['mean','std'])
    table['b']['mean'] = B_mean
    table['g']['mean'] = G_mean
    table['r']['mean'] = R_mean
    table['b']['std'] = B_std
    table['g']['std'] = G_std
    table['r']['std'] = R_std
    # jpl_global_flow.csv
    meanFile = "{}_{}_{}_mean.csv".format(dataset,structureType,evalType)
    table.to_csv(meanFile,
                 index=False,
                 header=False,
                 sep='\t') 
    print ("[Info]write to {}!".format(meanFile))
