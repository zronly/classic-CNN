# coding=utf-8
import os
import sys
import cv2
import time
import math
import h5py
import random
import numpy as np

np.random.seed(42)
import pandas as pd
from tqdm import tqdm
from keras import utils
import PIL.Image as Image
import PIL.ImageOps as ImageOps

'''
功能：数据预处理，包含网络输入处理层以及特征提取：写入特征文件
'''


# ---------------------  format input data-------------------------
# 按scale划分训练集和验证集，返回训练集和验证集对应DataFrame中的行号
def splitGroups(dataTable, valScale):
    label_indices_groups = dataTable.groupby('label')
    trainTable = pd.DataFrame(columns=dataTable.columns.values)
    valTable = pd.DataFrame(columns=dataTable.columns.values)
    # 划分训练集和验证集
    for label, group in label_indices_groups:
        clipsNum = len(group)
        group = group.sample(frac=1).reset_index()  # 打乱数据重新改变索引
        # 限制valScale范围
        if valScale == 1:
            trainTable = trainTable.append(group[:], sort=False)
            valTable = valTable.append(group[:], sort=False)
        elif valScale > 0:
            trainTable = trainTable.append(group[: int(clipsNum * (1 - valScale))], sort=False)
            valTable = valTable.append(group[int(clipsNum * (1 - valScale)):], sort=False)
        else:
            exit("[Info]Error: valScale can't greater than 0!")
    return trainTable, valTable


# 生成batchsize个clip,返回batchsize个clip像素数组和label
def generate_batch(label_indices_groups, mode, batchSize=0, clipLength=16,
                   cropSize=224, channel=3, cropType='center'):
    data = []
    labels = []
    # train:训练集按照batchSize取数据需计算从每个类别中抽取的样本个数;验证测试集则将整体作为一个batch
    label_nums_dict = {}
    if mode == 'train':
        label_nums_dict = get_label_nums_dict(label_indices_groups, batchSize)
    for label, group in label_indices_groups:
        # 训练集
        if mode == 'train':
            clipsNum = label_nums_dict[label]  # 一个类别抽取的样本数
            indicesTable = group.sample(clipsNum, replace=True)  # 对应到dict中抽取数据
        # 验证集和测试集
        else:
            clipsNum = len(group)  # 其它模式下取全部数据
            indicesTable = group
        for idx in range(clipsNum):
            clip_data = get_clipData(indicesTable.iloc[idx]['startFrmPath'],
                                     clipLength,
                                     indicesTable.iloc[idx]['sampleRate'],
                                     cropSize,
                                     channel,
                                     cropType,
                                     mode)  # 获取数据
            clip_label = indicesTable.iloc[idx]['label']  # 获取标签
            data.append(clip_data)
            labels.append(clip_label)
    data = np.array(data).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    # 将标签转化为one-hot形式
    labels = utils.to_categorical(labels,
                                  num_classes=len(label_indices_groups.groups))
    return data, labels


# 针对训练集：计算batch中每个类别对应的样本个数(考虑避免不能整除情况)
def get_label_nums_dict(label_indices_groups, batchSize):
    num_classes = len(label_indices_groups.groups)
    avgNum = int(batchSize / num_classes)  # 每个类别平均抽取样本个数
    deficit = batchSize - avgNum * num_classes  # 剩余还需抽取的样本个数
    # 1.赋初值
    label_nums_dict = {}  # '0':2 类别: 对应的样本个数
    for label, group in label_indices_groups:
        label_nums_dict[label] = avgNum  # 1个batch中各类别需要抽取的样本数
    # 2.补缺额
    if avgNum == 0:
        for i in range(deficit):
            label_nums_dict[np.random.choice([label for label, group in label_indices_groups])] += 1
    else:
        if deficit:  # 如果不能整除(存在缺额)，则随机选取一个类别抽取剩余的样本
            # np.random.choice用于从列表中随机抽选一个数据
            label_nums_dict[np.random.choice([label for label, group in label_indices_groups])] += deficit
    return label_nums_dict


# 根据起始帧获取对应的clip;若起始帧为/10_1/000001.jpg,sampleRate=2,clipLength-16;clip[1.jpg,3.jpg,..,31.jpg]
def get_clipData(start_frmPath, clipLength, sampleRate, cropSize, channel, cropType, mode):
    current_frmPath = start_frmPath  # 设置起始帧为当前帧
    clip = []
    for idx in range(clipLength):
        clip.append(current_frmPath)  # 将当前帧存入clip
        current_frmPath = get_nextFrame(current_frmPath, sampleRate)  # 更新当前帧
        # 处理帧不存在的情况，如果更新后的当前帧是不存在的，则重置当前帧为起始帧
        # 例如视频只包含30帧，则clip为[000001.jpg, 000003.jpg, ..., 000029.jpg, 000029.jpg, 000029.jpg]
        try:
            Image.open(current_frmPath)  # 试图打开不存在的帧会抛出异常
        except:
            '''拿最后帧补足
            current_idx = current_frmPath.split('/')[-1].split('.')[0]
            end_idx = int(current_idx) - sampleRate
            current_frmPath = start_frmPath.replace(current_idx,'{:06d}'.format(end_idx)) # 重置当前帧
            '''
            current_frmPath = start_frmPath
    clip_data = clip_process(clip, cropSize, channel, cropType, mode)  # 处理clip，得到数据
    return clip_data


# 根据sampleRate获取下一帧
def get_nextFrame(framePath, sampleRate):
    # framePath: /10_1/000001.jpg
    frameDir = os.path.dirname(framePath)  # /10_1
    frameName = os.path.basename(framePath)  # 000001.jpg
    next_frameId = int(frameName.split('.')[0]) + sampleRate
    next_frame = '{}/{:06d}.jpg'.format(frameDir, int(next_frameId))
    return next_frame


# ----------------------------- clip process ------------------------
def clip_process(clip, cropSize=120, channel=3, cropType="center", mode=''):  # clip为路径
    clipLength = len(clip)
    processedClip = np.zeros([clipLength, cropSize, cropSize, channel]).astype(np.float32)
    mirror = do_mirror()
    # 获取均值表格
    mean = get_mean_df(clip[0])
    # rgb
    if channel == 3:
        for idx in range(clipLength):
            frame = Image.open(clip[idx])
            # 1.翻转:只针对训练时的训练集且mirror=1；特征提取不需要
            if (mode == 'train') and (mirror == 1):
                frame = ImageOps.mirror(frame)
            # 2.缩放 
            frame = scale_frame(frame, cropSize)
            # 3.裁剪
            if cropType == "random":
                frame = random_crop(frame, cropSize)
            else:
                frame = center_crop(frame, cropSize)
            # 4.将处理过的每个帧添加进processedClip
            processedClip[idx, :, :, :] = frame
            # 5.减均值
            processedClip[idx, :, :, 0] -= mean['r']['mean']  # R 117.5
            processedClip[idx, :, :, 1] -= mean['g']['mean']  # G 116.5
            processedClip[idx, :, :, 2] -= mean['b']['mean']  # B 94.5
            processedClip[idx, :, :, 0] /= mean['r']['std']
            processedClip[idx, :, :, 1] /= mean['g']['std']
            processedClip[idx, :, :, 2] /= mean['b']['std']
    # flow
    elif channel == 2:
        for idx in range(clipLength):
            flow_x_frame = Image.open(clip[idx])
            flow_y_frame = Image.open(clip[idx].replace('flow_x', 'flow_y'))
            # 1.翻转:只针对训练时的训练集且mirror=1；特征提取不需要
            if (mode == 'train') and (mirror == 1):
                flow_x_frame = ImageOps.mirror(flow_x_frame)
                flow_y_frame = ImageOps.mirror(flow_y_frame)
            # 2.缩放 
            flow_x_frame = scale_frame(flow_x_frame, cropSize)  # b
            flow_y_frame = scale_frame(flow_y_frame, cropSize)  # g
            # 3.裁剪
            if cropType == "random":
                flow_x_frame = random_crop(flow_x_frame, cropSize)
                flow_y_frame = random_crop(flow_y_frame, cropSize)
            else:
                flow_x_frame = center_crop(flow_x_frame, cropSize)
                flow_y_frame = center_crop(flow_y_frame, cropSize)
            # 4.将处理过的每个帧添加进processedClip
            processedClip[idx, :, :, 0] = flow_x_frame
            processedClip[idx, :, :, 1] = flow_y_frame
            # 5.减均值
            processedClip[idx, :, :, 0] -= mean['b']['mean']  # b  
            processedClip[idx, :, :, 1] -= mean['g']['mean']  # g
            processedClip[idx, :, :, 0] /= mean['b']['std']
            processedClip[idx, :, :, 1] /= mean['g']['std']
    return processedClip


# 对帧按照裁剪尺寸进行比例缩放
def scale_frame(frame, scaleSize):
    # 跟据短边确定计算比例
    if frame.width > frame.height:
        scale = float(scaleSize) / float(frame.height)
        frame = np.array(cv2.resize(np.array(frame),
                                    (int(frame.width * scale + 1), scaleSize))).astype(np.float32)
    else:
        scale = float(scaleSize) / float(frame.width)
        frame = np.array(cv2.resize(np.array(frame),
                                    (scaleSize, int(frame.height * scale + 1)))).astype(np.float32)
    return frame


# 随机裁剪
def random_crop(frame, cropSize):
    crop_x = random.randint(0, (frame.shape[0] - cropSize))  # 左右包含
    crop_y = random.randint(0, (frame.shape[1] - cropSize))
    if frame.ndim == 2:
        return frame[crop_x:crop_x + cropSize, crop_y:crop_y + cropSize]
    elif frame.ndim == 3:
        return frame[crop_x:crop_x + cropSize, crop_y:crop_y + cropSize, :]


# 中心裁剪
def center_crop(frame, cropSize):
    crop_x = int((frame.shape[0] - cropSize) / 2)
    crop_y = int((frame.shape[1] - cropSize) / 2)
    if frame.ndim == 2:
        return frame[crop_x:crop_x + cropSize, crop_y:crop_y + cropSize]
    elif frame.ndim == 3:
        return frame[crop_x:crop_x + cropSize, crop_y:crop_y + cropSize, :]


# 三通道均值
def get_mean_df(path):
    # 根据绝对路径来判断,避免截取字符串以防列表内路径变动
    if 'jpl' in path:
        dataset = 'jpl'
    elif 'dog' in path:
        dataset = 'dog'
    if 'global' in path:
        structureType = 'global'
    elif 'local' in path:
        structureType = 'local'
    if 'rgb' in path:
        evalType = 'rgb'
    elif 'flow' in path:
        evalType = 'flow'
    HOME = "/mnt/151/sch/jhy/3dNet"
    meanfile_name = '{}_{}_{}_mean.csv'.format(dataset,
                                               structureType,
                                               evalType)
    # print ("[Info]Using {}".format(meanfile_name))
    mean = pd.read_csv(os.path.join('{}/tools/mean'.format(HOME),
                                    meanfile_name),
                       names=['b', 'g', 'r'],
                       sep='\t')
    mean.index = ['mean', 'std']
    return mean


# npy均值文件
def get_mean_npy(net, dataset, clipLength, cropSize, channel):
    meanFile_name = "{}_{}_clipLength{}_cropSize{}_mean.npy".format(net, dataset, clipLength, cropSize)
    meanFile_path = os.path.join("/home/jhy/workspace/3dNet/tools/mean/", meanFile_name)
    # print meanFile
    mean = np.load(meanFile_path).reshape([clipLength, cropSize, cropSize, channel])  # 均值
    # shape(16,224,224,3)
    # for idx in mean:
    #   print idx
    return mean


# 数据翻转
def do_mirror():
    mirror = np.random.randint(2)
    return mirror


# ---------------------- extract features -------------------
# 特征提取:将特征写入文件
def write_features_into_file(featuresList, pair_video_clips, featurePath):
    print("Feature dimension:{}".format(featuresList.shape))
    with h5py.File(featurePath, 'w') as f:
        idx = 0
        # video_clips为列表值，第一个值存储视频名，第二个值存储提取的特征数
        for video_clips in tqdm(pair_video_clips):
            # 每个短视频名称 4_7
            videoname = video_clips[0]
            video_clips_Num = video_clips[1]
            f[videoname] = featuresList[idx:idx + video_clips_Num]
            # 更新idx以便于读取下一个视频的特征
            idx += video_clips_Num
    print("********************** Done ***********************")
