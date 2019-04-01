# encoding=utf8
from __future__ import division
import os
import ast
import math
import argparse
import pandas as pd

'''表生成：
    样本长：可选
    采样率：根据长短视频计算调整
    步幅：固定为1，目的是为了增加样本数
    isFixSampleRate：目的是在原有测试集列表生成的基础上均衡测试集的每个类别样本个数
目的：尽量使得每个样本包含完整动作，因此对于长短视频需要设置不同的采样率。
'''

# 获取参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='jpl', help='[dog/jpl]')
parser.add_argument('--structureType', type=str, default='local', help='[global/local]')
parser.add_argument('--evalType', type=str, default='flow', help='[rgb/flow]')
parser.add_argument('--listType', type=str, default='test', help='[train/test]')
parser.add_argument('--clipLength', type=int, default=16, help='clipLength')
parser.add_argument('--isFixSampleRate', type=ast.literal_eval, default=False, help='whether use fixed sampleRate')
parser.add_argument('--isFusion', type=ast.literal_eval, default=True, help='whether use fusion')
args = parser.parse_args()
dataset = args.dataset
structureType = args.structureType
evalType = args.evalType
listType = args.listType

clipLength = args.clipLength
isFixSampleRate = args.isFixSampleRate
isFusion = args.isFusion
FixSampleRate = 8
STRIDE = 1

HOME = "/mnt/151/sch/jhy/3dNet"


# 获取光流类型
def getFlowType():
    flowLists = sorted(['dense', 'warp'])
    print('********************************************')
    for idx, item in enumerate(flowLists):
        print("{}. {}".format(idx, item))
    print('********************************************')
    try:
        while True:
            idx = eval(raw_input("Select flow type:"))  # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
            if idx in range(len(flowLists)):
                break
    except:
        print("[Error]Invalid Input!")
        exit()
    else:
        print("[Info]Using {} flow!".format(flowLists[idx]))
    return flowLists[idx]


# 获取类别标签
def get_label(video):
    if dataset == "dog" or dataset == 'jpl':
        return str(int(video.split('_')[1]) - 1)
    else:
        print('[Info]Invalid dataset!')


# 获取下一帧id
def get_nextStartFrm(curr_startFrm, sampleRate, framesNum):
    next_startFrm = None
    next_startFrm_id = int(curr_startFrm.split('.')[0]) + STRIDE
    if next_startFrm_id + (clipLength - 1) * sampleRate <= framesNum:
        next_startFrm = '{:06d}.jpg'.format(next_startFrm_id)
    return next_startFrm


# 将输入信息写入文件
def write_to_file(framesDir, list_dir, list_name):
    print("[Info]Generate {}!".format(list_name))
    table = pd.DataFrame(columns=('video', 'label', 'framesNum', 'clipLength', 'stride', 'sampleRate'))
    with open(os.path.join(list_dir, list_name), 'w') as f:
        for video in sorted(os.listdir(framesDir)):
            frameslist = sorted([frm for frm in os.listdir(os.path.join(framesDir, video))])
            framesNum = len(frameslist)
            if isFusion and evalType == 'rgb':
                framesNum -= 1
            label = get_label(video)   # 获得视频类别标签
            # 1.是否采用固定采样率
            if isFixSampleRate:
                sampleRate = FixSampleRate
            # 根据长短视频来改变采样率
            else:
                sampleRate = int(math.floor(framesNum / clipLength / 2))
                if sampleRate == 0: sampleRate = 1
                '''
                if sampleRate in [0,1] : 
                    sampleRate=1
                elif sampleRate in range(2,20): 
                    sampleRate = int(math.floor(sampleRate/2))
                else : 
                    sampleRate = int(math.ceil(sampleRate/1.5)) 
                '''
            # 2.写入起始帧
            curr_startFrm = frameslist[0]
            f.write(os.path.join(framesDir, video, curr_startFrm) + '\t' + label + '\t' + str(sampleRate) + '\n')
            next_startFrm = get_nextStartFrm(curr_startFrm,
                                             sampleRate,
                                             framesNum)
            while not next_startFrm is None:
                f.write(os.path.join(framesDir, video, next_startFrm) \
                        + '\t' + label + '\t' + str(sampleRate) + '\n')
                curr_startFrm = next_startFrm
                next_startFrm = get_nextStartFrm(curr_startFrm,
                                                 sampleRate,
                                                 framesNum)
            # 3.写入表格
            table = table.append(pd.DataFrame({'video': [video],
                                               'label': [label],
                                               'framesNum': [framesNum],
                                               'clipLength': [clipLength],
                                               'stride': STRIDE,
                                               'sampleRate': sampleRate}),
                                 ignore_index=True,
                                 sort=False)
          # print(table)
    return table


# 均衡测试集每个类别样本个数
def balance_testList(listTable, FixclipsNum):
    # 1.定义均衡后的表格 
    baltestTable = pd.DataFrame(columns=('startFrmPath', 'label', 'sampleRate'))
    # 2.按类别分组
    label_indices_groups = listTable.groupby('label')
    for label, group in label_indices_groups:
        clipsNum = len(group)  # 每个类别对应的clips数
        # 3.在每个类别中对视频分组
        video_indices_groups = group.groupby('video')
        videosNum = len(video_indices_groups.groups)  # 字典
        # 4. 分类讨论
        # 长视频
        if clipsNum > FixclipsNum:
            print("test长视频:", video_indices_groups.groups.keys(), \
                  "视频数目", videosNum, \
                  "clips数目", clipsNum)
            for video, group1 in video_indices_groups:
                video_clipsNum = len(group1)
                num = int(round(FixclipsNum * (video_clipsNum / clipsNum)))  # 加权，取消FixclipsNum/videosNum
                baltestTable = baltestTable.append(group1.sample(num, replace=True)  # 随机采样
                                                   [['startFrmPath', 'label', 'sampleRate']])
                # 分组后取前起始帧
                # baltestTable = baltestTable.append(group1[:num] #行切片
                #                                          [['startFrmPath','label','sampleRate']])  #列取值
        # 短视频
        elif clipsNum < FixclipsNum:
            print("test短视频:", video_indices_groups.groups.keys(), \
                  "视频数目", videosNum, \
                  "clips数目", clipsNum)
            for video, group1 in video_indices_groups:
                # baltestTable = pd.concat([baltestTable, group.sample(FixclipsNum,replace=True)])
                video_clipsNum = len(group1)
                num = int(round(FixclipsNum * (video_clipsNum / clipsNum)))  # 加权，取消FixclipsNum/videosNum
                baltestTable = baltestTable.append(group1.sample(num, replace=True)  # 随机采样
                                                   [['startFrmPath', 'label', 'sampleRate']])
        # 视频样本刚好
        else:
            baltestTable = baltestTable.append(group[['startFrmPath', 'label', 'sampleRate']])
    return baltestTable


if __name__ == "__main__":
    # 定义类别数
    print("[Info]Using {} Dataset!!!!!!!!!!!!".format(dataset))
    if dataset == 'jpl':
        num_classes = 7
    elif dataset == 'dog':
        num_classes = 10
    else:
        exit

    # 目录
    print("[Info]Using {} type!".format(evalType))
    if evalType == 'rgb':
        framesDir = os.path.join('{}/frames/{}/{}/{}/' \
                                 .format(HOME, dataset, structureType, evalType),
                                 "{}_frame".format(listType))
    elif evalType == 'flow':
        flowType = getFlowType()
        framesDir = os.path.join('{}/frames/{}/{}/{}/' \
                                 .format(HOME, dataset, structureType, evalType),
                                 flowType,
                                 "{}_frame".format(listType),
                                 'flow_x')
    if isFusion:
        list_dir = '../listfile/{}/fusion/{}/{}/'.format(dataset, structureType, evalType)
    else:
        list_dir = '../listfile/{}/{}/{}/'.format(dataset, structureType, evalType)
    if not os.path.exists(list_dir): os.makedirs(list_dir)

    # 控制列表名中的采样率
    if isFixSampleRate:  # 采用固定采样率
        sampleRate = FixSampleRate
    else:  # 非固定采样率
        sampleRate = '#'

    # 1.定义列表文件名
    list_name = '{}_clipLength{}_sampleRate{}_stride{}' \
        .format(listType, clipLength, sampleRate, STRIDE)
    if evalType == 'flow':
        list_name += ('_' + flowType)
    list_name += ('.txt')

    # 2.写入列表文件
    write_to_file(framesDir, list_dir, list_name)

    # 3.读取成表格
    listTable = pd.read_csv(os.path.join(list_dir, list_name),
                            names=('startFrmPath', 'label', 'sampleRate'),
                            sep='\t')
    # 插入列(测试集均衡要对视频名分组)
    listTable['video'] = listTable['startFrmPath'].map(lambda x: x.split('/')[-2])
    print("Sum:", len(listTable))
    print("原列表:\n", listTable['label'].value_counts().sort_index())

    # 3. 固定测试集样本个数
    # 根据原列表获取测试集均衡后的每个类别固定样本个数
    FixclipsNum = int(len(listTable) / num_classes)
    if listType == 'test':
        baltestTable = balance_testList(listTable, FixclipsNum)
        baltestlist_name = list_name.split('.')[0] + '_bal.txt'
        baltestTable.to_csv(os.path.join(list_dir, baltestlist_name),
                            index=False,
                            header=False,
                            sep='\t')
        print("[Info]Generate balanceed listfile: {}!".format(baltestlist_name))
        print("Sum:", len(baltestTable))
        print("新test列表:\n", baltestTable['label'].value_counts().sort_index())
    else:
        pass
