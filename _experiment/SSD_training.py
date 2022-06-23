# パッケージのimport
from logging import root
import os.path as osp
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data

from utils.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn
from utils.ssd_model import SSD, MultiBoxLoss

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ファイルパスのリストを作成
rootpath = './2_objectdetection/data/VOCdevkit/VOC2012/'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

# Datasetを作成する
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

color_mean = (104, 117, 123)
input_size = 300

train_dataset = VOCDataset(
    train_img_list, train_anno_list, phase='train',
    transform=DataTransform(
        input_size, color_mean
    ),
    transform_anno=Anno_xml2list(voc_classes)
)
val_dataset = VOCDataset(
    val_img_list, val_anno_list, phase='val',
    transform=DataTransform(
        input_size, color_mean
    ),
    transform_anno=Anno_xml2list(voc_classes)
)

# Dataloaderを作成する　
batch_size = 32

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

# 辞書型オブジェクトにまとめる
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

# SSD300の設定
ssd_cfg = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
}

# SSDネットワークモデル
net = SSD(phase='train', cfg=ssd_cfg)

# SSDの初期の重みを設定
# ssdのvgg部分にロードする
vgg_weights = torch.load('./2_objectdetection/weights/vgg16_reducedfc.pth')
net.vgg.load_state_dict(vgg_weights)

# SSDのその他のネットワークの重みはHeの初期値で初期化
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Heの初期値を適用
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# 損失関数の設定
criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

# 最適化手法の設定
optimizer = optim.SGD(net.parameters(), lr = 1e-3, momentum=0.9, weight_decay=5e-4)

# モデルを学習させる関数
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # ネットワークをデバイスへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # イテレーションカウンタをセット
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    # epochのループ
    for epoch in range(num_epochs+1):
        # 開始時刻の保存
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('--------------------------------------')
        print('Epoch{}/{}'.format(epoch+1, num_epochs))
        print('--------------------------------------')

        # epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print('train')
            else:
                if ((epoch+1) % 10 == 0):
                    net.eval()
                    print('--------------------------------------')
                    print('validation')
                else:
                    continue

            # データローダーからminibatchずつ取り出すループ
            for images, targets in dataloaders_dict[phase]:

                # GPUが使えるならGPUにデータを送る
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝播計算
                with torch.set_grad_enabled(phase == 'train'):

                    # 順伝播計算
                    outputs = net(images)

                    # 損失の計算
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 訓練時は誤差逆伝播
                    if phase == 'train':
                        loss.backward()

                        nn.utils.clip_grad_value_(
                            net.parameters(), clip_value=2.0
                        )

                        optimizer.step()

                        if (iteration % 10 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('iteration: {} || loss: {:.4f} || 10iter: {:.4f} sec.'.format(iteration, loss.item(), duration))

                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss =+ loss.item()
        
            # epochのphaseごとのlossと正解率
            t_epoch_finish = time.time()
            print('--------------------------------------')
            print('epoch: {} || loss: {:.4f} || val_loss: {:.4f}'.format(epoch+1, epoch_train_loss, epoch_val_loss))
            print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
            t_epoch_start = time.time()

            # ログを保存
            log_epoch = {
                'epoch': epoch+1,
                'loss': epoch_train_loss,
                'val_loss': epoch_val_loss
            }
            logs.append(log_epoch)
            df = pd.DataFrame(logs)

            df.to_csv('./2_objectdetection/log/log_output.csv')

            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            # ネットワークを保存する
            if ((epoch+1) % 10 == 0):
                torch.save(net.state_dict(), './2_objectdetection/weights/ssd300_' + str(epoch+1) + '.pth')

# 学習と検証を実行する
num_epochs = 50
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)






