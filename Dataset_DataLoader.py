import os
import random
from typing_extensions import dataclass_transform
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

# 学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する
def make_datapath_list(rootpath):
    """
    データへのパスを格納したリストを作成する

    Args:
        rootpath (str): データフォルダへのパス

    Returns:
        train_img_list, train_anno_list, val_img_list, val_anno_list (list): データへのパスを格納したリスト
    """
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = os.path.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = os.path.join(rootpath, 'Annotations', '%s.xml')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = os.path.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = os.path.join(rootpath + 'ImageSets/Main/val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を削除
        img_path = (imgpath_template % file_id) # 画像のパス
        anno_path = (annopath_template % file_id)   # アノテーションのパス
        train_img_list.append(img_path) # リストに追加
        train_anno_list.append(anno_path)   # リストに追加

    # 検証データの画像ファイルとアノテーションファイルのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を削除
        img_path = (imgpath_template % file_id) # 画像のパス
        anno_path = (annopath_template % file_id)   # アノテーションのパス
        val_img_list.append(img_path)   # リストに追加
        val_anno_list.append(anno_path) # リストに追加

    return train_img_list, train_anno_list, val_img_list, val_anno_list

# xml形式のアノテーションデータをリスト型に変換するクラス
class Anno_xml2list(object):
    """
    1枚の画像に対する「xml形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する

    Args:
        object (classes): VOCのクラス名を格納したリスト
    """
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        # 画像内のすべての物体のアノテーションをこのリストに格納する
        ret = []

        # xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体（object）の数だけループする
        for obj in xml.iter('object'):

            # アノテーションけで検知がdifficulltに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []

            name = obj.find('name').text.lower().strip()    # 物体名
            bbox = obj.find('bndbox')   # バウンディングボックス

            # アノテーションのxmin, ymin, xmax, ymaxを取得し、0～1に正規化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOCは原点が(1, 1)なので引き算して(0, 0)に
                cur_pixel = int(bbox.find(pt).text) - 1

                # 幅、高さで気書くか
                if (pt == 'xmin') or (pt == 'xmax'):
                    cur_pixel /= width
                else:
                    cur_pixel /= height
                
                bndbox.append(cur_pixel)

            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # retに[xmin, ymin, xmax, ymax, label_idx]を追加
            ret += [bndbox]
        
        return np.array(ret)

# 動作確認
voc_classes = ['aeroplane','bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

rootpath = './2_objectdetection/data/VOCdevkit/VOC2012/'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

transform_anno = Anno_xml2list(voc_classes)

# 画像の読み込み
ind = 1
image_file_path = val_img_list[ind]
img = cv2.imread(image_file_path)
height, width, channels = img.shape

# アノテーションをリストで表示
list = transform_anno(val_anno_list[ind], width, height)
print('list = ', list)
print(list[:, :4], list[:, 4])

# 入力画像の前処理をするクラス
class DataTransform():
    """
        画像とアノテーションの前処理クラス。訓練と推論で異なる動作をする。
        画像のサイズを300x300にする。
        学習時はデータオーギュメンテーションをする。

        Args:
            input_size (int): リサイズ先の画像の大きさ
            color_mean (B, G, R): 各色チャネルの平均値
        """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  # intをfloatに変換
                ToAbsoluteCoords(), # アノテーションデータの規格化を戻す
                PhotometricDistort(),   # 画像の色調などをランダムに変化
                Expand(color_mean), # 画像のキャンバスを広げる
                RandomSampleCrop(), # 画像内の部分をランダムに抜き出す
                RandomMirror(), # 画像を反転させる
                ToPercentCoords(),  # アノテーションデータを0-1に気書くか
                Resize(input_size), # 画像サイズをinput_size×input_sizeに変形
                SubtractMeans(color_mean)   # BGRの色の平均値を引き算 
            ]),
            'val': Compose([
                ConvertFromInts(),  # intをfloatに変換
                Resize(input_size), # 画像サイズをinput_size×input_sizeに変形
                SubtractMeans(color_mean)   # BGRの色の平均値を引き算
            ]) 
        }
    
    def __call__(self, img, phase, boxes, labels):
        """

        Args:
            img (cv2.Mat): 前処理する画像
            phase ('train' or 'val'): 前処理のモード
            boxes ([xmin, ymin, xmax, ymax]): バウンディングボックス
            labels (str): 物体のクラス

        Returns:
            img (cv2.Mat): 前処理後の画像
        """
        return self.data_transform[phase](img, boxes, labels)

# 動作の確認
# 1. 画像の読み込み
image_file_path = train_img_list[0]
img = cv2.imread(image_file_path)   # [高さ], [幅], [色BGR]
height, width , channels = img.shape # 画像のサイズを取得

# 2. アノテーションをリストに
transform_anno = Anno_xml2list(voc_classes)
anno_list = transform_anno(train_anno_list[0], width, height)

# 3. 元画像の表示
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# 4. 前処理クラスの作成
color_mean = (104, 117, 123)
input_size = 300
transform = DataTransform(input_size, color_mean)

# 5. train画像の表示
phase = 'train'
img_transformed, boxes, labels = transform(
    img, phase, anno_list[:, :4], anno_list[:, 4]
)
# plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
# plt.show()


# VOC201のデータセットを作成する
class VOCDataset(data.Dataset):

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  # 'train' or 'val'
        self.transform = transform  # 画像の変形
        self.transform_anno = transform_anno   # アノテーションデータをxmlからリストへ

    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.img_list)
    
    def __getitem__(self, index):
        """
        前処理をした画像のテンソル形式のデータとアノテーションデータを取得
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt
    
    def pull_item(self, index):
        """
        前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、画像の幅を取得する
        """

        # 1. 画像の読み込み
        image_file_path = train_img_list[0]
        img = cv2.imread(image_file_path)   # [高さ], [幅], [色BGR]
        height, width , channels = img.shape # 画像のサイズを取得

        # 2. xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 3. 前処理を実施
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4]
        )

        # 4. [高さ], [幅], [色BGR]を[色RGB], [高さ], [幅]に変更
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # 5. BBoxとラベルをセットにしたnp.arrayを作成
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width

# 動作確認
color_mean = (104, 117, 123)
input_size = 300

train_dataset = VOCDataset(
    train_img_list, train_anno_list, phase='train',
    transform=DataTransform(input_size, color_mean),
    transform_anno=Anno_xml2list(voc_classes)
    )

val_dataset = VOCDataset(
    val_img_list, val_anno_list, phase='val',
    transform=DataTransform(input_size, color_mean),
    transform_anno=Anno_xml2list(voc_classes)
)

# データの取り出し例
# print(val_dataset.__getitem__(1))


def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets

# データローダの作成
batch_size = 4
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size,
    shuffle=True, collate_fn=od_collate_fn
)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size,
    shuffle=False, collate_fn=od_collate_fn
)

# 辞書型変数にまとめる
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

# 動作の確認
batch_iterator = iter(dataloaders_dict['val'])  # イテレータに変換
images, targets = next(batch_iterator)  # 1番目の要素を取り出す
# print(images.size())
# print(len(targets))
# print(targets[1].size())

# print(train_dataset.__len__())
# print(val_dataset.__len__())


