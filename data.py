"""
Datasetの作成
"""

import cv2
import numpy as np
import os
import glob
from itertools import product as product
from math import sqrt as sqrt
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

def make_datapath_list(img_path:str, img_extension:str, xml_path:str, division_ratio=0.8):
    """
    教師データへのパスを格納したリストを作成する

    Args:
        img_path (str): 画像データを格納したフォルダへのパス
        img_extension (str): 画像データの拡張子
        xml_path (str): アノテーションデータを格納したフォルダへのパス
        division_ratio (float, optional): train用とvalidation用のデータを分割する比率. Defaults to 0.8.

    Returns:
        (tuple(list)): 各データへのパスを格納したリストのタプル
    """

    # globでファイルを一括取得
    img_list = glob.glob(os.path.join(img_path, '*' + img_extension))   # -> list[path1, path2, path3, ...]
    xml_list = glob.glob(os.path.join(xml_path, '*.xml'))               # -> list[path1, path2, path3, ...]

    # 教師データの数 = division_ratio × 教師データの総数
    train_num = int(division_ratio * len(img_list))     # -> int

    # train用データを [0:train_num] で取得
    train_img_list = img_list[:train_num]   # -> list[path1, path2, path3, ...]
    train_xml_list = xml_list[:train_num]   # -> list[path1, path2, path3, ...]

    # validation用データを [train_num:] で取得
    val_img_list = img_list[train_num:]     # -> list[path1, path2, path3, ...]
    val_xml_list = xml_list[train_num:]     # -> list[path1, path2, path3, ...]

    # 各データへのパスを格納したリストを返す
    return train_img_list, train_xml_list, val_img_list, val_xml_list

class Anno_xml2list(object):
    """
    1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

    Args:
        classes (list): クラス名を格納したリスト(20要素)
    """
    def __init__(self, classes:list):
        self.classes = classes
    
    def __call__(self, xml_path:str, width:int, height:int) -> np.ndarray:
        """
        1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

        Args:
            xml_path (str): xmlファイルへのパス
            width (int): 画像の幅
            height (int): 画像の高さ

        Returns:
            np.array: 物体のアノテーションデータを格納したリスト
                Example: [[xmin, ymin, xmax, ymax, label_ind], ... ]
        """
        # 1枚の画像内に含まれるアノテーションを格納するリスト
        annotations = []

        # xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体(object)の数だけ繰り返し
        for obj in xml.iter('object'):
            # 1つの物体を囲むbounding box情報を格納するリスト
            bndbox = []

            # クラス名
            name = obj.find('name').text.lower().strip()

            # bounding boxの情報
            bbox = obj.find('bndbox')

            # アノテーションの xmin, ymin, xmax, ymaxを取得し、0～1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # 4隅の座標を取得
                cur_pixel = int(bbox.find(pt).text)

                # 幅、高さで規格化
                if pt == 'xmin' or pt == 'xmax':  # x方向のときは幅で割算
                    cur_pixel /= width
                else:  # y方向のときは高さで割算
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # アノテーションのクラス名のindexを取得して追加
            class_idx = self.classes.index(name)
            bndbox.append(class_idx)

            # resに[xmin, ymin, xmax, ymax, class_idx]を足す
            annotations += [bndbox]

        return np.array(annotations)  # -> [[xmin, ymin, xmax, ymax, class_idx], ... ]

class DataTransform():
    """
    画像とアノテーションの前処理クラス。
    学習時はデータオーギュメンテーションする。
    """
    def __init__(self, input_size):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),      # intをfloat32に変換
                ToAbsoluteCoords(),     # アノテーションデータの正規化を元に戻す
                PhotometricDistort(),   # 画像の色調等をランダムに変更
                RandomMirror(),         # 画像をランダムに反転させる
                ToPercentCoords(),       # アノテーションデータを0-1に規格化
                Resize(input_size)
            ]),
            'val': Compose([
                ConvertFromInts(),      # intをfloat32に変換
                Resize(input_size)
            ])
        }
    
    def __call__(self, img:np.ndarray, phase:str, boxes, labels) ->object:
        """
        画像とアノテーションの前処理クラス。
        学習時はデータオーギュメンテーションする。

        Args:
            img (np.ndarray): 変換する画像
            phase (str): 'train' or 'val'
            boxes (list(float)): bounding box の情報を格納したリスト
            labels (list(float)): 物体のクラス名

        Returns:
            img, boxes, labels: 変換後の画像とアノテーションデータ
        """
        return self.data_transform[phase](img, boxes, labels)

class OD_Dataset(data.Dataset):
    """
    Datasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Args:
        img_list (list): 画像のパスを格納したリスト
        anno_list (list): アノテーションへのパスを格納したリスト
        phase ('train' or 'val' : str): 学習か訓練かを設定する
        transform (object): 前処理クラスのインスタンス
        transform_anno (object): xmlのアノテーションをリストに変換するインスタンス
    """
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  # 'train' or 'val'を指定
        self.transform = transform  # 画像の変形
        self.transform_anno = transform_anno  # アノテーションデータをxmlからリストへ
    
    def __len__(self) -> int:
        """
        画像の枚数を返す
        """
        return len(self.img_list)

    def __getitem__(self, index):
        """
        前処理をした画像のテンソル形式のデータとアノテーションを取得
        """
        im, gt, _, _ = self.pull_item(index)    # -> 画像、真値、高さ、幅
        return im, gt

    def pull_item(self, index):
        """
        前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、幅を取得する

        Args:
            index (int): データのインデックス

        Returns:
            img, gt, height, width: 前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、幅
        """

        # 画像読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, _ = img.shape  # 画像のサイズを取得

        # xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 前処理を実施
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])

        # 色チャネルの順番がBGRになっているので、RGBに順番変更
        # さらに（高さ、幅、色チャネル）の順を（色チャネル、高さ、幅）に変換
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # BBoxとラベルをセットにしたnp.arrayを作成
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width

def od_collate_fn(batch):
    """
    Datasetからミニバッチを取り出す関数

    Args:
        batch (list): ミニバッチ

    Returns:
        tuple: ミニバッチに対応する画像と正解ラベル
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets

# 動作確認
if __name__ == '__main__':

    # 1. 教師データをリスト形式で読み込み
    img_path = './data/VOCdevkit/VOC2012/JPEGImages'
    img_extension = '.jpg'
    xml_path = './data/VOCdevkit/VOC2012/Annotations'

    train_img_files, train_xml_files, val_img_files, val_xml_files = make_datapath_list(
        img_path, img_extension, xml_path
    )

    print("train_img_file_num =", len(train_img_files))
    print("train_xml_file_num =", len(train_xml_files))
    print("val_img_files_num =", len(val_img_files))
    print("val_xml_files_num =", len(val_xml_files))
    print("val_img_files[4] =", val_img_files[4])
    print("val_xml_files[4] =", val_xml_files[4])

    voc_classes = ['aeroplane','bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

    # 2. datasetを作成
    input_size = 300
    transform = DataTransform(input_size=input_size)
    transform_anno = Anno_xml2list(voc_classes)

    train_dataset = OD_Dataset(
        train_img_files, train_xml_files, phase='train',
        transform=transform, transform_anno=transform_anno
        )

    val_dataset = OD_Dataset(
        val_img_files, val_xml_files, phase='val',
        transform=transform, transform_anno=transform_anno
    )

    # 3. DataLoaderを作成
    batch_size = 4

    train_dataloader = data.DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=od_collate_fn
    )

    val_dataloader = data.DataLoader(
        val_dataset, batch_size, shuffle=False, collate_fn=od_collate_fn
    )

    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    batch_iterator = iter(dataloaders_dict['val'])
    images, targets = next(batch_iterator)
    print('len(targets) =', len(targets))
    print('targets[1] =', targets)

