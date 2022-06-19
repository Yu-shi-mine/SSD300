import cv2  # OpenCVライブラリ
import matplotlib.pyplot as plt 
import numpy as np
import torch


from utils.ssd_model import SSD, DataTransform
from utils.ssd_predict_show import SSDPredictShow

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

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
net = SSD(phase='inference', cfg=ssd_cfg)

# SSDの学習済みの重みを設定
net_weights = torch.load('./2_objectdetection/weights/ssd300_50.pth', map_location={'cuda:0':'cpu'})

net.load_state_dict(net_weights)

image_paths = [
    './2_objectdetection/data/cowboy-757575_640.jpg',
    './2_objectdetection/data/man-and-horses-2389833_960_720.png',
    './2_objectdetection/data/rodeo-515612_960_720.jpg'
]

for image_path in image_paths:

    # 1. 画像の読み込み
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # 2. 元画像の表示
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # 3. 前処理クラスの作成
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    # 4. 前処理
    phase = 'val'
    img_transformed, boxes, labels = transform(
        img, phase, '', ''
    )
    img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

    # 5. SSDで予測
    net.eval()
    x = img.unsqueeze(0)
    detections = net(x)

    print(detections.shape)
    print(detections)

    # 画像に対する予測
    result = SSDPredictShow(eval_categories=voc_classes, net=net)
    result.show(image_path, data_confidence_level=0.6)




