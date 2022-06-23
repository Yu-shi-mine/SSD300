from this import d
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# フォルダ「utils」にある関数matchを記述したmatch.pyからimport
from utils.match import match

class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu') -> None:
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = neg_pos
        self.device = device
    
    def forward(self, predictions, targets):
        # SSDモデルの出力が田プルになっているので、個々にばらす
        loc_data, conf_data, dbox_list = predictions

        # 当素数を把握
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        # 損失の計算に使用するものを格納する変数を作成
        # conf_t_label: 各DBoxに一番近い正解BBoxのラベルを格納させる
        # loc_t: 各DBoxに一番近い政界のBBoxの位置情報を格納させる
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # loc_t とconf_t_labelに、DBoxと正解アノテーションtargetsをmatchさせた結果を上書きする
        for idx in range(num_batch):
            # 現在のミニバッチの正解アノテーションのBBoxとラベルを取得
            truth = targets[idx][:, :-1].to(self.device)
            # ラベル[物体1のラベル, 物体2のラベル, ...]
            labels = targets[idx][:, :-1].to(self.device)

            # デフォルトボックスを新たな変数で用意
            dbox = dbox_list.to(self.device)

            # 関数matchを実行し、loc_tとconf_t_labelの内容を更新する
            variance = [0.1, 0.2]
            match(self.jaccard_thresh, truth, dbox, variance, labels, loc_t, conf_t_label, idx)

        # -----------------------------
        # 位置の損失: loss_lを計算
        # Smooth L1関数で損失を計算する
        # -----------------------------

        # 物体を検出したBBoxを取り出すマスクを作成
        pos_mask = conf_t_label > 0

        # pos_maskをloc_dataのサイズに変形
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # Positive DBoxのloc_dataと、教師データloc_tを取得
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # 物体を発見したPositive DBoxのオフセット情報loc_tの損失（誤差）を計算
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # -----------------------------
        # クラス予測の損失: loss_cを計算
        # 公差エントロピーで損失計算
        # Hard Negative Miningを実施
        # -----------------------------
        batch_conf = conf_data.view(-1, num_classes)

        # クラス予測の損失関数を計算
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none'
        )

        # -----------------------------
        # Hard Negative Mining のマスク作成
        # -----------------------------

        # 物体を発見したPositive DBoxの損失を0にする
        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_c = loss_c.view(num_batch, -1)
        loss_c[pos_mask] = 0

        # Hard Negative Miningの実施
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # 背景のDBoxの数num_negを決める
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

        # idx_rankは各DBoxの損失の大きさが上から何番目なのかが入っている
        # 背景のDBoxの数num_negよりも、順位が低い（すなわち損失が大きい）DBoxを取るマスクを作成
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # -----------------------------
        # マスク作成終了
        # -----------------------------

        # マスクの形江尾整形し、conf_dataに合わせる
        # pos_idx_maskはPositive DBoxのconfを取り出すマスク
        # neg_idx_maskはHard Negative Miningで抽出したNegative DBoxのconfを取り出すマスク
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # conf_dataからposとnegだけ取り出してconf_hnmにする
        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)

        # 教師データであるconf_t_labelからposとnegだけ取り出してconf_t_label_hnmに
        conf_t_label_hnm = conf_t_label[(pos_mask + neg_mask).gt(0)]

        # confidenceの損失関数を計算（要素の合計=sumを求める）
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        # 物体を発見したBBoxの数Ｎ（全ミニバッチの合計）で割り算
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c






