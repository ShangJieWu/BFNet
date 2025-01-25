import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import distance_transform_edt as bwdist
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import distance_transform_edt

""" Loss Functions -------------------------------------- """


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class MultiClassBCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        loss = []
        for i in range(inputs.shape[1]):
            yp = inputs[:, i]
            yt = targets[:, i]
            BCE = F.binary_cross_entropy(yp, yt, reduction='mean')

            if i == 0:
                loss = BCE
            else:
                loss += BCE

        return loss


""" Metrics ------------------------------------------ """


class cal_sm(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def cal(self, pred, gt):
        gt = gt > 0.5
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = ndimage.center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score


class cal_em(object):
    # Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self):
        self.prediction = []

    def cal(self, pred, gt):
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(gt.shape)
        FM[pred >= th] = 1
        FM = np.array(FM, dtype=bool)
        GT = np.array(gt, dtype=bool)
        dFM = np.double(FM)
        if (sum(sum(np.double(GT))) == 0):
            enhanced_matrix = 1.0 - dFM
        elif (sum(sum(np.double(~GT))) == 0):
            enhanced_matrix = dFM
        else:
            dGT = np.double(GT)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        [w, h] = np.shape(GT)
        score = sum(sum(enhanced_matrix)) / (w * h - 1 + 1e-8)
        return score

    def AlignmentTerm(self, dFM, dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        align_Matrix = 2. * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + 1e-8)
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix):
        enhanced = np.power(align_Matrix + 1, 2) / 4
        return enhanced


class cal_wfm(object):
    def __init__(self, beta=1):
        self.beta = beta
        self.eps = 1e-6

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def cal(self, pred, gt):
        gt = gt > 0.5
        if gt.max() == 0:
            Q = 0
        else:
            # [Dst,IDXT] = bwdist(dGT);
            Dst, Idxt = bwdist(gt == 0, return_indices=True)

            # %Pixel dependency
            # E = abs(FG-dGT);
            E = np.abs(pred - gt)
            # Et = E;
            # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
            Et = np.copy(E)
            Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

            # K = fspecial('gaussian',7,5);
            # EA = imfilter(Et,K);
            # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
            K = self.matlab_style_gauss2D((7, 7), sigma=5)
            EA = convolve(Et, weights=K, mode='constant', cval=0)
            MIN_E_EA = np.where(gt & (EA < E), EA, E)

            # %Pixel importance
            # B = ones(size(GT));
            # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
            # Ew = MIN_E_EA.*B;
            B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
            Ew = MIN_E_EA * B

            # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
            # FPw = sum(sum(Ew(~GT)));
            TPw = np.sum(gt) - np.sum(Ew[gt == 1])
            FPw = np.sum(Ew[gt == 0])

            # R = 1- mean2(Ew(GT)); %Weighed Recall
            # P = TPw./(eps+TPw+FPw); %Weighted Precision
            R = 1 - np.mean(Ew[gt])
            P = TPw / (self.eps + TPw + FPw)

            # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
            Q = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return Q


class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt))


class cal_dice(object):
    # mean absolute error
    def __init__(self):
        self.pred = []

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


class cal_iou(object):
    # mean absolute error
    def __init__(self):
        self.pred = []

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_pred = y_pred > 0.5
        target = y_true > 0.5
        intersection = (y_pred & target).sum()
        union = (y_pred | target).sum()
        return (intersection + smooth) / (union + smooth)


class cal_acc(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def cal(self, y_pred, y_true):
        binary = np.zeros_like(y_pred)
        binary[y_pred >= 0.5] = 1
        hard_gt = np.zeros_like(y_true)
        hard_gt[y_true > 0.5] = 1
        tp = (binary * hard_gt).sum()
        tn = ((1 - binary) * (1 - hard_gt)).sum()
        Np = hard_gt.sum()
        Nn = (1 - hard_gt).sum()
        acc = ((tp + tn) / (Np + Nn))
        return acc


def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def F2(y_true, y_pred, beta=2):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta ** 2.) * (p * r) / float(beta ** 2 * p + r + 1e-15)


def dice_score(y_true, y_pred):
    dice = cal_dice()
    return dice.cal(y_pred, y_true)


def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def sm_score(y_true, y_pred):
    sm = cal_sm()
    return sm.cal(y_pred, y_true)


def em_score(y_true, y_pred):
    em = cal_em()
    return em.cal(y_pred, y_true)


def wfm_score(y_true, y_pred):
    wfm = cal_wfm()
    return wfm.cal(y_pred, y_true)


def mae_socre(y_true, y_pred):
    mae = cal_mae()
    return mae.cal(y_pred, y_true)


def acc_score(y_true, y_pred):
    acc = cal_acc()
    return acc.cal(y_pred, y_true)


def iou_score(y_true, y_pred):
    iou = cal_iou()
    return iou.cal(y_true, y_pred)

def calculate_metrics(y_true, y_pred):
    # c, h, w = y_true.shape()
    # y_true = y_true.detach().cpu().numpy()
    # y_pred = y_pred.detach().cpu().numpy()
    #
    # y_pred = y_pred.astype(np.float64)
    #
    # # y_true[y_true > 0.5] = 1
    # # y_true[y_true != 1] = 0
    #
    # y_p = y_pred.reshape(h, w)
    # y_t = y_true.reshape(h, w)

    score_smeasure = sm_score(y_true, y_pred)
    score_wfmeasure = wfm_score(y_true, y_pred)
    score_mae = mae_socre(y_true, y_pred)
    score_emeasure = em_score(y_true, y_pred)
    score_fmeasure = F2(y_true, y_pred)

    # y_pred = y_pred.astype(np.uint8)
    # y_true = y_true.astype(np.uint8)
    #
    # y_pred = y_pred.reshape(-1)
    # y_true = y_true.reshape(-1)

    ## Score
    score_f1 = dice_score(y_true, y_pred)
    score_acc = acc_score(y_true, y_pred)
    score_iou = iou_score(y_true, y_pred)

    return [score_f1, score_iou, score_acc, score_smeasure, score_emeasure, score_wfmeasure, score_fmeasure, score_mae]