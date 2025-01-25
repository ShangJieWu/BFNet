import torch
import torch.nn.functional as F
from model.model import MyPolypPVT
import argparse
from dataset import get_loader
from utils import get_metrics
import cv2
import numpy as np
import os

def evaluate(model, val_loader, device, name, args):
    DICE, IOU, ACC, RECALL, F2, MAE = 0, 0, 0, 0, 0, 0
    for (image, mask, image_name, image_size) in val_loader:
        image, mask = image.to(device), mask.to(device)
        image_name = image_name[0]
        if name in ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300", "Kvasir"]:
            image_name = image_name.split("_")[-1]
        image_size = eval(image_size[0])
        with torch.inference_mode():
            pred, _,_,_,_ = model(image)
            pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            pred_save = F.interpolate(pred, size=(image_size[1], image_size[0]), mode="bilinear", align_corners=False)
            pred_save = pred_save.sigmoid()
            pred_save = pred_save.cpu().numpy()
            pred_save = (pred_save - pred_save.min()) / (pred_save.max() - pred_save.min() + 1e-8)
            cv2.imwrite(os.path.join(args.save_dir, name, image_name), pred_save[0, 0, :, :] * 255)
            pred = pred.sigmoid()
            dice, iou, acc, recall, f2, mae = get_metrics(pred, mask)
            DICE += dice.sum().item()
            IOU += iou.sum().item()
            ACC += acc.sum().item()
            RECALL += recall.sum().item()
            F2 += f2.sum().item()
            MAE += mae.sum().item()

    mdice = DICE / len(val_loader.dataset)
    miou = IOU / len(val_loader.dataset)
    macc = ACC / len(val_loader.dataset)
    mrecall = RECALL / len(val_loader.dataset)
    mf2 = F2 / len(val_loader.dataset)
    mmae = MAE / len(val_loader.dataset)

    return mdice, miou, macc, mrecall, mf2, mmae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/public/xxx/projects/BFNet/checkpoint.pth")
    parser.add_argument("--val_txt", type=str, nargs='+',
                        default=["/public/xxx/datasets/polyp_PraNet/TestDataset/CVC-ClinicDB.txt",
                                 "/public/xxx/datasets/polyp_PraNet/TestDataset/CVC-ColonDB.txt",
                                 "/public/xxx/datasets/polyp_PraNet/TestDataset/ETIS-LaribPolypDB.txt",
                                 "/public/xxx/datasets/polyp_PraNet/TestDataset/Kvasir.txt",
                                 "/public/xxx/datasets/polyp_PraNet/TestDataset/CVC-300.txt"])
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_results",default=True ,action="store_true")
    parser.add_argument("--save_note", type=str, default="NBI1K")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.batch_size = 1
    args.save_dir = "./results/{}".format(args.save_note)
    _, val_loaders = get_loader(args)

    device = torch.device("cuda")
    model = MyPolypPVT().to(device)
    state_dict = torch.load(args.checkpoint)
    print(args.checkpoint)
    state_dict.pop("backbone.head.weight", None)
    state_dict.pop("backbone.head.bias", None)
    model.load_state_dict(state_dict)
    model.eval()

    for name, val_loader in val_loaders.items():
        if not os.path.exists("./{}/{}".format(args.save_dir, name)):
            os.makedirs("./{}/{}".format(args.save_dir, name))
        mdice, miou, macc, mrecall, mf2, mmae = evaluate(model, val_loader, device, name, args)
        print("{:<17}: mdice:{:.6f}, miou:{:.6f}, macc:{:.6f}, mrecall:{:.6f}, mf2:{:.6f}, mmae:{:.6f}".format(name, mdice, miou, macc, mrecall, mf2, mmae))
