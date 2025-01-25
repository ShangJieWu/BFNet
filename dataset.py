import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from utils import seed_worker


class PolypDataset(Dataset):
    def __init__(self, train, txt_file, transforms):
        super().__init__()
        self.train = train
        self.images = []
        data_root = os.path.split(txt_file)[0]
        with open(txt_file, 'r') as f:
            for line in f:
                filename = line.strip()
                filepath = os.path.join(data_root, "image", filename)
                self.images.append(filepath)
        self.masks = [filepath.replace("image", "mask") for filepath in self.images]
        self.edges = [filepath.replace("image", "canny_edge") for filepath in self.images]
        self.transforms = transforms

    def __getitem__(self, idx):
        transforms = self.transforms
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        edge = Image.open(self.edges[idx]).convert("L")
        image_name = os.path.basename(self.images[idx])
        image_size = str(mask.size)
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)
        edge = tv_tensors.Mask(edge)
        image, mask, edge = transforms(image, mask, edge)
        mask = mask / 255.
        edge = edge / 255.
        mask = (mask > 0.5).float()
        return image, mask, edge, image_name, image_size

    def __len__(self):
        return len(self.images)
class testdataset(Dataset):
    def __init__(self, train, txt_file, transforms):
        super().__init__()
        self.train = train
        self.images = []
        data_root = os.path.split(txt_file)[0]
        with open(txt_file, 'r') as f:
            for line in f:
                filename = line.strip()
                filepath = os.path.join(data_root, "images", filename)
                self.images.append(filepath)
        self.masks = [filepath.replace("images", "masks") for filepath in self.images]
        self.transforms = transforms

    def __getitem__(self, idx):
        transforms = self.transforms
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        image_name = os.path.basename(self.images[idx])
        image_size = str(mask.size)
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)
        image, mask = transforms(image, mask)
        mask = mask / 255.
        mask = (mask > 0.5).float()
        return image, mask, image_name, image_size

    def __len__(self):
        return len(self.images)

def get_loader(args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.data_aug:
        train_transforms = v2.Compose([
            v2.Resize(size=(352, 352), antialias=True),
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=15),
            v2.ToDtype({tv_tensors.Image: torch.float, tv_tensors.Mask: torch.float}, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
            v2.ToPureTensor(),
        ])
    else:
        train_transforms = v2.Compose([
            v2.Resize(size=(352, 352), antialias=True),
            v2.RandomHorizontalFlip(p=0.33),
            v2.ToDtype({tv_tensors.Image: torch.float, tv_tensors.Mask: torch.float}, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
            v2.ToPureTensor(),
        ])
    val_transforms = v2.Compose([
        v2.Resize(size=(352, 352), antialias=True),
        v2.ToDtype({tv_tensors.Image: torch.float, tv_tensors.Mask: torch.float}, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
        v2.ToPureTensor(),
    ])
    train_set = PolypDataset(train=True, txt_file=args.train_txt, transforms=train_transforms)
    if not args.seed == 0:
        print("Using seed_worker for DataLoader")
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  num_workers=args.num_workers, pin_memory=True)
    val_loaders = {}
    for val_txt in args.val_txt:
        dataset_name = os.path.basename(val_txt)[:-4]
        val_set = testdataset(train=False, txt_file=val_txt, transforms=val_transforms)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loaders[dataset_name] = val_loader
    return train_loader, val_loaders


if __name__ == "__main__":
    import argparse

    from torchvision.utils import save_image
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", type=str, default="/public/wsj/datasets/polyp_PraNet/TrainDataset/train.txt")
    parser.add_argument("--val_txt", type=str, nargs='+', default=["/public/wsj/datasets/polyp_PraNet/TestDataset/Kvasir.txt"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_loader, val_loaders = get_loader(args)
    print(len(train_loader))
    
    for i, (image, mask, edge) in enumerate(train_loader):
        print(image.shape, mask.shape, edge.shape, i)
        save_image(image, "image.png")
        save_image(mask, "mask.png")
        save_image(edge, "edge.png")
        break
    for (image, mask, edge) in val_loaders["Kvasir"]:
        print(image.shape, mask.shape, edge.shape)
        break
