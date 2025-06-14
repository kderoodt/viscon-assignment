#!/usr/bin/env python3
import argparse, time, pathlib, random
from typing import Tuple

import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom Dice loss function for segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        # Compute softmax over classes and isolate the positive class channel
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        targets = (targets == 1).float()  # ensure binary mask
        intersection = (probs * targets).sum(dim=(1,2))
        union = probs.sum(dim=(1,2)) + targets.sum(dim=(1,2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# Building block: two convolutional layers with batch norm and ReLU
class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

# Basic U-Net architecture implementation
class UNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # Downsampling path
        self.d1 = DoubleConv(1,   64); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128); self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128,256); self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256,512); self.p4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(512,1024)  # bottleneck

        # Upsampling path
        self.u4 = nn.ConvTranspose2d(1024,512,2,2); self.c4 = DoubleConv(1024,512)
        self.u3 = nn.ConvTranspose2d(512, 256,2,2); self.c3 = DoubleConv(512 ,256)
        self.u2 = nn.ConvTranspose2d(256,128,2,2); self.c2 = DoubleConv(256 ,128)
        self.u1 = nn.ConvTranspose2d(128,64 ,2,2); self.c1 = DoubleConv(128 ,64)

        self.out= nn.Conv2d(64, n_classes, 1)  # final 1x1 convolution

    def forward(self,x):
        # Encoder
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        x  = self.mid(self.p4(d4))

        # Decoder with skip connections
        x  = self.c4(torch.cat([self.u4(x), d4], 1))
        x  = self.c3(torch.cat([self.u3(x), d3], 1))
        x  = self.c2(torch.cat([self.u2(x), d2], 1))
        x  = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)

<<<<<<< Updated upstream

=======
# Custom dataset for loading rail images and masks
>>>>>>> Stashed changes
class RailSet(Dataset):
    def __init__(self, root: pathlib.Path, crop: int=0, train: bool = True):
        self.imgs  = sorted((root/'images').glob('*.png'))
        assert self.imgs, f"No images found in {root/'images'}"
        self.mask_dir = root/'masks'
        self.crop = crop

        # Define augmentations for training
        if train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ])
        else:
            self.transform = None

    def __len__(self): return len(self.imgs)

    def _load_img_mask(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        # Load image and mask from disk
        img_p  = self.imgs[idx]
        mask_p = self.mask_dir/img_p.name
        img  = cv2.imread(str(img_p),  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)

        if img is None or mask is None:
            raise FileNotFoundError(f"Failed reading {img_p} or {mask_p}")

        if mask.ndim == 3: mask = mask[...,0]  
        mask = (mask > 0).astype(np.uint8)     

        # Optionally crop image and mask around centre -> NOT USED
        if self.crop:
            h,w = img.shape; ch=cw=self.crop
            y0,x0 = (h-ch)//2, (w-cw)//2
            img, mask = img[y0:y0+ch, x0:x0+cw], mask[y0:y0+ch, x0:x0+cw]
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load_img_mask(idx)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask

# Compute Intersection-over-Union score for a batch
@torch.no_grad()
def iou_batch(logits, masks):
    pred = logits.argmax(1).cpu().ravel()
    gt   = masks.cpu().ravel()
    return jaccard_score(gt, pred, average='binary', zero_division=1)

# Train one epoch
def train_one_epoch(model, loader, loss_fn, opt, device, scaler):
    model.train(); running=[]
    for imgs, masks in tqdm(loader, desc='train', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(imgs); loss=loss_fn(logits, masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            logits = model(imgs); loss=loss_fn(logits, masks)
            loss.backward(); opt.step()
        running.append(loss.item())
    return float(np.mean(running))

# Run validation over dataset
@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch):
    model.eval(); losses,ious=[],[]
    for b,(imgs,masks) in enumerate(loader):
        imgs,masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        losses.append(loss_fn(logits, masks).item())
        ious.append(iou_batch(logits,masks))
<<<<<<< Updated upstream

        if epoch == 20:
            for b, (imgs, masks) in enumerate(loader):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)

                for i in range(imgs.size(0)):
                    idx = b * loader.batch_size + i
                    gt = masks[i].cpu().numpy() * 255
                    raw_pred = logits.argmax(1)[i].cpu().numpy() * 255
                    cleaned_pred = clean_mask(raw_pred.astype(np.uint8), min_area=100)
                    cv2.imwrite(f"val_pred_epoch{epoch}_sample{idx}.png", cleaned_pred)
                    cv2.imwrite(f"val_gt_epoch{epoch}_sample{idx}.png", gt)
=======
>>>>>>> Stashed changes

    return float(np.mean(losses)), float(np.mean(ious))

# Remove small connected components from mask
def clean_mask(mask: np.ndarray, min_area: int = 2000) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned

# Main training script
def main():
    # Argument parser
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data'       , default='dataset')
    ap.add_argument('--device'     , default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu','cuda'])
    ap.add_argument('--epochs'     , type=int, default=20)
    ap.add_argument('--batch_size' , type=int, default=2)
    ap.add_argument('--lr'         , type=float, default=1e-4)
    ap.add_argument('--crop'       , type=int, default=0)
    ap.add_argument('--amp'        , action='store_true', help='mixed-precision (CUDA)')
    ap.add_argument('--export_onnx')
    args = ap.parse_args()

    # Set device
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Load datasets and dataloaders
    ds_tr = RailSet(pathlib.Path(args.data) / 'train', crop=args.crop, train=True)
    ds_va = RailSet(pathlib.Path(args.data) / 'val', crop=args.crop, train=False)
    pin = (device.type=='cuda')
    tr_dl=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=True ,num_workers=4,pin_memory=pin)
    va_dl=DataLoader(ds_va,batch_size=1              ,shuffle=False,num_workers=2,pin_memory=pin)

    # Load segmentation model from SMP (ResNet34 U-Net)
    net = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=2,
    ).to(device)

    # Optimiser and loss functions
    opt=torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler=torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=='cuda'))
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).to(device))  # class balancing
    dice_loss = DiceLoss().to(device)

    # Combined loss function
    def loss_fn(logits, targets):
        return ce_loss(logits, targets) + dice_loss(logits, targets)

    best_miou = 0
    for ep in range(1,args.epochs+1):
        t0=time.time()
        tr_loss=train_one_epoch(net,tr_dl,loss_fn,opt,device,scaler)
        va_loss,miou=validate(net,va_dl,loss_fn,device,ep)
        print(f"Epoch {ep:02}/{args.epochs}  loss {tr_loss:.3f}  "
              f"val {va_loss:.3f}  mIoU {miou:.3f}  ({time.time()-t0:.1f}s)")

        # Save best model
        if miou > best_miou:
            best_miou = miou
            torch.save(net.state_dict(), 'rail_unet_best.pt')
            print(f'Saved best model (mIoU={best_miou:.3f}) → rail_unet_best.pt')

<<<<<<< Updated upstream
=======
    # ONNX export
>>>>>>> Stashed changes
    if args.export_onnx:
        print('Exporting ONNX model...')
        net.eval()
        dummy = torch.randn(1, 1, ds_tr[0][0].shape[-2], ds_tr[0][0].shape[-1])
        torch.onnx.export(net.cpu(), dummy, "models/rail_detector.onnx",
                          input_names=['input'], output_names=['logits'],
                          opset_version=11,
                          dynamic_axes={'input': {0: 'B'}, 'logits': {0: 'B'}})
        print('Exported ONNX → rail_detector.onnx')


if __name__ == '__main__':
    main()