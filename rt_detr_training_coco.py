########################################
# rt_detr_training_coco.py
########################################
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection

from config.config_coco import Config
from rt_detr import RTDETRv2, DETRLoss, HungarianMatcher, postprocess_nms
from rt_detr import generalized_iou

class COCOWrapper(CocoDetection):
    """
    In this example, we sort the cat_ids from COCO and map them consistently to label indices.
    Bboxes are converted from [x,y,w,h] to normalized (cx,cy,w,h).
    """
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform=transform)
        self.coco = self.coco
        cat_ids = self.coco.getCatIds()
        cat_ids = sorted(cat_ids) 
        self.catId_to_labelIdx = {0: 0}  # background => 0
        for idx, cat_id in enumerate(cat_ids, start=1):
            self.catId_to_labelIdx[cat_id] = idx

    def __getitem__(self, index):
        img, anns = super().__getitem__(index)
        boxes = []
        labels = []

        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        width, height = img_info["width"], img_info["height"]

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.catId_to_labelIdx:
                continue
            label_idx = self.catId_to_labelIdx[cat_id]
            x, y, w, h = ann["bbox"]  # [x_min, y_min, w, h]
            cx = (x + w/2.0) / width
            cy = (y + h/2.0) / height
            w  = w / width
            h  = h / height
            boxes.append([cx, cy, w, h])
            labels.append(label_idx)

        boxes  = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return img, boxes, labels

def build_transform(train=True):
    """
    Example of random data augmentation with multi-scale,
    random horizontal flip, color jitter, etc.
    """
    transforms_list = []
    if train:
        transforms_list.append(T.Resize((640, 640)))
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        transforms_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    else:
        transforms_list.append(T.Resize((640, 640)))

    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return T.Compose(transforms_list)

def collate_fn(batch):
    """
    Merges a list of samples to form a mini-batch for COCO.
    Each sample: (img_tensor, boxes, labels)
    """
    imgs, boxes, labels = list(zip(*batch))
    imgs_tensor = torch.stack(imgs, dim=0)
    return imgs_tensor, boxes, labels

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch_idx):
    model.train()
    criterion.train()
    total_loss = 0.0

    for step, (images, boxes, labels) in enumerate(data_loader):
        images = images.to(device)

        # build targets
        targets = []
        for b, l in zip(boxes, labels):
            targets.append({"labels": l.to(device), "boxes": b.to(device)})

        optimizer.zero_grad()
        pred_logits, pred_boxes = model(images)  
        print(pred_logits.argmax(dim=-1))

        loss_dict = criterion({"pred_logits": pred_logits, "pred_boxes": pred_boxes}, targets)
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch[{epoch_idx}] Train Loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    total_loss = 0.0
    for step, (images, boxes, labels) in enumerate(data_loader):
        images = images.to(device)
        targets = []
        for b, l in zip(boxes, labels):
            targets.append({"labels": l.to(device), "boxes": b.to(device)})

        pred_logits, pred_boxes = model(images)
        loss_dict = criterion({"pred_logits": pred_logits, "pred_boxes": pred_boxes}, targets)
        total_loss += loss_dict["loss_total"].item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Checkpoints will be saved to: {Config.OUTPUT_DIR}")
    print(f"[INFO] Using device: {device}")

    # 1) Build dataset
    train_transform = build_transform(train=True)
    val_transform   = build_transform(train=False)

    train_dataset = COCOWrapper(
        root=Config.COCO_IMG_TRAIN,
        annFile=Config.COCO_ANN_TRAIN,
        transform=train_transform
    )
    val_dataset = COCOWrapper(
        root=Config.COCO_IMG_VAL,
        annFile=Config.COCO_ANN_VAL,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    # 2) Build model
    model = RTDETRv2(
        num_classes=Config.NUM_CLASSES,
        num_queries=Config.NUM_QUERIES,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        num_feature_levels=Config.NUM_FEATURE_LEVELS,
        enc_n_points=Config.ENC_N_POINTS,
        dec_n_points=Config.DEC_N_POINTS,
        aux_loss=Config.AUX_LOSS
    ).to(device)

    # 3) Build loss
    matcher = HungarianMatcher(
        class_weight=Config.CLASS_WEIGHT,
        bbox_weight=Config.BBOX_WEIGHT,
        giou_weight=Config.GIOU_WEIGHT
    )
    criterion = DETRLoss(
        num_classes=Config.NUM_CLASSES,
        matcher=matcher,
        eos_coef=0.1,
        class_weight=Config.CLASS_WEIGHT,
        bbox_weight=Config.BBOX_WEIGHT,
        giou_weight=Config.GIOU_WEIGHT
    ).to(device)

    # 4) Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    best_val_loss = float("inf")

    # 5) Training Loop
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch)
        val_loss   = evaluate(model, criterion, val_loader, device)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(Config.OUTPUT_DIR, f"model_best_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  => Saved best checkpoint: {ckpt_path}")

    print("Training Completed.")

    # 6) Optional Inference + NMS
    if Config.USE_NMS:
        print("Performing a quick inference with NMS post-processing on val set:")
        images, boxes, labels = next(iter(val_loader))
        images = images.to(device)
        with torch.no_grad():
            pred_logits, pred_boxes = model(images)
            # apply NMS
            results = postprocess_nms(
                pred_logits,
                pred_boxes,
                score_thresh=Config.SCORE_THRESH,
                iou_thresh=Config.NMS_THRESH
            )
        print("NMS results example (first batch):")
        for idx, r in enumerate(results):
            print(f"Image {idx} => {len(r['boxes'])} final detections.")

if __name__ == "__main__":
    main()
