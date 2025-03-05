import os
import time
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import numpy as np

from rt_detr import RTDETR, HungarianMatcher, DETRLoss, generalized_iou
from config.config_coco import Config


# COCO dataset
COCO_CATEGORY_NAMES = [
    "background",  # 0번
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

class COCOWrapper(CocoDetection):
    """
    CocoDetection
    annotation bbox: [x, y, w, h] -> (cx, cy, w, h)
    label category_id -> index
    """
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform=transform)
        self.catId_to_labelIdx = {}
        for i, name in enumerate(COCO_CATEGORY_NAMES):
            if i == 0:
                # background
                continue
            pass

        for cat_id in range(1, 81):
            self.catId_to_labelIdx[cat_id] = cat_id  # background=0, person=1,...

    def __getitem__(self, index):
        img, anns = super().__getitem__(index)
        boxes  = []
        labels = []

        # Get image ID and height and width
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        width, height = img_info["width"], img_info["height"]

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.catId_to_labelIdx:
                continue
            label_idx = self.catId_to_labelIdx[cat_id]

            # bbox
            x, y, w, h = ann["bbox"]  # coco: [x_min, y_min, w, h]
            cx = (x + (w / 2)) / width
            cy = (y + (h / 2)) / height
            w = w / width
            h = h / height

            boxes.append([cx, cy, w, h])
            labels.append(label_idx)

        boxes  = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return img, boxes, labels
    

def collate_fn(batch):
    imgs, boxes, labels = list(zip(*batch))
    imgs = torch.stack(imgs, dim=0)
    return imgs, boxes, labels

# Training
def train_one_epoch(model, criterion, data_loader, optimizer, device):
    model.train()
    criterion.train()
    total_loss = 0.0

    for step, (images, boxes, labels) in enumerate(data_loader):
        images = images.to(device)

        # Target
        targets = []
        for b, l in zip(boxes, labels):
            targets.append({
                "labels": l.to(device),
                "boxes":  b.to(device)
            })

        optimizer.zero_grad()
        pred_logits, pred_boxes = model(images)  # (B, Q, num_classes), (B, Q, 4)
        loss_dict = criterion(
            {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
            targets
        )

        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Evaluation
@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    """
    Accuracy, IoU, mAP(Prec/Recall/F1)
    """
    model.eval()
    criterion.eval()

    total_matched = 0
    correct_class = 0
    sum_iou = 0.0

    tp = 0
    fp = 0
    fn = 0

    for images, boxes, labels in data_loader:
        images = images.to(device)

        targets = []
        for b, l in zip(boxes, labels):
            targets.append({"labels": l.to(device), "boxes": b.to(device)})

        pred_logits, pred_boxes = model(images)
        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        # Matching
        indices = criterion.matcher(outputs, targets)

        batch_size = images.shape[0]
        for b_idx in range(batch_size):
            q_idx, t_idx = indices[b_idx]
            if len(q_idx) == 0:
                fn += len(targets[b_idx]["labels"])
                continue

            pred_cls = pred_logits[b_idx, q_idx].argmax(dim=-1)
            gt_cls   = targets[b_idx]["labels"][t_idx]

            # IoU
            ious_mat = generalized_iou(pred_boxes[b_idx, q_idx], targets[b_idx]["boxes"][t_idx])
            ious_diag = ious_mat.diagonal()

            matched_count = len(q_idx)
            total_matched += matched_count
            correct_class += (pred_cls == gt_cls).sum().item()
            sum_iou       += ious_diag.sum().item()

            # TP: (pred_cls==gt_cls) & (IoU>0.5)
            mask_tp = (pred_cls == gt_cls) & (ious_diag > 0.5)
            tp_batch = mask_tp.sum().item()

            # No background
            pred_bg_mask = (pred_logits[b_idx].argmax(dim=-1) == 0)
            pred_non_bg  = (~pred_bg_mask).sum().item()
            fp_batch = pred_non_bg - tp_batch

            gt_count = len(targets[b_idx]["labels"])
            fn_batch = gt_count - tp_batch

            tp += tp_batch
            fp += fp_batch
            fn += fn_batch

    if total_matched > 0:
        accuracy = correct_class / total_matched
        avg_iou  = sum_iou / total_matched
    else:
        accuracy = 0.0
        avg_iou  = 0.0

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "accuracy": accuracy,
        "avg_iou":  avg_iou,
        "precision": precision,
        "recall":    recall,
        "f1":        f1
    }

@torch.no_grad()
def measure_inference_speed(model, data_loader, device, max_iter=50):
    model.eval()
    start = time.time()
    total_images = 0

    for i, (images, boxes, labels) in enumerate(data_loader):
        images = images.to(device)
        _ = model(images)  # forward
        total_images += images.size(0)
        if (i+1) >= max_iter:
            break

    elapsed = time.time() - start
    fps = total_images / elapsed
    return fps

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Checkpoints will be saved to: {Config.OUTPUT_DIR}")

    transform = T.Compose([
        T.Resize((480, 480)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = COCOWrapper(
        root=Config.COCO_IMG_TRAIN,
        annFile=Config.COCO_ANN_TRAIN,
        transform=transform
    )
    val_dataset = COCOWrapper(
        root=Config.COCO_IMG_VAL,
        annFile=Config.COCO_ANN_VAL,
        transform=transform
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

    # Model, Matcher, Loss
    model = RTDETR(
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
        dec_n_points=Config.DEC_N_POINTS
    ).to(device)

    matcher = HungarianMatcher(class_weight=1.0, bbox_weight=5.0, giou_weight=2.0)
    criterion = DETRLoss(num_classes=Config.NUM_CLASSES, matcher=matcher, eos_coef=0.1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    # Train
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
        print(f"[Epoch {epoch+1}/{Config.EPOCHS}] Train Loss: {train_loss:.4f}")

        eval_res = evaluate(model, criterion, val_loader, device)
        acc     = eval_res["accuracy"]
        avg_iou = eval_res["avg_iou"]
        prec    = eval_res["precision"]
        rec     = eval_res["recall"]
        f1      = eval_res["f1"]

        print(f"    [Eval] Acc={acc:.3f}, IoU={avg_iou:.3f}, "
              f"Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

        fps = measure_inference_speed(model, val_loader, device, max_iter=50)
        print(f"    [Inference Speed] ~{fps:.2f} FPS (50 batches)")

        ckpt_path = os.path.join(Config.OUTPUT_DIR,
                                 f"model_epoch_{epoch+1}_loss_{train_loss:.4f}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"    => Saved checkpoint: {ckpt_path}")

    print("Training Finished.")


if __name__ == "__main__":
    main()
