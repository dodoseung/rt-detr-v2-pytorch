import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from config.config_coco import Config
from rt_detr_training_coco import COCOWrapper, collate_fn
from torch.utils.data import DataLoader
import torchvision.transforms as T

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

def visualize_coco_samples(dataset, indices=[0, 10, 20, 30, 40], num_show=5):
    """
    dataset에서 몇 개 샘플을 가져와 바운딩박스와 레이블을 시각화합니다.
    
    Args:
        dataset: COCO Wrapper로 만든 dataset (train_dataset or val_dataset)
        indices: 시각화할 샘플 인덱스 목록
        num_show: 표시할 샘플 개수
    """
    # 역정규화에 사용할 mean, std (학습 시와 동일)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    for i, idx in enumerate(indices[:num_show]):
        img_tensor, boxes, labels = dataset[idx]  # (C,H,W), (N,4), (N,)

        # 1) 역정규화
        # img_tensor 형식: [3, H, W], 범위: [-x, +x]
        # => (img * std + mean) 후 [0,1] 범위로 clamp
        unnorm_img = img_tensor.clone()
        unnorm_img = unnorm_img * std + mean
        unnorm_img = unnorm_img.clamp(0, 1)  # (3,H,W)

        # 2) NumPy 변환 (H,W,3)
        np_img = unnorm_img.permute(1,2,0).cpu().numpy()

        # 3) matplotlib으로 시각화
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.imshow(np_img)  # 배경 이미지

        H_img, W_img = np_img.shape[:2]  # (H,W)
        
        # boxes: (cx,cy,w,h) in [0,1]
        for b_idx, box in enumerate(boxes):
            cx, cy, w, h = box.tolist()
            # 픽셀 좌표 변환
            x_min = (cx - w/2) * W_img
            y_min = (cy - h/2) * H_img
            box_w = w * W_img
            box_h = h * H_img

            rect = patches.Rectangle(
                (x_min, y_min), box_w, box_h,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            # 라벨 표시
            lbl = labels[b_idx].item()
            if 0 <= lbl < len(COCO_CATEGORY_NAMES):
                cls_name = COCO_CATEGORY_NAMES[lbl]
            else:
                cls_name = f"cls{lbl}"
            text_str = f"{cls_name}"
            ax.text(
                x_min, y_min-5, text_str,
                bbox=dict(facecolor='yellow', alpha=0.5),
                fontsize=9, color='black'
            )

        ax.set_title(f"Dataset idx={idx}")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"train_sample_{idx}.png") 

def main():
    # 1) Transform
    transform = T.Compose([
        T.Resize((480, 480)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])

    # 2) Dataset 로드 (train & val)
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

    # 3) 시각화: train dataset
    print("[Train Dataset] Visualizing samples")
    visualize_coco_samples(train_dataset, indices=[0,5,10,15,20])

    # 4) 시각화: val dataset
    print("[Val Dataset] Visualizing samples")
    visualize_coco_samples(val_dataset, indices=[0,2,4,6,8])

if __name__ == "__main__":
    main()
