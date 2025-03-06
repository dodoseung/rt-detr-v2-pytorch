import argparse
import os
import time
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from rt_detr import RTDETR
from config.config_coco import Config

# COCO dataset
COCO_CATEGORY_NAMES = [
    "background",  # 0
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the trained RT-DETR model (.pth)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with input images")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="Directory to save output images with detections")
    parser.add_argument("--score_thresh", type=float, default=0.5,
                        help="Confidence threshold for showing bounding boxes")
    parser.add_argument("--image_size", type=int, default=480,
                        help="Inference image resize (width=height)")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model loading
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

    # Checkpoint loading
    print(f"Loading checkpoint from {args.ckpt} ...")
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Transform
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])

    # Directory
    image_files = sorted(os.listdir(args.input_dir))
    print(f"Found {len(image_files)} images in {args.input_dir}")

    # Inference
    for img_name in image_files:
        img_path = os.path.join(args.input_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        # Image load
        pil_img = Image.open(img_path).convert("RGB")
        W_orig, H_orig = pil_img.size
        img_tensor = transform(pil_img).unsqueeze(0).to(device)  # (1, 3, H, W)

        # Inference
        with torch.no_grad():
            pred_logits, pred_boxes = model(img_tensor)  # (1, Q, num_classes), (1, Q, 4)

        # pred_logits: shape (1, Q, num_classes) => (Q, num_classes)
        # pred_boxes: shape (1, Q, 4) => (Q, 4)
        pred_logits = pred_logits[0]  # (Q, num_classes)
        pred_boxes  = pred_boxes[0]   # (Q, 4)

        # Softmax
        probs = F.softmax(pred_logits, dim=-1)  # (Q, num_classes)
        scores, labels = probs.max(dim=-1)      # (Q,), (Q,)

        # Thresholding
        keep = (labels != 0) & (scores > args.score_thresh)  # 0=배경
        keep_indices = keep.nonzero(as_tuple=True)[0].tolist()

        # Pixel
        ratio_x = W_orig / float(args.image_size)
        ratio_y = H_orig / float(args.image_size)

        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()

        for i in keep_indices:
            cls_id = labels[i].item()
            score  = scores[i].item()
            cx, cy, w, h = pred_boxes[i].tolist()  # [0,1] 범위

            # (cx,cy,w,h) -> (xmin,ymin,xmax,ymax) in resized scale
            x_min = (cx - 0.5*w) * args.image_size
            y_min = (cy - 0.5*h) * args.image_size
            x_max = (cx + 0.5*w) * args.image_size
            y_max = (cy + 0.5*h) * args.image_size

            # Ratio
            x_min *= ratio_x
            x_max *= ratio_x
            y_min *= ratio_y
            y_max *= ratio_y

            # Drawing
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            cls_name = COCO_CATEGORY_NAMES[cls_id] if cls_id < len(COCO_CATEGORY_NAMES) else f"cls{cls_id}"
            text = f"{cls_name} {score:.2f}"
            print(text)
            draw.text((x_min, y_min), text, fill="red", font=font)

        # Saving
        out_path = os.path.join(args.output_dir, f"det_{img_name}")
        pil_img.save(out_path)
        print(f"[SAVE] {out_path}")

    print("Done. All results saved to", args.output_dir)

if __name__ == "__main__":
    main()
