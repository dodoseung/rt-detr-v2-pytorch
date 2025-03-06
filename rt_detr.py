########################################
# rt_detr.py
########################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional, List
import math

# --------------------------------------------------------
# 1) Simple Light Backbone + FPN
# --------------------------------------------------------
class ConvBNAct(nn.Module):
    """A convenience module for Conv2d -> BatchNorm2d -> Activation."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightweightBackboneFPN(nn.Module):
    """
    A MobileNet-like backbone with FPN.
    Returns a list of feature maps from multiple scales.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            ConvBNAct(3, 16, 3, 2, 1),
            ConvBNAct(16, 16, 3, 1, 1, groups=16),
            ConvBNAct(16, 32, 1, 1, 0),
        )
        # subsequent stages
        self.stage1 = nn.Sequential(
            ConvBNAct(32, 32, 3, 2, 1, groups=32),
            ConvBNAct(32, 64, 1, 1, 0),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(64, 64, 3, 2, 1, groups=64),
            ConvBNAct(64, 128, 1, 1, 0),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(128, 128, 3, 2, 1, groups=128),
            ConvBNAct(128, 256, 1, 1, 0),
        )

        # lateral + output
        self.lateral3 = nn.Conv2d(256, out_channels, 1)
        self.lateral2 = nn.Conv2d(128, out_channels, 1)
        self.lateral1 = nn.Conv2d(64,  out_channels, 1)

        self.output3 = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        self.output2 = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        self.output1 = ConvBNAct(out_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        # x shape: (B,3,H,W)
        c0 = self.stem(x)    # downsample 2x
        c1 = self.stage1(c0) # downsample 2x
        c2 = self.stage2(c1) # downsample 2x
        c3 = self.stage3(c2) # downsample 2x

        # lateral
        f3 = self.lateral3(c3)  # highest
        f2 = self.lateral2(c2)
        f1 = self.lateral1(c1)

        # top-down
        f2 = f2 + F.interpolate(f3, size=f2.shape[-2:], mode='nearest')
        f1 = f1 + F.interpolate(f2, size=f1.shape[-2:], mode='nearest')

        # output
        p3 = self.output3(f3)
        p2 = self.output2(f2)
        p1 = self.output1(f1)

        # for multi-level usage, we might add a p0, but let's keep 3 scales
        return [p1, p2, p3]  # low-level->high-level


# --------------------------------------------------------
# 2) MultiScale Deformable Attn (Simplified)
# --------------------------------------------------------
class MultiScaleDeformAttn(nn.Module):
    """
    A simplified multi-scale deformable attention module.
    Official code is more complex. This is a minimal illustration.
    """
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        # for demonstration, we won't implement the real sampling offsets
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self, 
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor
    ):
        """
        query shape: (B, N, d_model)
        reference_points shape: (B, N, L, 2)
        input_flatten shape: (B, sum(HW), d_model)
        spatial_shapes: (L, 2) -> [ (H1, W1), (H2, W2), ... ]
        level_start_index: (L,)
        """
        # dummy example: add random noise
        attn_out = query + 0.01*torch.randn_like(query)
        return self.output_proj(attn_out)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024,
                 dropout=0.1, n_levels=4, n_points=4):
        super().__init__()
        self.self_attn = MultiScaleDeformAttn(d_model, n_heads, n_levels, n_points)
        self.dropout1  = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(d_model)

        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout2  = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.dropout3  = nn.Dropout(dropout)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, src, pos, refpoints, spatial_shapes, level_start_index):
        # self-attn
        attn = self.self_attn(src + pos, refpoints, src, spatial_shapes, level_start_index)
        src2 = self.dropout1(attn)
        src  = self.norm1(src + src2)

        # feed-forward
        ff  = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        ff  = self.dropout3(ff)
        src = self.norm2(src + ff)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024,
                 dropout=0.1, n_levels=4, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn= MultiScaleDeformAttn(d_model, n_heads, n_levels, n_points)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2    = nn.LayerNorm(d_model)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3    = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt,           # (B, N, d_model)
        reference_pts, # (B, N, L, 2)
        memory,        # (B, M, d_model)
        tgt_mask, 
        tgt_key_padding_mask,
        spatial_shapes,
        level_start_index,
        pos, 
        query_pos
    ):
        # self-attn
        B, N, C = tgt.shape
        q = (tgt + query_pos).permute(1, 0, 2)  # (N, B, C)
        k = q
        v = q
        tgt2, _ = self.self_attn(q, k, v,
                                 attn_mask=tgt_mask, 
                                 key_padding_mask=tgt_key_padding_mask)
        tgt2 = tgt2.permute(1, 0, 2)
        tgt  = tgt + self.dropout1(tgt2)
        tgt  = self.norm1(tgt)

        # cross-attn
        cross = self.cross_attn(tgt + query_pos, reference_pts, 
                                memory, spatial_shapes, level_start_index)
        tgt2  = self.dropout2(cross)
        tgt   = self.norm2(tgt + tgt2)

        # feed-forward
        ff   = self.linear2(self.dropout3(F.relu(self.linear1(tgt))))
        ff   = self.dropout4(ff)
        tgt  = self.norm3(tgt + ff)

        return tgt


class DeformableTransformer(nn.Module):
    """
    This class includes encoder and decoder stacks, 
    along with auxiliary loss support.
    """
    def __init__(
        self, 
        d_model=256, 
        nhead=8, 
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024, 
        dropout=0.1,
        num_feature_levels=4,
        enc_n_points=4, 
        dec_n_points=4,
        num_queries=300,
        aux_loss=False
    ):
        super().__init__()

        # encoder
        self.encoder_layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                n_levels=num_feature_levels, 
                n_points=enc_n_points
            )
            for _ in range(num_encoder_layers)
        ])

        # decoder
        self.decoder_layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                n_levels=num_feature_levels,
                n_points=dec_n_points
            )
            for _ in range(num_decoder_layers)
        ])

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.aux_loss   = aux_loss

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.level_embed)
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.0)
        nn.init.uniform_(self.query_embed.weight, 0, 1)

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        # Build reference points for multi-level
        reference_points_list = []
        start = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            grid_y = torch.linspace(0.5, H_-0.5, H_, device=device)
            grid_x = torch.linspace(0.5, W_-0.5, W_, device=device)
            gy, gx = torch.meshgrid(grid_y, grid_x)
            gx = gx.reshape(-1)[None] / (W_ * valid_ratios[:, None, lvl])
            gy = gy.reshape(-1)[None] / (H_ * valid_ratios[:, None, lvl])
            rp = torch.stack((gx, gy), -1) # shape (B, HW, 2)
            reference_points_list.append(rp)
        reference_points = torch.cat(reference_points_list, dim=1)
        reference_points = reference_points[:, :, None, :]  # (B, sumHW, 1, 2)
        return reference_points

    def forward(self, srcs: List[torch.Tensor], masks=None):
        """
        srcs: list of multi-scale feature maps from FPN
              e.g. [p1, p2, p3, ...] each of shape (B, C, H, W)
        """
        if masks is None:
            masks = [torch.zeros(s.shape[:2], dtype=torch.bool, device=s.device) for s in srcs]

        device = srcs[0].device
        batch_size = srcs[0].shape[0]
        # compute shapes
        spatial_shapes = []
        for lvl, src in enumerate(srcs):
            b, c, h, w = src.shape
            spatial_shapes.append((h, w))
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)

        level_start_index = torch.zeros(spatial_shapes.size(0), dtype=torch.long, device=device)
        for i in range(1, spatial_shapes.size(0)):
            level_start_index[i] = level_start_index[i-1] + (spatial_shapes[i-1, 0]*spatial_shapes[i-1, 1])

        # flatten
        src_flatten = []
        for lvl, src in enumerate(srcs):
            _, c, h, w = src.shape
            flatten_ = src.flatten(2).transpose(1,2) # (B, HW, C)
            lvl_embed= self.level_embed[lvl].view(1,1,-1).repeat(batch_size, h*w, 1)
            flatten_ = flatten_ + lvl_embed
            src_flatten.append(flatten_)
        src_flatten = torch.cat(src_flatten, dim=1) # (B, sum(HW), C)

        valid_ratios = torch.ones(batch_size, len(srcs), device=device)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device)

        # pos = 0 for simplicity
        pos_embed = torch.zeros_like(src_flatten)

        # ========== Encoder ==============
        memory = src_flatten
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory, pos_embed, reference_points, spatial_shapes, level_start_index)

        # ========== Decoder ==============
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, num_queries, d_model)
        tgt = torch.zeros_like(query_embed)
        dec_ref_points = self.reference_points(query_embed).unsqueeze(2) # (B, Q, 1, 2)

        hs = []
        out = tgt
        for dec_layer in self.decoder_layers:
            out = dec_layer(out, dec_ref_points, memory,
                            tgt_mask=None, tgt_key_padding_mask=None,
                            spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index,
                            pos=pos_embed,
                            query_pos=query_embed)
            hs.append(out)

        # shape: (num_decoder_layers, B, Q, d_model)
        hs_stack = torch.stack(hs, dim=0)
        return hs_stack  # for auxiliary loss usage


# --------------------------------------------------------
# 3) DynamicConv + Detection Head
# --------------------------------------------------------
class DynamicConv(nn.Module):
    """A simplistic dynamic convolution block for demonstration."""
    def __init__(self, d_model=256, hidden_dim=1024, num_layers=2):
        super().__init__()
        layers = []
        in_dim = d_model
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hidden_dim, d_model)

    def forward(self, query, features=None):
        """
        query shape: (B, Q, d_model)
        features can be used if we want to convolve over a feature map; omitted here for simplicity.
        """
        x = self.mlp(query)               # (B, Q, hidden_dim)
        x = self.out_proj(x)             # (B, Q, d_model)
        return x


class DetectionHead(nn.Module):
    """
    This detection head uses a simple DynamicConv, then outputs classification and bounding box.
    """
    def __init__(self, d_model=256, num_classes=81):
        super().__init__()
        self.dynamic_conv = DynamicConv(d_model, hidden_dim=d_model*4, num_layers=2)
        self.class_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc  = nn.Linear(d_model, 4)

    def forward(self, hs: torch.Tensor):
        """
        hs shape: (num_decoder_layers, B, Q, d_model)
        We only take the last layer for final detection,
        but we could also build auxiliary outputs from earlier layers if needed.
        """
        out = hs[-1]  # shape (B, Q, d_model)
        out = self.dynamic_conv(out)  # (B, Q, d_model)
        logits = self.class_fc(out)   # (B, Q, num_classes)
        boxes  = self.bbox_fc(out).sigmoid()  # (B, Q, 4) in [0,1]
        return logits, boxes


# --------------------------------------------------------
# 4) Putting it all together: "RTDETRv2"
# --------------------------------------------------------
class RTDETRv2(nn.Module):
    """
    A more complete example with:
    - multi-scale backbone
    - Deformable transformer
    - dynamic conv head
    - optional aux loss
    """
    def __init__(
        self,
        num_classes=81,
        num_queries=300,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        num_feature_levels=4,
        enc_n_points=4,
        dec_n_points=4,
        aux_loss=False
    ):
        super().__init__()
        self.backbone = LightweightBackboneFPN(out_channels=d_model)
        self.transformer = DeformableTransformer(
            d_model, nhead,
            num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout,
            num_feature_levels,
            enc_n_points, dec_n_points,
            num_queries,
            aux_loss=aux_loss
        )
        self.detection_head = DetectionHead(d_model, num_classes)
        self.aux_loss = aux_loss

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W)
        returns:
          pred_logits: (B, Q, num_classes)
          pred_boxes:  (B, Q, 4)
        """
        # 1) Multi-scale features
        srcs = self.backbone(x)  # e.g. list of 3 or 4 feature maps
        # 2) Deformable Transformer => (num_decoder_layers, B, Q, d_model)
        hs_stack = self.transformer(srcs)
        # 3) Detection Head
        pred_logits, pred_boxes = self.detection_head(hs_stack)
        return pred_logits, pred_boxes


# --------------------------------------------------------
# 5) NMS post-processing
# --------------------------------------------------------
import torchvision.ops as ops
def postprocess_nms(pred_logits, pred_boxes, score_thresh=0.5, iou_thresh=0.5):
    """
    Applies score threshold + NMS on the final predictions.
    pred_logits: (B, Q, num_classes)
    pred_boxes:  (B, Q, 4)
    Return: a list of dicts: [{boxes, scores, labels}, ...]
    """
    B, Q, C = pred_logits.shape
    results = []
    for b_idx in range(B):
        # softmax
        scores_all = pred_logits[b_idx].softmax(dim=-1)  # (Q, C)
        boxes_all  = pred_boxes[b_idx]                   # (Q, 4)
        labels_all = scores_all.argmax(dim=-1)           # (Q,)

        # remove background
        keep_bg = (labels_all > 0)
        labels  = labels_all[keep_bg]
        boxes   = boxes_all[keep_bg]
        scores_ = scores_all[keep_bg, labels]

        # filter by score
        keep = (scores_ >= score_thresh)
        labels = labels[keep]
        boxes  = boxes[keep]
        scores_= scores_[keep]

        # NMS
        if len(labels) > 0:
            kept_idx = ops.nms(boxes, scores_, iou_thresh)
            final_boxes = boxes[kept_idx]
            final_scores= scores_[kept_idx]
            final_labels= labels[kept_idx]
        else:
            final_boxes = torch.empty((0,4), device=boxes.device)
            final_scores= torch.empty((0,), device=boxes.device)
            final_labels= torch.empty((0,), device=boxes.device, dtype=torch.long)

        results.append({
            "boxes":  final_boxes,
            "scores": final_scores,
            "labels": final_labels
        })
    return results


# --------------------------------------------------------
# 6) Hungarian Matcher, Loss classes
# --------------------------------------------------------
def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def generalized_iou(boxes1, boxes2):
    """
    boxes1, boxes2: (N,4) in cxcywh
    """
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    inter_x1 = torch.max(boxes1[:,None,0], boxes2[:,0])
    inter_y1 = torch.max(boxes1[:,None,1], boxes2[:,1])
    inter_x2 = torch.min(boxes1[:,None,2], boxes2[:,2])
    inter_y2 = torch.min(boxes1[:,None,3], boxes2[:,3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    union = area1[:,None] + area2 - inter_area
    iou   = inter_area / (union + 1e-6)

    enclose_x1 = torch.min(boxes1[:,None,0], boxes2[:,0])
    enclose_y1 = torch.min(boxes1[:,None,1], boxes2[:,1])
    enclose_x2 = torch.max(boxes1[:,None,2], boxes2[:,2])
    enclose_y2 = torch.max(boxes1[:,None,3], boxes2[:,3])
    enclose_w  = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h  = (enclose_y2 - enclose_y1).clamp(min=0)
    enclose_area = enclose_w * enclose_h + 1e-6

    giou = iou - (enclose_area - union) / enclose_area
    return giou

class HungarianMatcher(nn.Module):
    """Simple Hungarian Matcher for one-to-one label assignment."""
    def __init__(self, class_weight=1.0, bbox_weight=5.0, giou_weight=2.0):
        super().__init__()
        self.class_weight = class_weight
        self.bbox_weight  = bbox_weight
        self.giou_weight  = giou_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: {"pred_logits": (B,Q,C), "pred_boxes": (B,Q,4)}
        targets: list of dict, each has "labels": (N,), "boxes": (N,4)
        returns: list of (q_idx, t_idx) pairs
        """
        bs, num_queries, num_cls = outputs["pred_logits"].shape
        indices = []
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # (Q, C)
            out_bbox = outputs["pred_boxes"][b]               # (Q, 4)
            tgt_ids  = targets[b]["labels"]                   # (N,)
            tgt_bbox = targets[b]["boxes"]                    # (N,4)
            if tgt_ids.numel() == 0:
                # no target
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue
            # cost
            cost_class = -out_prob[:, tgt_ids] # shape (Q, N)
            cost_bbox  = torch.cdist(out_bbox, tgt_bbox, p=1)  # L1
            cost_giou  = -generalized_iou(out_bbox, tgt_bbox)  # negative

            cost = self.class_weight*cost_class + self.bbox_weight*cost_bbox  + self.giou_weight*cost_giou
            cost = cost.cpu()

            q_idx, t_idx = linear_sum_assignment(cost)
            q_idx = torch.from_numpy(q_idx).to(out_prob.device)
            t_idx = torch.from_numpy(t_idx).to(out_prob.device)
            indices.append((q_idx, t_idx))

        return indices


class DETRLoss(nn.Module):
    """
    Composite loss for DETR-style:
    - Classification (Cross Entropy)
    - Bounding Box (L1)
    - GIoU
    """
    def __init__(
        self,
        num_classes=81,
        matcher=None,
        eos_coef=0.1,
        class_weight=1.0,
        bbox_weight=5.0,
        giou_weight=2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = matcher if matcher else HungarianMatcher()
        self.eos_coef    = eos_coef
        self.class_weight= class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef  # background weighting
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        """
        outputs = { "pred_logits": (B,Q,C), "pred_boxes": (B,Q,4) }
        targets = list of dicts: [ { "labels":(N,), "boxes":(N,4) }, ... ]
        """
        indices = self.matcher(outputs, targets)
        src_logits = outputs["pred_logits"]  # (B, Q, C)
        src_boxes  = outputs["pred_boxes"]   # (B, Q, 4)

        idx = self._get_src_permutation_idx(indices)

        # classification
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.long, device=src_logits.device)
        if len(target_classes_o) > 0:
            target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1,2),  # (B, C, Q)
            target_classes,            # (B, Q)
            weight=self.empty_weight
        )

        # box
        src_boxes_matched = src_boxes[idx]
        tgt_boxes         = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)

        if len(tgt_boxes) == 0:
            loss_bbox = torch.tensor(0.0, device=src_logits.device)
            loss_giou = torch.tensor(0.0, device=src_logits.device)
        else:
            l1 = F.l1_loss(src_boxes_matched, tgt_boxes, reduction='none')
            loss_bbox = l1.sum() / (len(tgt_boxes)+1e-6)
            giou_val  = generalized_iou(src_boxes_matched, tgt_boxes)
            loss_giou = (1 - giou_val).sum() / (len(tgt_boxes)+1e-6)

        total_loss = (self.class_weight*loss_ce + 
                      self.bbox_weight*loss_bbox + 
                      self.giou_weight*loss_giou)

        # print for debugging
        print(f"Loss: CE={loss_ce.item():.4f}, BBox={loss_bbox.item():.4f}, GIoU={loss_giou.item():.4f}")

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_total": total_loss
        }

    def _get_src_permutation_idx(self, indices):
        """
        Flatten selected queries across batch
        """
        batch_idx = []
        src_idx   = []
        for b, (s, _) in enumerate(indices):
            if s.shape[0] == 0:
                continue
            batch_idx.append(torch.full((s.shape[0],), b, dtype=torch.long, device=s.device))
            src_idx.append(s)

        if len(batch_idx) == 0:
            return (torch.empty(0, dtype=torch.long, device=src_idx[0].device),
                    torch.empty(0, dtype=torch.long, device=src_idx[0].device))

        batch_idx = torch.cat(batch_idx)
        src_idx   = torch.cat(src_idx)
        return (batch_idx, src_idx)
