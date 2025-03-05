import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment

# Lightweigth backbone + FPN
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class LightweightBackboneFPN(nn.Module):
    """
    MobileNet-like + FPN
    """
    def __init__(self, out_channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 16, 3, 2, 1),
            ConvBNAct(16, 16, 3, 1, 1, groups=16),
            ConvBNAct(16, 32, 1, 1, 0),
        )
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

        self.lateral3 = nn.Conv2d(256, out_channels, 1)
        self.lateral2 = nn.Conv2d(128, out_channels, 1)
        self.lateral1 = nn.Conv2d(64, out_channels, 1)

        self.output3 = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        self.output2 = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        self.output1 = ConvBNAct(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        c0 = self.stem(x)
        c1 = self.stage1(c0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)

        f3 = self.lateral3(c3)
        f2 = self.lateral2(c2)
        f1 = self.lateral1(c1)

        # top-down
        f2 = f2 + F.interpolate(f3, size=f2.shape[-2:], mode='nearest')
        f1 = f1 + F.interpolate(f2, size=f1.shape[-2:], mode='nearest')

        p3 = self.output3(f3)
        p2 = self.output2(f2)
        p1 = self.output1(f1)
        return [p1, p2, p3]

# Deformable Transformer
class MSDeformAttn(nn.Module):
    """
    Deformable Attention (데모용 간소화)
    """
    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads*n_levels*n_points*2)
        self.attention_weights = nn.Linear(d_model, n_heads*n_levels*n_points)
        self.value_proj        = nn.Linear(d_model, d_model)
        self.output_proj       = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        # 실제 구현을 단순화 - 노이즈 추가
        attn_output = query + 0.01*torch.randn_like(query)
        attn_output = self.output_proj(attn_output)
        return attn_output

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024,
                 dropout=0.1, n_levels=3, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.dropout1  = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(d_model)

        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout2  = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.dropout3  = nn.Dropout(dropout)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        attn_out = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index)
        src2 = self.dropout1(attn_out)
        src  = self.norm1(src + src2)

        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src2 = self.dropout3(src2)
        src  = self.norm2(src + src2)
        return src

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024,
                 dropout=0.1, n_levels=3, n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2    = nn.LayerNorm(d_model)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3    = nn.LayerNorm(d_model)

    def forward(self, tgt, reference_points, memory,
                tgt_mask, tgt_key_padding_mask,
                memory_spatial_shapes, memory_level_start_index,
                pos, query_pos):
        # self-attn
        q = tgt + query_pos
        q = q.permute(1, 0, 2)
        tgt2, _ = self.self_attn(q, q, q, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt2 = tgt2.permute(1, 0, 2)
        tgt  = tgt + self.dropout1(tgt2)
        tgt  = self.norm1(tgt)

        # cross-attn
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, memory,
                               memory_spatial_shapes, memory_level_start_index)
        tgt  = tgt + self.dropout2(tgt2)
        tgt  = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout3(F.relu(self.linear1(tgt))))
        tgt2 = self.dropout4(tgt2)
        tgt  = self.norm3(tgt + tgt2)
        return tgt

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, pos, reference_points,
                spatial_shapes, level_start_index):
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index)
        return output

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt, reference_points, memory,
                tgt_mask, tgt_key_padding_mask,
                memory_spatial_shapes, memory_level_start_index,
                pos, query_pos):
        output = tgt
        for layer in self.layers:
            output = layer(
                output, reference_points, memory,
                tgt_mask, tgt_key_padding_mask,
                memory_spatial_shapes, memory_level_start_index,
                pos, query_pos
            )
        return output

class DeformableTransformer(nn.Module):
    def __init__(self,
                 d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 num_feature_levels=3,
                 enc_n_points=4, dec_n_points=4,
                 num_queries=100):
        super().__init__()
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            n_levels=num_feature_levels, n_points=enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            n_levels=num_feature_levels, n_points=dec_n_points
        )
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)
        self.query_embed = nn.Embedding(num_queries, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.level_embed)
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.)

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            grid_y = torch.linspace(0.5, H_-0.5, H_, device=device)
            grid_x = torch.linspace(0.5, W_-0.5, W_, device=device)
            gy, gx = torch.meshgrid(grid_y, grid_x)
            gx = gx.reshape(-1)[None] / (W_ * valid_ratios[:, None, lvl])
            gy = gy.reshape(-1)[None] / (H_ * valid_ratios[:, None, lvl])
            reference_points = torch.stack((gx, gy), -1)
            reference_points_list.append(reference_points)
        reference_points = torch.cat(reference_points_list, dim=1)
        reference_points = reference_points[:, :, None, :]
        return reference_points

    def forward(self, srcs, masks=None):
        """
        srcs: FPN 출력: list of 3 scales, 각 scale은 (B, C, H, W) 
        """
        if masks is None:
            masks = [torch.zeros(s.shape[:2], dtype=torch.bool, device=s.device) for s in srcs]

        device = srcs[0].device
        batch_size = srcs[0].shape[0]

        # (H, W) for each scale
        spatial_shapes = []
        for lvl in range(len(srcs)):
            b, c, h, w = srcs[lvl].shape
            spatial_shapes.append((h, w))
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)

        level_start_index = torch.zeros(spatial_shapes.size(0), dtype=torch.long, device=device)
        for i in range(1, spatial_shapes.size(0)):
            level_start_index[i] = level_start_index[i-1] + (
                spatial_shapes[i-1, 0]*spatial_shapes[i-1, 1]
            )

        # flatten
        src_flatten = []
        for lvl, src in enumerate(srcs):
            b, c, h, w = src.shape
            src_ = src.flatten(2).transpose(1, 2)  # (B, HW, C)
            lvl_embed = self.level_embed[lvl].view(1,1,-1).repeat(b, h*w, 1)
            src_ = src_ + lvl_embed
            src_flatten.append(src_)
        src_flatten = torch.cat(src_flatten, 1)  # (B, \sum(HW), C)

        valid_ratios = torch.ones(batch_size, len(srcs), device=device)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device)

        pos_embed = torch.zeros_like(src_flatten)  # 간단히 0
        memory = self.encoder(src_flatten, pos_embed, reference_points, spatial_shapes, level_start_index)

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        tgt = torch.zeros_like(query_embed)
        dec_ref_points = self.reference_points(query_embed).unsqueeze(2)

        hs = self.decoder(
            tgt, dec_ref_points, memory,
            tgt_mask=None, tgt_key_padding_mask=None,
            memory_spatial_shapes=spatial_shapes,
            memory_level_start_index=level_start_index,
            pos=pos_embed,
            query_pos=query_embed
        )
        return hs

# RTDETR (backbone + Transformer + head)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class RTDETR(nn.Module):
    def __init__(self,
                 num_classes=21,
                 num_queries=100,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 num_feature_levels=3,
                 enc_n_points=4,
                 dec_n_points=4):
        super().__init__()
        self.backbone = LightweightBackboneFPN(out_channels=d_model)
        self.transformer = DeformableTransformer(
            d_model, nhead,
            num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout,
            num_feature_levels,
            enc_n_points, dec_n_points,
            num_queries
        )
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed  = MLP(d_model, d_model, 4, 3)

    def forward(self, x):
        # 1) FPN
        srcs = self.backbone(x)  
        # 2) Deformable Transformer
        hs   = self.transformer(srcs)  # (num_decoder_layers, B, Q, d_model)
        # 3) 마지막 디코더 레이어만 사용 → (B, Q, d_model)
        out  = hs
        # 4) Detection heads
        pred_logits = self.class_embed(out)          # (B, Q, num_classes)
        pred_boxes  = self.bbox_embed(out).sigmoid() # (B, Q, 4)
        return pred_logits, pred_boxes


# Hungarian Matcher + Loss (GIoU ...)
def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return torch.stack([x1, y1, x2, y2], -1)

def generalized_iou(boxes1, boxes2):
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    inter_x1 = torch.max(boxes1[:,None,0], boxes2[:,0])
    inter_y1 = torch.max(boxes1[:,None,1], boxes2[:,1])
    inter_x2 = torch.min(boxes1[:,None,2], boxes2[:,2])
    inter_y2 = torch.min(boxes1[:,None,3], boxes2[:,3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = ((boxes1[:,2]-boxes1[:,0]).clamp(min=0) *
             (boxes1[:,3]-boxes1[:,1]).clamp(min=0))
    area2 = ((boxes2[:,2]-boxes2[:,0]).clamp(min=0) *
             (boxes2[:,3]-boxes2[:,1]).clamp(min=0))
    union = area1[:,None] + area2 - inter_area
    iou   = inter_area / union

    enclose_x1 = torch.min(boxes1[:,None,0], boxes2[:,0])
    enclose_y1 = torch.min(boxes1[:,None,1], boxes2[:,1])
    enclose_x2 = torch.max(boxes1[:,None,2], boxes2[:,2])
    enclose_y2 = torch.max(boxes1[:,None,3], boxes2[:,3])
    enclose_w  = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h  = (enclose_y2 - enclose_y1).clamp(min=0)
    enclose_area = enclose_w * enclose_h

    giou = iou - (enclose_area - union)/enclose_area
    return giou

class HungarianMatcher(nn.Module):
    def __init__(self, class_weight=1.0, bbox_weight=5.0, giou_weight=2.0):
        super().__init__()
        self.class_weight = class_weight
        self.bbox_weight  = bbox_weight
        self.giou_weight  = giou_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        # (B, Q, num_classes)
        bs, num_queries, num_cls = outputs["pred_logits"].shape

        indices = []
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # (Q, num_cls)
            out_bbox = outputs["pred_boxes"][b]               # (Q, 4)
            tgt_ids  = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            cost_class = -out_prob[:, tgt_ids]
            cost_bbox  = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou  = -generalized_iou(out_bbox, tgt_bbox)

            cost = (self.class_weight*cost_class +
                    self.bbox_weight*cost_bbox +
                    self.giou_weight*cost_giou)
            cost = cost.cpu()

            q_idx, t_idx = linear_sum_assignment(cost)
            q_idx = torch.from_numpy(q_idx).to(out_prob.device)
            t_idx = torch.from_numpy(t_idx).to(out_prob.device)
            indices.append((q_idx, t_idx))

        return indices

class DETRLoss(nn.Module):
    """
    1) Class loss (CrossEntropy)
    2) BBox L1 loss
    3) GIoU loss
    """
    def __init__(self, num_classes=21,
                 matcher=None,
                 eos_coef=0.1,
                 class_weight=1.0,
                 bbox_weight=5.0,
                 giou_weight=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher if matcher else HungarianMatcher()
        self.eos_coef = eos_coef
        self.class_weight = class_weight
        self.bbox_weight  = bbox_weight
        self.giou_weight  = giou_weight

        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef  # background 가중치
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        """
        outputs: {"pred_logits": (B, Q, num_classes),
                  "pred_boxes":  (B, Q, 4)}
        """
        indices = self.matcher(outputs, targets)
        src_logits = outputs["pred_logits"]  # (B, Q, num_classes)
        src_boxes  = outputs["pred_boxes"]   # (B, Q, 4)

        idx = self._get_src_permutation_idx(indices)

        # 클래스 타겟 만들기
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],  # (B, Q)
            0,
            dtype=torch.long,
            device=src_logits.device
        )
        if len(target_classes_o) > 0:
            target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),  # (B, num_classes, Q)
            target_classes,              # (B, Q)
            weight=self.empty_weight
        )

        # 박스 타겟
        src_boxes_matched = src_boxes[idx]
        target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)

        if len(target_boxes) == 0:
            loss_bbox = torch.tensor(0.0, device=src_logits.device)
            loss_giou = torch.tensor(0.0, device=src_logits.device)
        else:
            loss_bbox = F.l1_loss(src_boxes_matched, target_boxes, reduction='none')
            loss_bbox = loss_bbox.sum() / (len(target_boxes)+1e-6)

            giou = generalized_iou(src_boxes_matched, target_boxes)
            loss_giou = (1 - giou).sum() / (len(target_boxes)+1e-6)

        total_loss = (self.class_weight*loss_ce +
                      self.bbox_weight*loss_bbox +
                      self.giou_weight*loss_giou)

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_total": total_loss
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = []
        src_idx = []
        for b, (s, _) in enumerate(indices):
            if s.shape[0] == 0:
                continue
            batch_idx.append(
                torch.full((s.shape[0],), b, dtype=torch.long, device=s.device)
            )
 
            src_idx.append(s)

        if len(batch_idx) == 0:
            return (torch.empty(0, dtype=torch.long, device=s.device),
                    torch.empty(0, dtype=torch.long, device=s.device))

        batch_idx = torch.cat(batch_idx)
        src_idx   = torch.cat(src_idx)
        return (batch_idx, src_idx)
