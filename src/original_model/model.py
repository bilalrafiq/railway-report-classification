import torch
import torch.nn as nn
from transformers import AutoModel



class GLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff * 2)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        value, gate = self.w1(x).chunk(2, dim=-1)
        return self.w2(value * torch.sigmoid(gate))


class StructuredExpert(nn.Module):
    """论文4.1：Dropout、DSCNN、Pre-Norm Transformer和原始token残差。"""
    def __init__(self, num_count, cat_sizes, cfg):
        super().__init__()
        from model_zoo import FeatureTokenizer
        self.tokenizer = FeatureTokenizer(num_count, cat_sizes, cfg.D_MODEL)
        self.dropout = nn.Dropout(cfg.STRUCT_DROPOUT)
        self.cnn = nn.Sequential(
            nn.Conv1d(cfg.D_MODEL, cfg.D_MODEL, 5, padding=2, groups=cfg.D_MODEL),
            nn.Conv1d(cfg.D_MODEL, cfg.D_MODEL, 1), nn.ReLU())
        self.transformer = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(cfg.D_MODEL, 8, batch_first=True),
                'ffn': GLU(cfg.D_MODEL, cfg.D_MODEL * 2),
                'norm1': nn.LayerNorm(cfg.D_MODEL),
                'norm2': nn.LayerNorm(cfg.D_MODEL)
            }) for _ in range(cfg.TRANSFORMER_LAYERS)
        ])
        self.res_scale = nn.Parameter(torch.full((1,), 0.2))

    def forward(self, nums, cats):
        x_raw = self.tokenizer(nums, cats)
        x = self.cnn(self.dropout(x_raw).transpose(1, 2)).transpose(1, 2)
        for layer in self.transformer:
            z = layer['norm1'](x)
            x = x + self.dropout(layer['attn'](z, z, z, need_weights=False)[0])
            x = x + self.dropout(layer['ffn'](layer['norm2'](x)))
        # 式(7)：alpha乘编码后表示，不是乘原始表示。
        return x_raw.mean(dim=1) + self.res_scale * x.mean(dim=1)

class VectorCalibrationFusion(nn.Module):
    def __init__(self, d_model, gate_bias=-3.0):
        super().__init__()
        self.s2t_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.proj_delta = nn.Linear(d_model, d_model * 2)
        
        self.vector_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2),
            nn.Sigmoid()
        )
        nn.init.constant_(self.vector_gate[-2].bias, gate_bias)
        nn.init.zeros_(self.proj_delta.weight)
        nn.init.zeros_(self.proj_delta.bias)
        self.norm = nn.LayerNorm(d_model * 2)

    def forward(self, h_t_seq, v_s, v_t, mask=None):
        v_anchor = torch.cat([v_s, v_t], dim=-1) # [B, 256]
        
        # 计算交互增量
        v_inter_raw, attn_w = self.s2t_attn(v_s.unsqueeze(1), h_t_seq, h_t_seq,
            key_padding_mask=None if mask is None else ~mask.bool())
        v_delta = self.proj_delta(v_inter_raw.squeeze(1)) # [B, 256]
        
        # 向量门控：256 个维度独立决策
        g = self.vector_gate(v_anchor)
        v_fusion = self.norm(v_anchor + g * v_delta)
        
        return v_fusion, g, attn_w

class RailwayDualExpertModel(nn.Module):
    def __init__(self, num_count, cat_sizes, cfg):
        super().__init__()
        self.struct_expert = StructuredExpert(num_count, cat_sizes, cfg)
        self.plm = AutoModel.from_pretrained(cfg.PLM_PATH, local_files_only=True)
        for name, param in self.plm.named_parameters():
            if not any(f"encoder.layer.{i}." in name for i in [10, 11]): param.requires_grad = False
        self.text_adapter = nn.Linear(768, cfg.D_MODEL)
        self.fusion = VectorCalibrationFusion(cfg.D_MODEL, cfg.GATE_BIAS)
        self.classifier = nn.Sequential(nn.Linear(cfg.D_MODEL * 2, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, nums, cats, ids, mask):
        v_s = self.struct_expert(nums, cats)
        plm_out = self.plm(input_ids=ids, attention_mask=mask).last_hidden_state
        h_t_seq = self.text_adapter(plm_out)
        v_t = h_t_seq[:, 0, :]
        v_f, g_vec, attn_w = self.fusion(h_t_seq, v_s, v_t, mask)
        return self.classifier(v_f), g_vec, {'v_s': v_s, 'v_t': v_t, 'attn_w': attn_w}
