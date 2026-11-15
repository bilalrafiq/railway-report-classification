import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    """FT思想：每个特征一个独有的 Token"""
    def __init__(self, num_count, cat_sizes, d_model):
        super().__init__()
        # 数值特征投影层: 每个特征分配一个独立的线性层
        self.num_projs = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_count)])
        # 类别特征 Embedding
        self.cat_embs = nn.ModuleDict({
            name: nn.Embedding(size + 1, d_model) for name, size in cat_sizes.items()
        })

    def forward(self, nums, cats_idx):
        tokens = []
        # 处理数值
        for i in range(nums.size(1)):
            tokens.append(self.num_projs[i](nums[:, i:i+1]).unsqueeze(1))
        # 处理类别 (假定 cats_idx 是按照 cat_sizes 顺序排列的 Tensor)
        for i, name in enumerate(self.cat_embs.keys()):
            tokens.append(self.cat_embs[name](cats_idx[:, i]).unsqueeze(1))
        return torch.cat(tokens, dim=1)
