import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function  # 用于梯度反转
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
try:
    import cupy as cp
except ImportError:
    cp = np


def _asnumpy(array):
    return cp.asnumpy(array) if hasattr(cp, "asnumpy") else np.asarray(array)

text_dim = 20*300

# --- 辅助函数：高斯似然度 ---
def log_gaussian_likelihood(x, mu, log_var):
    """
    计算 x 在 N(mu, diag(exp(log_var))) 下的对数似然度。
    """
    # 确保 mu 和 log_var 的 shape 为 [B, D] 以便广播
    mu = mu.unsqueeze(0) if mu.dim() == 1 else mu
    log_var = log_var.unsqueeze(0) if log_var.dim() == 1 else log_var
    
    # 将 log_var 限制在合理范围以提高数值稳定性
    log_var = torch.clamp(log_var, -10, 10)
    
    # log P(x) = sum_i {-0.5 * [log(2*pi) + log_var_i] - 0.5 * [(x_i - mu_i)^2 / exp(log_var_i)]}
    log_prob_per_dim = -0.5 * (torch.log(2 * torch.tensor(torch.pi, device=x.device)) + log_var) \
                       -0.5 * ((x - mu)**2 * torch.exp(-log_var))
                       
    # 将所有维度的 log_prob 相加，得到每个样本的总 log_prob
    return torch.sum(log_prob_per_dim, dim=2) # Shape: [B]

# --- 新增：缺失的损失函数 ---

def mmd_rbf_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算 MMD (Maximum Mean Discrepancy) 损失，使用RBF核。
    """
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    # 确保 batch_size > 1
    if batch_size > 1:
        XX = torch.mean(XX + XX.T - 2 * torch.diag(XX)) / (batch_size * (batch_size - 1) + 1e-8)
        YY = torch.mean(YY + YY.T - 2 * torch.diag(YY)) / (batch_size * (batch_size - 1) + 1e-8)
    else:
        XX = torch.mean(XX)
        YY = torch.mean(YY)
        
    XY = torch.mean(XY + YX)
    
    return XX + YY - 2 * XY

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    
    L2_distance = ((total0 - total1)**2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples + 1e-8) # 避免除以0
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def covariance_loss(z_invariant, z_domain):
    """
    计算协方差损失 (L_independence)，促使 z_invariant 和 z_domain 解耦。
    """
    # 零均值化
    z_inv_mean = z_invariant - z_invariant.mean(dim=0)
    z_dom_mean = z_domain - z_domain.mean(dim=0)
    
    # 计算协方差矩阵的非对角元素 (的平方)
    # 我们只关心 z_inv 和 z_dom 之间的协方差
    cross_cov = torch.mm(z_inv_mean.T, z_dom_mean) / (z_inv_mean.size(0) - 1)
    
    # 最小化协方差矩阵的 L2 范数 (Frobenius 范数)
    loss = torch.sum(cross_cov**2)
    return loss


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

# CORRECTION: 自定义多通道特征增强函数（替换torchvision.transforms）
def feature_augment(x, brightness=0.2, contrast=0.2, affine_deg=10, translate=0.1, scale=(0.9, 1.1), blur_sigma=0.5, erase_p=0.3):
    B, C, H, W = x.shape
    
    # 通道-wise 亮度/对比度抖动（模拟颜色抖动）：随机缩放和偏移每个通道
    if brightness > 0:
        offset = (torch.rand(B, C, 1, 1, device=x.device) * 2 - 1) * brightness
        x = x + offset * x.mean(dim=[2,3], keepdim=True)
    if contrast > 0:
        factor = 1 + (torch.rand(B, C, 1, 1, device=x.device) * 2 - 1) * contrast
        x = x * factor
    
    # 随机仿射变换（旋转、平移、缩放）
    theta = torch.rand(B, device=x.device) * affine_deg * 2 - affine_deg  # 旋转角度
    tx = (torch.rand(B, device=x.device) * 2 - 1) * translate * W
    ty = (torch.rand(B, device=x.device) * 2 - 1) * translate * H
    s = torch.rand(B, device=x.device) * (scale[1] - scale[0]) + scale[0]
    
    affine_matrix = torch.stack([
        s * torch.cos(theta * torch.pi / 180), -s * torch.sin(theta * torch.pi / 180), tx,
        s * torch.sin(theta * torch.pi / 180), s * torch.cos(theta * torch.pi / 180), ty
    ], dim=1).view(B, 2, 3)
    
    grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    
    # CORRECTION: 高斯模糊：使用可分离2D卷积实现（水平 + 垂直），支持多通道
    if blur_sigma > 0:
        kernel_size = 3
        sigma = torch.rand(1, device=x.device).item() * blur_sigma + 0.1
        pos = torch.arange(kernel_size, dtype=torch.float, device=x.device) - (kernel_size - 1) / 2  # 居中
        kernel = torch.exp(-pos**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # 水平模糊：[C, 1, 1, kernel_size]
        weight_h = kernel.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
        x = F.conv2d(x, weight_h, groups=C, padding=(0, kernel_size//2))
        
        # 垂直模糊：[C, 1, kernel_size, 1]
        weight_v = kernel.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)
        x = F.conv2d(x, weight_v, groups=C, padding=(kernel_size//2, 0))
    
    # 随机擦除
    if torch.rand(1) < erase_p:
        erase_h = int(torch.rand(1) * H * 0.3 + 0.1 * H)
        erase_w = int(torch.rand(1) * W * 0.3 + 0.1 * W)
        top = int(torch.rand(1) * (H - erase_h))
        left = int(torch.rand(1) * (W - erase_w))
        x[:, :, top:top+erase_h, left:left+erase_w] = 0  # 或随机值，但0模拟擦除
    
    return x

# ---------------------------------------------------------------------------
# --- 1. 新的增量运动颈部 (替换旧的 MotionModel) ---
# ---------------------------------------------------------------------------
class MotionModel(nn.Module):
    """
    一个增量学习颈部模块。
    
    它接收来自主干网络的特征图 `feats`，并根据 `task_id` 将其路由到
    特定任务的域头，生成 z_invariant 和 z_domain。
    
    它还内部维护一个统计模型，用于在推理时预测任务 ID。
    """
    def __init__(self, in_channels, input_size, invariant_dim, domain_dim, num_tasks=3 , kde_max_samples=1000):
        """
        参数:
        - in_channels (int): 输入特征图 `feats` 的通道数 (例如 256)
        - input_size (tuple): 输入特征图 `feats` 的空间 H, W (例如 (64, 64))
        - invariant_dim (int): 不变潜空间的维度
        - domain_dim (int): 域潜空间的维度
        - num_tasks (int): 增量任务的总数
        """
        super(MotionModel, self).__init__()
        self.invariant_dim = invariant_dim
        self.domain_dim = domain_dim
        self.num_tasks = num_tasks
        self.num_bins = 20
        H, W = input_size
        self.kde_max_samples = kde_max_samples # KDE 使用的样本数
        # --- 1. 共享的特征提取器 (Neck 的第一部分) ---
        # (与 MotionModel.visual_conv* 匹配)
        self.visual_conv = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.visual_bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.shared_feature_dim = 32 * H * W # (例如 32 * 64 * 64)

        # --- 2. 潜空间头 (Neck 的第二部分) ---
        # 共享的不变特征头
        self.visual_fc_invariant = nn.Linear(self.shared_feature_dim, self.invariant_dim)
        
        # 特定于任务的域头 (Multi-Head)
        self.domain_heads = nn.ModuleList(
            [nn.Linear(self.shared_feature_dim, self.domain_dim) for _ in range(num_tasks)]
        )

        # --- 3. 任务分类的统计模型 ---
        for i in range(num_tasks):
            # 注册均值向量 (Mu)
            self.register_buffer(f'task_{i}_mu', torch.zeros(self.domain_dim))
            # 注册逆协方差矩阵 (Precision Matrix)
            self.register_buffer(f'task_{i}_inv_cov', torch.eye(self.domain_dim))
        
        self.register_buffer('tasks_trained', torch.zeros(num_tasks, dtype=torch.bool))


    def _shared_visual_encoder(self, feats):
        """ 内部辅助：运行共享的 conv-bn-dropout-flatten """
        B = feats.size(0)
        # feats 形状: [B, in_channels, H, W]
        x_vis = F.relu(self.visual_bn(self.visual_conv(feats)))
        x_vis_flat = self.dropout(x_vis.view(B, -1)) # 形状: [B, shared_feature_dim]
        return x_vis_flat

    def forward(self, feats, task_id):
        """
        训练时的前向传播。
        由主模型的 `train_forward` 调用。
        """
        # 1. 提取共享特征
        x_vis_flat = self._shared_visual_encoder(feats)
        
        # 2. 计算 z_invariant (共享)
        z_inv = self.visual_fc_invariant(x_vis_flat)
        
        # 3. 计算 z_domain (选择特定的头)
        z_dom = self.domain_heads[task_id](x_vis_flat)
        
        return z_inv, z_dom

    @torch.no_grad()
    def update_task_statistics(self, task_id, task_dataloader, backbone, conv_vl, device):
        """
        在任务 task_id 训练 *完成后*，调用此方法来计算并存储其 z_domain 的 KDE 模型。
        """
        self.eval()
        backbone.eval()
        conv_vl.eval()
        
        z_domains = []
        
        num_frame = 2
        for batch in task_dataloader:
            inputs = batch[0].to(device) # 假设 'inputs'
            
            feat_frames = []
            for i in range(num_frame-2, num_frame):
                f_feats = backbone(inputs[:,:,i,:,:]) 
                feat_frames.append(f_feats)
            feats = conv_vl(torch.cat(feat_frames,1)).squeeze(1)
            
            x_vis_flat = self._shared_visual_encoder(feats)
            z_dom = self.domain_heads[task_id](x_vis_flat)
            z_domains.append(z_dom) # 直接收集在 GPU 上
        
        all_z_domains_raw = torch.cat(z_domains, dim=0).to(device) # [N, D]
        all_z_domains = F.normalize(all_z_domains_raw, p=2, dim=1, eps=1e-6)
        N, D = all_z_domains.shape
        
        if D != self.domain_dim:
             print(f"警告: 域维度不匹配! 模型 {self.domain_dim}, 数据 {D}")
             return

        # --- 替换直方图计算 ---
        
        # 1. 计算均值 (Mu)
        # .detach() 确保不追踪梯度
        mu_k = torch.mean(all_z_domains, dim=0).detach() 

        # 2. 计算协方差矩阵 (Cov)
        # (z - mu).T @ (z - mu) / (N - 1)
        # 注意: torch.cov 期望 [D, N]
        if N > 1:
            cov_k = torch.cov(all_z_domains.T).detach()
            # 添加小的-正则化项 (epsilon) 防止矩阵奇异
            cov_k += torch.eye(D, device=device) * 1e-6 
        else:
            # 如果只有一个样本，无法计算协方差，使用单位矩阵
            cov_k = torch.eye(D, device=device)
            
        # 3. 计算并存储逆协方差矩阵 (Precision Matrix)
        try:
            # inv_cov_k = torch.inverse(cov_k) # torch.inverse 可能不稳定
            inv_cov_k = torch.linalg.pinv(cov_k) # 伪逆 (pinv) 更数值稳定
        except torch.linalg.LinAlgError:
            print(f"警告: 任务 {task_id} 的协方差矩阵是奇异的。使用单位矩阵。")
            inv_cov_k = torch.eye(D, device=device)

        # 4. 存储统计参数
        getattr(self, f'task_{task_id}_mu').data = mu_k
        getattr(self, f'task_{task_id}_inv_cov').data = inv_cov_k
        self.tasks_trained[task_id] = True
        print(f"Updated Gaussian statistics for Task {task_id}.")
        
        # 恢复训练模式
        self.train()
        backbone.train()
        conv_vl.train()



    @torch.no_grad()
    def predict_task_statistically(self, feats):
        self.eval()
        B = feats.size(0)
        x_vis_flat = self._shared_visual_encoder(feats)
        
        # [B, Num_Tasks]
        # 我们存储的是负马氏距离 (与 log-likelihood 成正比)
        log_likelihoods = torch.zeros(B, self.num_tasks, device=feats.device)
        
        for k in range(self.num_tasks):
            if not self.tasks_trained[k]:
                log_likelihoods[:, k] = -torch.inf
                continue
                
            mu_k = getattr(self, f'task_{k}_mu')           # [D]
            inv_cov_k = getattr(self, f'task_{k}_inv_cov') # [D, D]
            
            # 关键：z_dom_k 是通过第 k 个头生成的
            z_dom_k = self.domain_heads[k](x_vis_flat)      # [B, D]
            z_dom_k = F.normalize(z_dom_k, p=2, dim=1, eps=1e-6)
            # 计算马氏距离
            # D_M(x) = (x - mu).T @ InvCov @ (x - mu)
            
            diff = z_dom_k - mu_k.unsqueeze(0) # [B, D]
            #print(z_dom_k.mean(),mu_k.mean())
            # (diff @ inv_cov_k) -> [B, D]
            # (diff @ inv_cov_k) * diff -> [B, D] (逐元素乘)
            # .sum(dim=1) -> [B] (马氏距离的平方)
            
            # 矩阵乘法版本:
            # diff.unsqueeze(1) -> [B, 1, D]
            # inv_cov_k.unsqueeze(0) -> [1, D, D]
            # temp = torch.bmm(diff.unsqueeze(1), inv_cov_k.unsqueeze(0).expand(B, -1, -1)) -> [B, 1, D]
            # dist_sq = torch.bmm(temp, diff.unsqueeze(2)).squeeze() -> [B]
            
            # 更高效的 einsum 版本:
            # 'bi, ij, bj -> b' (i 和 j 是 D)
            temp = torch.einsum('bi,ij->bj', diff, inv_cov_k)
            dist_sq = torch.einsum('bj,bj->b', temp, diff) # [B]
            
            # 对数似然度 ~ -0.5 * dist_sq 
            # (我们只关心排序，所以 -dist_sq 即可)
            log_likelihoods[:, k] = -dist_sq
            
        #print(log_likelihoods)
        predicted_task_ids = torch.argmax(log_likelihoods, dim=1)
        return predicted_task_ids, log_likelihoods

    @torch.no_grad()
    def forward_inference(self, feats):
        """
        推理时的前向传播。
        由主模型的 `inference_forward` 调用。
        
        它会自动预测 task_id，并返回用于解码的 z_inv 和 z_dom。
        """
        self.eval()
        B = feats.size(0)
        
        # 1. 预测任务 ID
        predicted_task_ids, _ = self.predict_task_statistically(feats) # Shape: [B]
        
        # 2. 运行共享编码器
        x_vis_flat = self._shared_visual_encoder(feats)
        
        # 3. 编码不变特征
        z_invariant = self.visual_fc_invariant(x_vis_flat)
        
        # 4. 根据预测的 ID，使用对应的 domain_head 编码域特征 (Hard Gating)
        z_domain = torch.zeros(B, self.domain_dim, device=feats.device)
        for k in range(self.num_tasks):
            mask = (predicted_task_ids == k)
            if mask.any():
                z_domain[mask] = self.domain_heads[k](x_vis_flat[mask])
        
        # 5. 返回潜变量和预测的任务 ID
        return z_invariant, z_domain, predicted_task_ids

# ---------------------------------------------------------------------------
# --- 2. GAT (不变) ---
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """
    Simplified version
    """
    def __init__(self, in_dim, out_dim, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)
        
    def forward(self, h, adj):
        B, N, _ = h.size()
        Wh = self.W(h) 
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1) 
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1) 
        Wh_cat = torch.cat([Wh_i, Wh_j], dim=-1)     
        e = torch.matmul(Wh_cat, self.a).squeeze(-1)  
        e = self.leaky_relu(e)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)  
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, alpha=0.2):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, out_dim, alpha=alpha) 
            for _ in range(num_heads)
        ])
        
    def forward(self, h, adj):
        out = [head(h, adj) for head in self.heads] 
        out = torch.cat(out, dim=-1) 
        return out

class GATNet(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=128, out_dim=256, 
                 num_heads_1=8, num_heads_2=8, alpha=0.2):
        super(GATNet, self).__init__()
        self.gat1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads=num_heads_1, alpha=alpha)
        self.gat2 = MultiHeadGATLayer(hidden_dim * num_heads_1, out_dim, num_heads=num_heads_2, alpha=alpha)
        
    def forward(self, x, adj):
        x = self.gat1(x, adj)  
        x = self.gat2(x, adj)  
        return x

# ---------------------------------------------------------------------------
# --- 3. 主干网络 (不变) ---
# ---------------------------------------------------------------------------
class Feature_Extractor(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features
        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
        P5          = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)
        P4          = self.reduce_conv1(P5_upsample) 
        P4_upsample = self.upsample(P4) 
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        P3_out      = self.C3_p3(P4_upsample)  
        
        return P3_out

# ---------------------------------------------------------------------------
# --- 4. 重构后的主模型 (PaDN) ---
# ---------------------------------------------------------------------------


class PaDN(nn.Module):
    def __init__(self, num_classes,val = True, num_frame=5, num_tasks=3): # <-- 增加了 num_tasks
        super(PaDN, self).__init__()
        self.val = val
        self.base = True
        self.num_frame = num_frame
        self.num_tasks = num_tasks # <-- 保存
        self.backbone = Feature_Extractor(0.33,0.50)
        self.fusion = Fusion_Module(channels=[128], num_frame=num_frame)
        self.head = nn.ModuleList(YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [256], act = "silu") for _ in range(num_tasks))
        self.conv_vl = nn.Sequential(
            BaseConv(128*2,256,3,1),
            BaseConv(256,256,3,1),
            BaseConv(256,256,1,1))
        self.conv_m = nn.Sequential(
            BaseConv(1,64,3,2),
            BaseConv(64,128,3,2),
            BaseConv(128,256,3,2),
            BaseConv(256,256,1,1))

        # --- VAE/Motion 组件 (从旧 MotionModel 移入) ---
        latent_dim = 128
        hidden_dim = 1024 # 您原来 MotionModel __init__ 中的 hidden_dim
        domain_dim = 64
        self.latent_dim = latent_dim
        self.invariant_dim = latent_dim - domain_dim
        self.domain_dim = domain_dim

        # 文本调节器 
        self.text_fc1 = nn.Linear(text_dim, hidden_dim)
        self.text_fc_mu = nn.Linear(hidden_dim, self.invariant_dim)
        self.text_fc_logvar = nn.Linear(hidden_dim, self.invariant_dim)
        self.text_fc_domain = nn.Linear(hidden_dim, self.domain_dim)
                
        # 解码器
        self.heatmap_fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=4, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=4, padding=0)
        self.bn3 = nn.BatchNorm2d(1)

        # --- 替换 self.motion 为 self.motion ---
        # feats [B, 256, H, W], 假设 H, W = 64, 64 (基于旧 MotionModel 的 32*64*64)
        feats_in_channels = 256 
        feats_input_size = (64, 64) # 您需要确认 P3_out 的 H, W 是否为 64x64
        
        self.motion = MotionModel(
            in_channels=feats_in_channels,
            input_size=feats_input_size,
            invariant_dim=self.invariant_dim,
            domain_dim=self.domain_dim,
            num_tasks=num_tasks
        )
        
        # (您的其他 m1, m2 模块)
        self.m1 = nn.Sequential(
            BaseConv(16,64,3,1),
            BaseConv(64,128,3,1),
            BaseConv(128,128,1,1))
        self.m2 = nn.Linear(1024,4096)

    def initialize_new_task_modules(self, current_task_id):
        """
        在开始训练新任务 *之前*，将所有任务特定的模块权重
        从上一个任务 (current_task_id - 1) 复制到当前任务。
        """
        if current_task_id > 0 and current_task_id < self.num_tasks:
            print(f"--- Initializing Task {current_task_id} Modules from Task {current_task_id - 1} ---")
            
            # 1. 复制 YOLOXHead

            prev_head_state = self.head[current_task_id - 1].state_dict()
            self.head[current_task_id].load_state_dict(prev_head_state)
            
            # 2. 复制 Fusion_Module
            prev_fusion_state = self.fusion.lora_pre[current_task_id - 1].state_dict()
            self.fusion.lora_pre[current_task_id].load_state_dict(prev_fusion_state)
            
            #prev_motion_state = self.motion.domain_heads[current_task_id - 1].state_dict()
            #self.motion.domain_heads[current_task_id].load_state_dict(prev_motion_state)
            
            prev_dilate_state = self.fusion.dynamic_dilate[current_task_id - 1].state_dict()
            self.fusion.dynamic_dilate[current_task_id].load_state_dict(prev_dilate_state)
            print(f"--- Module Initialization for Task {current_task_id} Complete ---")
        
        elif current_task_id == 0:
            print("Task 0: Using randomly initialized modules.")
        
        else:
             print(f"Warning: Invalid task_id {current_task_id} for initialization.")
    # --- 新增：VAE 辅助函数 ---
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def text_encode(self, descriptions_flat):
        """ 辅助: 运行文本编码器 """
        B = descriptions_flat.size(0)
        h_text = F.relu(self.text_fc1(descriptions_flat))
        mu = self.text_fc_mu(h_text)
        logvar = self.text_fc_logvar(h_text)
        
        z_invariant = self.reparameterize(mu, logvar)
        z_domain_text = self.text_fc_domain(h_text)
        
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar) / (B * mu.size(1))
        
        return z_invariant, z_domain_text, kl_loss

    def decode_heatmap(self, z_invariant, z_domain):
        """ 辅助: 运行解码器 """
        z = torch.cat([z_invariant, z_domain], dim=1)
        B = z.size(0)
        x = F.relu(self.heatmap_fc(z))
        x = x.view(B, 256, 8, 8)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.bn3(self.deconv3(x)))
        return x.squeeze(1)

    def _calculate_inter_task_separation_loss(self, current_z_domain, current_task_id, motion_model, margin=1.0):
        """
        计算任务间域分离损失 (Hinge Loss)。
        
        参数:
        - current_z_domain (Tensor): 当前任务的 z_domain, shape [B, D]
        - current_task_id (int): 当前任务的 ID
        - motion_model (nn.Module): self.motion 模块
        - margin (float): 期望的分离边界的平方 (因为我们用平方距离)
        
        返回:
        - loss_inter_task_sep (Tensor): 标量损失
        """
        # 只有在训练时且不是第一个任务时才计算
        if not (self.training and current_task_id > 0):
            return 0.0

        batch_losses = []
        # 遍历所有已经训练过的前序任务
        for prev_task_id in range(current_task_id):
            if motion_model.tasks_trained[prev_task_id]:
                
                # 1. 获取前一个任务的分布中心 (均值)
                # .detach() 是必须的，我们不希望梯度流向前序任务的统计模型
                prev_center = getattr(motion_model, f'task_{prev_task_id}_mu').detach() # Shape: [domain_dim]
                
                # 2. 计算当前批次 z_domain 到该中心的 L2 距离的平方
                # (current_z_domain - prev_center) -> [B, D]
                dist_sq = torch.sum((current_z_domain - prev_center.unsqueeze(0))**2, dim=1) # Shape: [B]
                
                # 3. Hinge Loss: 惩罚靠得太近的样本
                # loss = max(0, margin - distance^2)
                # 我们希望 distance^2 > margin，所以如果 distance^2 达不到 margin，就产生损失
                loss_per_task = torch.mean(torch.clamp(margin - dist_sq, min=0.0))
                batch_losses.append(loss_per_task)

        if not batch_losses:
            return 0.0
        
        # 将所有先前任务的损失相加（或平均）
        loss_inter_task_sep = torch.sum(torch.stack(batch_losses))
        return loss_inter_task_sep

    def _calculate_head_orthogonalization_loss(self, motion_model):
            """
            [方案二] 计算 MotionModel 中 domain_heads 的权重正交化损失。
            
            参数:
            - motion_model (nn.Module): self.motion 模块
            
            返回:
            - loss_ortho (Tensor): 标量损失
            """
            # 只有一个任务时不需要正交
            if not (self.training and motion_model.num_tasks > 1):
                return 0.0
                
            # 1. 收集所有 domain_heads 的权重
            # 形状: [num_tasks, domain_dim, shared_feature_dim]
            weights = torch.stack([head.weight for head in motion_model.domain_heads])
            num_heads = weights.size(0)
            
            # 2. 展平权重以便计算
            weights_flat = weights.view(num_heads, -1) # [num_tasks, D_out * D_in]
            
            # 3. 归一化 (L2-norm)
            weights_norm = F.normalize(weights_flat, p=2, dim=1)
            
            # 4. 计算两两之间的余弦相似度 (W @ W.T)
            cosine_sim = torch.mm(weights_norm, weights_norm.t())
            
            # 5. 我们希望余弦相似度矩阵接近单位矩阵 I
            # 目标是最小化 (CosineSim - I) 的 L2 范数 (Frobenius Norm)
            I = torch.eye(num_heads, device=cosine_sim.device)
            loss_ortho = torch.sum((cosine_sim - I)**2)
            
            return loss_ortho

    def forward(self, inputs, current_task_id=1): # <-- 增加了参数
        feat = []
        outputs = []
        task_id = current_task_id # 使用传入的 task_id

        for i in range(self.num_frame-2,self.num_frame):
            #print(inputs.shape)
            f_feats = self.backbone(inputs[:,:,i,:,:]) 
            feat.append(f_feats)
        B, N, W, H = f_feats.shape
        feats = self.conv_vl(torch.cat(feat,1)).squeeze(1)

        if self.training and self.base:           
            # --- 2. 重构 VAE 和对齐损失 (替换 self.motion.train_forward) ---
            
            # VAE 参数 (您原来 train_forward 中的默认值)
            alpha=1.0 # latent_consistency_loss
            beta=1.0  # kl_loss
            gamma_align=1.0 # mmd_loss
            gamma_sep=0.1   # covariance_loss
            delta_sep=0.5   # [新] 方案一 权重
            lambda_ortho=0.1 # [新] 方案二 权重      
                  
            B = feats.size(0)

            
            # --- 文本分支 ---
            #z_invariant_text, z_domain_text, kl_loss = self.text_encode(descriptions[:,-1,:,:].view(B, -1))
            #motion_text = self.decode_heatmap(z_invariant_text, z_domain_text)
            #text_recon_loss = F.mse_loss(motion_text, motion_prior)

            # --- 视觉分支 (使用 Neck) ---
            # 明确使用 current_task_id
            z_current_invariant, z_current_domain = self.motion(feats, task_id)
            motion_vis = self.decode_heatmap(z_current_invariant, z_current_domain)
            
            motion = motion_vis # (用于后续 fusion)
            
            # --- 潜在一致性 (文本与视觉) ---
            #latent_consistency_loss = F.mse_loss(z_current_invariant, z_invariant_text)
            
            # --- 域适应损失 ---
            z_pre_invariant, z_pre_domain = self.motion(feats, 0)
            if z_pre_invariant is not None and z_pre_domain is not None:
                latent_alignment_loss = F.relu(mmd_rbf_loss(z_current_invariant, z_pre_invariant.detach()))
                latent_separation_loss = covariance_loss(z_pre_domain, z_current_domain)
            else:
                # 如果是第一个任务或不需要对齐，损失为0
                latent_alignment_loss = 0.0
                latent_separation_loss = 0.0
            # --- [新] 方案一: 任务间域分离损失 (单行调用) ---
            loss_inter_task_sep = self._calculate_inter_task_separation_loss(
                z_current_domain, task_id, self.motion, margin=1.0
            )
            loss_ortho = self._calculate_head_orthogonalization_loss(self.motion)
            # --- VAE 总损失 ---
            loss_alignment = (
                #text_recon_loss + 
                #visual_recon_loss + 
                #beta * kl_loss + 
                #alpha * latent_consistency_loss +
                gamma_align * latent_alignment_loss +
                gamma_sep * latent_separation_loss +
                delta_sep * loss_inter_task_sep
            )

            # --- 3. 生成 motion_aug (用于 Fusion) ---
            with torch.no_grad():
                # (这个逻辑已移到文本分支)
                motion_aug = self.decode_heatmap(z_pre_invariant, z_pre_domain).detach()

        else:
            # --- 推理 ---
            # 1. 使用 Neck 进行推理 (自动预测 task_id)
            z_inv, z_dom, task_id = self.motion.forward_inference(feats)
            if self.val:
                task_id = current_task_id
            # 2. 解码
            motion = self.decode_heatmap(z_inv, z_dom)
            
            loss_alignment = 0
            motion_aug = None  # 推理时无需增强

        if motion.dim() == 3:  # [B, H, W]
            motion = motion.unsqueeze(1)  # [B, 1, H, W]
        motion = self.conv_m(motion)  

        if self.training and motion_aug is not None:
            if motion_aug.dim() == 3:  # [B, H, W]
                motion_aug = motion_aug.unsqueeze(1)  # [B, 1, H, W]

            motion_aug = self.conv_m(motion_aug)  # [B, 256, H, W]
        
        feat_last = feat[-1]  # [B, C, H, W]，如C=128  
        
        fused_res, fusion_loss = self.fusion.train_forward(motion, feat_last, task_id, motion_aug=motion_aug, 
                                                      alpha=0.1, gamma=0.1) if self.training else (self.fusion(motion, feat_last,task_id), 0)
        
        outputs  = self.head[task_id](fused_res) 
        
        if self.training:
            return outputs, loss_alignment + fusion_loss + lambda_ortho * loss_ortho
        else:
            return outputs, task_id # 推理时也返回 task_id

            

def add_comet_kernel(heatmap, head_center, tail_angle, kernel_length=61, kernel_width=21, 
                     alpha=0.1, sigma=3, brightness_scale=1.0):
    half_width = (kernel_width - 1) / 2.0
    u = cp.arange(kernel_length).reshape(kernel_length, 1)  
    v = cp.arange(kernel_width).reshape(1, kernel_width) - half_width 
    weight = brightness_scale * cp.exp(-alpha * u) * cp.exp(-(v**2) / (2 * sigma**2))
    cos_angle = cp.cos(tail_angle)
    sin_angle = cp.sin(tail_angle)
    dx = u * cos_angle - v * sin_angle
    dy = u * sin_angle + v * cos_angle
    x = cp.rint(head_center[0] + dx).astype(cp.int32)
    y = cp.rint(head_center[1] + dy).astype(cp.int32)
    H, W = heatmap.shape
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    cp.add.at(heatmap, (y[mask], x[mask]), weight[mask])
    
    return heatmap

def generate_motion(multi_targets, inputs, base_kernel_length=1, length_scale=3, 
                            kernel_width=21, alpha=0.1, sigma=3, brightness_factor=0.1, motion_threshold=0.1):
    batch_size = len(multi_targets)
    image_h, image_w = 512, 512
    heatmaps = []
    num_frames = 5

    for i in range(batch_size):
        heatmap = cp.zeros((image_h, image_w), dtype=cp.float32)
        boxes_frames = multi_targets[i] 
        first_targets = boxes_frames[0]
        last_targets = boxes_frames[-1]
        if first_targets.shape[0] == 0:
            heatmaps.append(_asnumpy(heatmap))
            continue

        num_targets = first_targets.shape[0]
        for t in range(num_targets):
            initial = first_targets[t]
            initial_conv = initial 
            if last_targets.shape[0] == 0:
                final_conv = initial_conv
            else:
                if last_targets.shape[0] == num_targets:
                    final = last_targets[t]
                    final_conv = final
                else:
                    centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in last_targets])
                    initial_center = initial_conv[0:2]
                    distances = np.linalg.norm(centers - initial_center, axis=1)
                    min_index = np.argmin(distances)
                    final = last_targets[min_index]
                    final_conv = final
            
            x0, y0 = initial_conv[0], initial_conv[1]
            xf, yf = final_conv[0], final_conv[1]
            dx = xf - x0
            dy = yf - y0
            displacement = np.sqrt(dx**2 + dy**2)
            motion_angle = np.arctan2(dy, dx)
            
            if displacement < motion_threshold:
                x_head, y_head = int(round(xf)), int(round(yf))
                if 0 <= x_head < image_w and 0 <= y_head < image_h:
                    heatmap[y_head, x_head] += 1.0
            else:
                dynamic_kernel_length = int(max(1, base_kernel_length + length_scale * displacement))
                brightness_scale = 1 + brightness_factor * displacement
                tail_angle = motion_angle + np.pi  
                head_center = (xf, yf)
                heatmap = add_comet_kernel(heatmap, head_center, tail_angle,
                                           kernel_length=dynamic_kernel_length,
                                           kernel_width=kernel_width, alpha=alpha, sigma=sigma,
                                           brightness_scale=brightness_scale)
        heatmaps.append(heatmap)
    return heatmaps


# --- 剩余所有类 (DBaseConv, CoordinateAttention, LoRA, SimAM, ODConv2d, GhostModule, Fusion_Module, YOLOXHead) ---
# ... (将您文件中所有剩余的类定义粘贴到此处，保持不变) ...

class DBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False):
        super(DBaseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# Define CoordinateAttention (as suggested optimization)
class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))  # 水平池化
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))  # 垂直池化
        mid = channels // reduction
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        
        # 水平和垂直池化
        x_h = self.avg_pool_x(x)  # [B, C, H, 1]
        x_w = self.avg_pool_y(x)  # [B, C, 1, W]
        
        # 修复：转置 x_w 以匹配 x_h 的形状
        x_w = x_w.permute(0, 1, 3, 2)  # [B, C, W, 1]
        
        # 拼接：沿 dim=2，得到 [B, C, H+W, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        
        # 卷积处理
        y = self.act(self.bn(self.conv1(y)))  # [B, mid, H+W, 1]
        
        # 分割：按 H 和 W 分割
        x_h, x_w = torch.split(y, [H, W], dim=2)  # [B, mid, H, 1], [B, mid, W, 1]
        
        # 生成水平和垂直注意力
        a_h = torch.sigmoid(self.conv_h(x_h))  # [B, C, H, 1]
        a_w = torch.sigmoid(self.conv_w(x_w))  # [B, C, W, 1]
        
        # 修复：转置 a_w 以匹配广播
        a_w = a_w.permute(0, 1, 3, 2)  # [B, C, 1, W]
        
        # 广播相乘
        out = identity * a_h * a_w  # [B, C, H, W]
        return out

# OPTIMIZATION: LoRA适配器（参数高效，用于few-shot fine-tune）
class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super(LoRA, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(in_dim, rank) / in_dim**0.5)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        # x: [B, C, H, W] 或 [B, C]，需适配
        if x.dim() == 4:  # 输入为 [B, C, H, W]
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1).transpose(1, 2).reshape(-1, C)  # [B*H*W, C]
            out = self.alpha * (x_flat @ self.A @ self.B)  # [B*H*W, out_dim]
            out = out.view(B, H*W, self.out_dim).transpose(1, 2).reshape(B, self.out_dim, H, W)  # [B, out_dim, H, W]
        else:  # 输入为 [B, C]
            out = self.alpha * (x @ self.A @ self.B)  # [B, out_dim]
        return out

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda  # 正则化参数，防止除零

    def forward(self, x):
        # x: 输入特征图 [B, C, H, W]
        b, c, h, w = x.size()
        n = h * w - 1  # 空间像素数减1（排除当前像素）

        # 计算 (x - mu)^2，其中 mu 是空间均值
        mu = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        x_minus_mu_square = (x - mu).pow(2)  # [B, C, H, W]

        # 计算方差-like项：sum((x - mu)^2) / n + lambda
        var_like = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda  # [B, C, 1, 1]

        # 计算能量函数 t，并偏移0.5
        y = x_minus_mu_square / (4 * var_like) + 0.5  # [B, C, H, W]

        # Sigmoid激活得到注意力权重，并乘回原特征
        attention = self.activation(y)
        return x * attention


class ODConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True,
                 head_num=4, kernel_num=4, temperature=32.0, reduction_ratio=16):
        super(ODConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.head_num = head_num
        self.kernel_num = kernel_num
        self.temperature = temperature
        self.reduction = max(16, in_channels // reduction_ratio)

        # 注意力FC层
        self.fc = nn.Linear(in_channels, self.reduction)
        self.bn = nn.BatchNorm1d(self.reduction)
        self.att_head = nn.Linear(self.reduction, head_num * kernel_num, bias=False)
        self.att_spatial = nn.Linear(self.reduction, head_num * kernel_num * self.kernel_size ** 2, bias=False)
        self.att_channel = nn.Linear(self.reduction, head_num * kernel_num * (out_channels // head_num), bias=False)
        self.att_filter = nn.Linear(self.reduction, head_num * kernel_num, bias=False)

        # 基础权重
        self.weight = nn.Parameter(torch.randn(head_num * kernel_num, out_channels // head_num, in_channels // groups, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(head_num * kernel_num, out_channels // head_num))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        b, c, h, w = x.shape

        # 全局平均池化 + FC降维
        xa = x.mean(dim=(2, 3))  # [b, c]
        xa = F.relu(self.bn(self.fc(xa)), inplace=True)  # [b, reduction]

        # 计算四个维度的注意力
        att_head = F.softmax(self.att_head(xa) / self.temperature, dim=1).view(b, self.head_num * self.kernel_num, 1, 1, 1, 1)  # [b, head*kernel, 1, 1, 1, 1]
        att_spatial = F.softmax(self.att_spatial(xa) / self.temperature, dim=1).view(b, self.head_num * self.kernel_num, 1, 1, self.kernel_size, self.kernel_size)  # [b, head*kernel, 1, 1, k, k]
        att_channel = F.softmax(self.att_channel(xa) / self.temperature, dim=1).view(b, self.head_num * self.kernel_num, self.out_channels // self.head_num, 1, 1, 1)  # [b, head*kernel, out/head, 1, 1, 1]
        att_filter = F.softmax(self.att_filter(xa) / self.temperature, dim=1).view(b, self.head_num * self.kernel_num, 1, 1, 1, 1)  # [b, head*kernel, 1, 1, 1, 1]

        # 融合注意力
        att = att_head * att_spatial * att_channel * att_filter  # [b, head*kernel, out/head, 1, k, k]
        att = att.expand(-1, -1, -1, self.in_channels // self.groups, -1, -1)  # [b, head*kernel, out/head, in/group, k, k]

        # 聚合权重
        weight = self.weight.unsqueeze(0)  # [1, head*kernel, out/head, in/group, k, k]
        aggregate_weight = (weight * att).sum(dim=1)  # [b, out/head, in/group, k, k]
        # 修复：重塑为完整的 out_channels
        aggregate_weight = aggregate_weight.view(b, self.out_channels // self.head_num, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        aggregate_weight = aggregate_weight.repeat(1, self.head_num, 1, 1, 1)  # [b, out_channels, in/group, k, k]
        aggregate_weight = aggregate_weight.view(b * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            bias = self.bias.unsqueeze(0)  # [1, head*kernel, out/head]
            aggregate_bias = (bias * att_head.squeeze(-1).squeeze(-1).squeeze(-1)).sum(dim=1)  # [b, out/head]
            aggregate_bias = aggregate_bias.repeat(1, self.head_num)  # [b, out_channels]
            aggregate_bias = aggregate_bias.view(b * self.out_channels)
        else:
            aggregate_bias = None

        # 批组卷积
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=self.groups * b)
        out = out.view(b, self.out_channels, out.shape[2], out.shape[3])

        return out


# (第二个 DBaseConv 定义是多余的, 已移除)

# Ghost模块：从GhostNet论文，生成"ghost"特征以减少计算
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, act=nn.ReLU(inplace=True)):
        super(GhostModule, self).__init__()
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            act if act else nn.Identity()
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            act if act else nn.Identity()
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

# GhostDBaseConv：结合GhostModule和DBaseConv，实现带扩张的轻量卷积
class GhostDBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=1, ratio=2, act=nn.ReLU(inplace=True)):
        super(GhostDBaseConv, self).__init__()
        self.ghost = GhostModule(in_channels, out_channels, kernel_size=kernel_size, ratio=ratio, dw_kernel_size=kernel_size, stride=stride, act=act)
        # 应用扩张：GhostModule内部的conv不支持直接dilation，所以后置一个dilation调整（如果需要>1）
        self.dilate_adjust = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, groups=out_channels, bias=False) if dilation > 1 else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act if act else nn.Identity()

    def forward(self, x):
        x = self.ghost(x)
        x = self.dilate_adjust(x)
        return self.act(self.bn(x))

class Fusion_Module(nn.Module):
    def __init__(self, channels=[128, 256, 512], num_frame=5, num_tasks = 3):
        super(Fusion_Module, self).__init__()
        self.k_conv = BaseConv(channels[0], channels[0] * 2, 3, 1, 1)
        self.ca = CoordinateAttention(channels[0] * 4)
        self.simam = SimAM()
        self.fusion_conv = BaseConv(channels[0] * 4, channels[0] * 2, 3, 1, 1)
        mid_channels = channels[0] * 2  # 256
        self.pre_conv = BaseConv(channels[0] * 4, mid_channels, 1, 1)
        self.lora_pre = nn.ModuleList(LoRA(channels[0] * 4, mid_channels, rank=4, alpha=1.0) for _ in range(num_tasks))
        self.upsample_branch = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            BaseConv(mid_channels, mid_channels, 3, 1, 1)
        )
        self.dilated1 = GhostDBaseConv(mid_channels, mid_channels, 3, 1, dilation=1, padding=1)
        self.dilated2 = GhostDBaseConv(mid_channels, mid_channels, 3, 1, dilation=2, padding=2)
        self.dilated3 = GhostDBaseConv(mid_channels, mid_channels, 3, 1, dilation=4, padding=4)
        self.dynamic_dilate = nn.ModuleList(ODConv2d(mid_channels, mid_channels, kernel_size=3, padding=1) for _ in range(num_tasks))
        self.dilated_fuse = BaseConv(mid_channels * 4, mid_channels, 1, 1)
        self.merge_scale = BaseConv(mid_channels * 2, channels[0] * 2, 3, 1)
        self.residual_conv = nn.Sequential(
            BaseConv(channels[0] * 4, channels[0] * 2, 1, 1),
            nn.BatchNorm2d(channels[0] * 2)
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, motion, k_feat, task_id):
        k_feat_trans = self.k_conv(k_feat)
        fused = torch.cat([motion, k_feat_trans], dim=1)
        att = self.ca(fused) + self.simam(fused)
        att = self.dropout(att)
        pre_feat = self.pre_conv(att) + self.lora_pre[task_id](att)
        up_feat = self.upsample_branch(pre_feat)
        d1 = self.dilated1(pre_feat)
        d2 = self.dilated2(pre_feat)
        d3 = self.dilated3(pre_feat)
        d_dynamic = self.dynamic_dilate[task_id](pre_feat)
        dilated_out = torch.cat([d1, d2, d3, d_dynamic], dim=1)
        dilated_out = self.dilated_fuse(dilated_out)
        up_feat_down = F.interpolate(up_feat, size=pre_feat.shape[2:], mode='bilinear', align_corners=False)
        multi_scale_fused = torch.cat([dilated_out, up_feat_down], dim=1)
        multi_scale_fused = self.merge_scale(multi_scale_fused)
        fused_out = self.fusion_conv(att)
        fused_out_final = fused_out + multi_scale_fused
        fused_res = fused_out_final + self.residual_conv(fused)
        return [fused_res]

    def train_forward(self, motion, k_feat,task_id, motion_aug=None, alpha=0.1, gamma=0.1):
        fused_res = self.forward(motion, k_feat,task_id)[0]
        total_loss = 0
        if motion_aug is not None:
            fused_aug = self.forward(motion_aug, k_feat,task_id)[0]
            contrast_loss = F.mse_loss(fused_res.mean(dim=[2,3]), fused_aug.mean(dim=[2,3]).detach())
            def irm_penalty(loss_fn, inputs, targets):
                scale = torch.ones(1, requires_grad=True).to(inputs.device)
                loss = loss_fn(inputs * scale, targets)
                grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
                return torch.sum(grad**2)
            irm_pen = gamma * irm_penalty(F.mse_loss, fused_res, fused_aug) # 你的代码这里是 motion，也许是 fused_res 和 motion_prior？
            total_loss = contrast_loss + alpha * irm_pen
        return [fused_res], total_loss

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        
        outputs = []
        for k, x in enumerate(inputs):
            x       = self.stems[k](x)
            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)
            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)
            obj_output  = self.obj_preds[k](reg_feat)
            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
