import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function  # 用于梯度反转
import torchvision.transforms as transforms  # 用于增强
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from einops import rearrange
import matplotlib.pyplot as plt
import cv2
import cupy as cp
from matplotlib.colors import Normalize

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
    return torch.sum(log_prob_per_dim, dim=1) # Shape: [B]

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
    
    XX = torch.mean(XX + XX.T - 2 * torch.diag(XX)) / (batch_size * (batch_size - 1) + 1e-8)
    YY = torch.mean(YY + YY.T - 2 * torch.diag(YY)) / (batch_size * (batch_size - 1) + 1e-8)
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
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    
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


# --- 梯度反转 (不变) ---
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

# --- 特征增强 (不变) ---
def feature_augment(x, brightness=0.2, contrast=0.2, affine_deg=10, translate=0.1, scale=(0.9, 1.1), blur_sigma=0.5, erase_p=0.3):
    # ... (您的代码，保持不变) ...
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
    def __init__(self, in_channels, input_size, invariant_dim, domain_dim, num_tasks=3):
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

        H, W = input_size
        
        # --- 1. 共享的特征提取器 (Neck 的第一部分) ---
        self.visual_conv = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.visual_bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.shared_feature_dim = 32 * H * W # (例如 32 * 64 * 64)
        self.num_bins = 20
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
            z_dom_k_raw = self.domain_heads[k](x_vis_flat)      # [B, D]
            z_dom_k = F.normalize(z_dom_k_raw, p=2, dim=1, eps=1e-6)
            # 计算马氏距离
            # D_M(x) = (x - mu).T @ InvCov @ (x - mu)
            
            diff = z_dom_k - mu_k.unsqueeze(0) # [B, D]
            
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
# --- 2. 主干网络 (不变) ---
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
    def get_shared_feats(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
        return feat1, feat2
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
        #print(P3_out.mean())
        return P3_out

# ---------------------------------------------------------------------------
# --- 3. 重构后的主模型 (PaDN) ---
# ---------------------------------------------------------------------------
class PaDN(nn.Module):
    def __init__(self, num_classes, val = True, num_frame=5, num_tasks=3): # <-- 增加了 num_tasks
        super(PaDN, self).__init__()

        self.base = True
        self.num_frame = num_frame
        self.backbone = Feature_Extractor(0.33,0.50)
        self.fusion = Fusion_Module(channels=[128], num_frame=num_frame) 
        self.head = nn.ModuleList(YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [256], act = "silu") for _ in range(num_tasks))
        self.val = val
        self.conv_vl = nn.Sequential(
            BaseConv(128*2,256,3,1), # 128*2 -> 256. feats 输入通道为 256
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
        feats_input_size = (64, 64) 
        
        self.motion = MotionModel(
            in_channels=feats_in_channels,
            input_size=feats_input_size,
            invariant_dim=self.invariant_dim,
            domain_dim=self.domain_dim,
            num_tasks=num_tasks
        )
        
        # (您的其他 m1, m2, task_disc 等模块)
        self.m1 = nn.Sequential(
            BaseConv(16,64,3,1),
            BaseConv(64,128,3,1),
            BaseConv(128,128,1,1))
        self.m2 = nn.Linear(1024,4096)

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


    def forward(self, inputs, descriptions=None, multi_targets=None, relation=None, 
                z_pre_invariant=None, z_pre_domain=None, current_task_id=0): # <-- 增加了参数
        
        feat = []
        outputs = []
        # task_id = 0 # (从函数参数 current_task_id 获取)
        task_id = current_task_id # 使用传入的 task_id

        for i in range(self.num_frame-2,self.num_frame):
            f_feats = self.backbone(inputs[:,:,i,:,:]) 
            feat.append(f_feats)
        B, N, W, H = f_feats.shape
        
        # feats 是 [B, 256, H, W]，将作为 neck 的输入
        feats = self.conv_vl(torch.cat(feat,1)).squeeze(1)

        if self.training and self.base: 
            # --- 1. 生成 Motion Prior (不变) ---
            multi_targets = [mt.cuda() if isinstance(mt, torch.Tensor) else mt for mt in multi_targets]
            motion_prior = generate_motion(multi_targets, inputs,
                                                base_kernel_length=1, length_scale=3,
                                                kernel_width=21, alpha=0.1, sigma=3,
                                                brightness_factor=0.1, motion_threshold=1e-2)
            motion_prior_cpu = [cp.asnumpy(h) if hasattr(h, 'ndim') else h for h in motion_prior]
            motion_prior = torch.tensor(np.stack(motion_prior_cpu)).cuda()

            # --- 2. 重构 VAE 和对齐损失 (替换 self.motion.train_forward) ---
            
            # VAE 参数 (您原来 train_forward 中的默认值)
            alpha=1.0 # latent_consistency_loss
            beta=1.0  # kl_loss
            gamma_align=1.0 # mmd_loss
            gamma_sep=0.1   # covariance_loss
            
            B = feats.size(0)
            

            # --- 视觉分支 (使用 Neck) ---
            # 明确使用 current_task_id
            z_current_invariant, z_current_domain = self.motion(feats, task_id)
            #z_current_domain = F.normalize(z_current_domain_raw, p=2, dim=1, eps=1e-6)
            motion_vis = self.decode_heatmap(z_current_invariant, z_current_domain)
            visual_recon_loss = F.mse_loss(motion_vis, motion_prior)
            
            motion = motion_vis # (用于后续 fusion)
            
            # --- 潜在一致性 (文本与视觉) ---
            #latent_consistency_loss = F.mse_loss(z_current_invariant, z_invariant_text)
            
            # --- 域适应损失 ---
            if z_pre_invariant is not None and z_pre_domain is not None:
                latent_alignment_loss = mmd_rbf_loss(z_current_invariant, z_pre_invariant.detach())
                latent_separation_loss = covariance_loss(z_current_invariant, z_current_domain)
            else:
                # 如果是第一个任务或不需要对齐，损失为0
                latent_alignment_loss = 0.0
                latent_separation_loss = 0.0

            # --- VAE 总损失 ---
            loss_alignment = (
                #text_recon_loss + 
                visual_recon_loss + 
                #beta * kl_loss + 
                #alpha * latent_consistency_loss +
                gamma_align * latent_alignment_loss +
                gamma_sep * latent_separation_loss
            )

            # --- 3. 生成 motion_aug (用于 Fusion) ---

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

        # --- 4. 运动图后处理 (不变) ---
        if motion.dim() == 3:  # [B, H, W]
            motion = motion.unsqueeze(1)  # [B, 1, H, W]
        motion = self.conv_m(motion)  


        # --- 5. Fusion 和 Head (不变) ---
        fused_res = self.fusion(motion, feat[-1])
        
        outputs  = self.head[task_id](fused_res) 
        
        if self.training:
            return outputs, loss_alignment 
        else:
            # 推理时也可以返回预测的 task_id
            return outputs, task_id

            
# ---------------------------------------------------------------------------
# --- 4. 剩余所有辅助模块/函数 (保持不变) ---
# ---------------------------------------------------------------------------

def add_comet_kernel(heatmap, head_center, tail_angle, kernel_length=61, kernel_width=21, 
                     alpha=0.1, sigma=3, brightness_scale=1.0):
    # ... (您的代码，保持不变) ...
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
    # ... (您的代码，保持不变) ...
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
            heatmaps.append(cp.asnumpy(heatmap))
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
        #print(y.shape)
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

# (您文件中的第二个 DBaseConv 定义是多余的，我保留一个)

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
    def __init__(self, channels=[128, 256, 512], num_frame=5):
        super(Fusion_Module, self).__init__()
        self.k_conv = BaseConv(channels[0], channels[0] * 2, 3, 1, 1)
        self.fusion_conv = BaseConv(512, 256, 1, 1, 1)

    def forward(self, motion, k_feat):
        k_feat_trans = self.k_conv(k_feat)
        fused = torch.cat([motion, k_feat_trans], dim=1)         
        fused_out = self.fusion_conv(fused) 
        return [fused_out]


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