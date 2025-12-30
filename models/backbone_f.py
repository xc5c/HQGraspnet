
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from all_code.graspnet.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from all_code.graspnet.pointnet2.pointnet2_utils import CylinderQueryAndGroup, cylinder_query
import all_code.graspnet.pointnet2.pytorch_utils as pt_utils
from all_code.graspnet.utils.label_generation_new import process_grasp_labels
# from all_code.graspnet.utils.label_generation import process_grasp_labels

class Local_attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #  x.shape   b*n, c, k
        x_q = self.q_conv(x).permute(0, 2, 1)  # b*n, k, c
        x_k = self.k_conv(x)  # b*n, c, k
        x_v = self.v_conv(x)  # b*n, c, k
        energy = x_q @ x_k  # b*n, k, k
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=2, keepdims=True))
        x_r = x_v @ attention  # b*n, c, k
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.02):
        super().__init__()
        self.nsample = nsample  # 64
        self.in_dim = seed_feature_dim  # 3
        self.cylinder_radius = cylinder_radius
        mlps = [128+3, 256]

        self.groupers = CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        self.local_att = Local_attention(256)

    def forward(self, seed_xyz, pointcloud, vp_rot, up_feature):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors

            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        B, num_seed, _, _ = vp_rot.size()
        grouped_features = self.groupers(
                pointcloud, seed_xyz, vp_rot, features=up_feature)  # (batch_size, feature_dim,  nsample)
        vp_features = self.mlps(
            grouped_features)  # (batch_size, mlps[-1], num_seed, nsample)

        vp_features = vp_features.permute(0, 2, 1, 3).contiguous().view(B * num_seed, 256,
                                                                        self.nsample)  # (B*num_seed*num_depth, C, K)
        vp_features = self.local_att(vp_features).contiguous().view(B, num_seed, 256, self.nsample).permute(
            0, 2, 1, 3)

        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]).squeeze(-1)  # (batch_size, mlps[-1], num_seed)
        return vp_features


class LocalFeatureExtractor(nn.Module):

    def __init__(self, in_channels, out_channels, radius=0.05, hmin=-0.02, hmax=0.02, nsample=32):
        super().__init__()
        self.radius = radius
        self.hmin = hmin
        self.hmax = hmax
        self.nsample = nsample


        self.cloud_crop = CloudCrop(
            nsample=nsample,
            seed_feature_dim=in_channels,
            cylinder_radius=radius,
            hmin=hmin,
            hmax=hmax
        )


        self.adjust_mlp = nn.Sequential(
            nn.Conv1d(256, out_channels, 1),  # CloudCrop输出256维特征
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, features, xyz, seed_xyz, rot=None):

        B, C, N = features.shape
        npoint = seed_xyz.shape[1]

        if xyz.dim() != 3 or xyz.shape[2] != 3:
            xyz = xyz.permute(0, 2, 1) if xyz.shape[1] == 3 else xyz

        if seed_xyz.dim() != 3 or seed_xyz.shape[2] != 3:
            seed_xyz = seed_xyz.permute(0, 2, 1) if seed_xyz.shape[1] == 3 else seed_xyz

        if rot is not None:
            rot = rot.view(B, npoint, 3, 3)
        else:
            ident = torch.eye(3, device=xyz.device).view(1, 1, 3, 3).expand(B, npoint, 3, 3)
            rot = ident.contiguous()

        local_features = self.cloud_crop(
            seed_xyz=seed_xyz.contiguous(),
            pointcloud=xyz.contiguous(),
            vp_rot=rot.contiguous(),
            up_feature=features.contiguous()
        )  # (B, 256, npoint)

        local_features = self.adjust_mlp(local_features)  # (B, out_channels, npoint)

        return local_features


class FusionModule(nn.Module):

    def __init__(self, local_dim=128, global_dim=256, fused_dim=256):
        super().__init__()

        self.proj_local = nn.Conv1d(local_dim, fused_dim, 1, bias=False)
        self.proj_global = nn.Conv1d(global_dim, fused_dim, 1, bias=False)

        self.att_mlp = nn.Sequential(
            nn.Conv1d(fused_dim * 2, fused_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(fused_dim // 4, fused_dim, 1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(fused_dim, fused_dim, 1),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_features, global_features):
        local = self.proj_local(local_features)   # (B, 256, N)
        global_ = self.proj_global(global_features)  # (B, 256, N)

        concat = torch.cat([local, global_], dim=1)  # (B, 512, N)
        attn = self.att_mlp(concat)  # (B, 256, N)

        fused = local * attn + global_ * (1 - attn)  # 可学习混合

        return self.out_conv(fused)

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0,is_training=True, is_demo=False):
        super().__init__()
        self.is_traning = is_training
        self.is_demo = is_demo
        self.fusion_module = FusionModule(local_dim=128, global_dim=256, fused_dim=256)

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.04,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.3,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

        self.local_feature_extractor = LocalFeatureExtractor(256, 128)  # 输入通道改为256，因为使用fp2_features

        self.initial_approach_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1),
            nn.Tanh()
        )

        self.initial_binormal_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1),
            nn.Tanh()
        )

        self.final_approach_head = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
            nn.Tanh()
        )

        self.final_binormal_head = nn.Sequential(
            nn.Conv1d(256+ 3, 64, 1),  # 128是局部特征，3是接近向量
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
            nn.Tanh()
        )

        self.mlp_graspable = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
            nn.Sigmoid()
        )
        self.mlp_offsets = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1),
            nn.ReLU(inplace=True)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,D,K)
                XXX_inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz
        end_points['input_features'] = features

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds

        # fp2_features的形状为 (B, 256, num_seed)
        fp2_features = end_points['fp2_features']

        initial_approach = self.initial_approach_head(fp2_features) + 1e-6
        initial_approach = F.normalize(initial_approach, p=2, dim=1)

        initial_binormal = self.initial_binormal_head(fp2_features) + 1e-6
        initial_binormal = F.normalize(initial_binormal, p=2, dim=1)

        z_vector = torch.cross(initial_approach, initial_binormal, dim=1)
        z_vector = z_vector + 1e-6
        z_vector = F.normalize(z_vector, p=2, dim=1)

        initial_rot = torch.stack([
            initial_approach.transpose(1, 2),  # (B, num_seed, 3)
            initial_binormal.transpose(1, 2),  # (B, num_seed, 3)
            z_vector.transpose(1, 2)           # (B, num_seed, 3)
        ], dim=3)  # (B, num_seed, 3, 3)



        if not self.is_demo:
            end_points = process_grasp_labels(end_points)

        if self.is_traning:
            seed_xyz = end_points['batch_grasp_point']
        else:
            seed_xyz = end_points['fp2_xyz']


        local_features = self.local_feature_extractor(
            end_points['sa1_features'],
            end_points['sa1_xyz'],
            seed_xyz,
            initial_rot
        )
        fused_features = self.fusion_module(local_features, fp2_features)  # (B, 256, N)

        final_approach = self.final_approach_head(fused_features)
        # final_approach = F.normalize(final_approach, p=2, dim=1)

        binormal_input = torch.cat([fused_features, final_approach], dim=1)
        final_binormal = self.final_binormal_head(binormal_input)
        # final_binormal = F.normalize(final_binormal, p=2, dim=1)

        rot_out = torch.cat([final_approach, final_binormal], dim=1)

        graspable_out = self.mlp_graspable(fp2_features)  # (B, 1, num_seed)
        offsets_out = self.mlp_offsets(fp2_features)  # (B, 3, num_seed)

        # end_points['initial_rot'] = initial_rot
        end_points['objectness_pred'] = graspable_out
        end_points['rot_pred'] = rot_out
        end_points['offsets_pred'] = offsets_out
        end_points['initial_approach_pred'] = initial_approach
        end_points['initial_binormal_pred'] = initial_binormal
        end_points['approach_pred'] = final_approach
        end_points['binormal_pred'] = final_binormal
        end_points['local_features'] = local_features

        return features, end_points['fp2_xyz'], end_points

if __name__=='__main__':
    model = Pointnet2Backbone(input_feature_dim=0)
    print(model)