
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time


def enforce_direction_symmetry(gvC):

    reference_dir = torch.tensor([1.0, 0.0, 0.0], device=gvC.device).view(1, 1, 3)

    dot_product = torch.sum(gvC * reference_dir, dim=-1)

    mask = dot_product < 0

    gvC_corrected = torch.where(mask.unsqueeze(-1), -gvC, gvC)

    return gvC_corrected

class E_disLoss(nn.Module):
    def __init__(self):
        super(E_disLoss, self).__init__()

    def forward(self, pred, gt):

        l2_norms = torch.norm(pred - gt, p=2, dim=2)  # shape: (B, N)
        loss = l2_norms
        return loss


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, pred, gt):

        dot_product = torch.sum(pred * gt, dim=2)  # shape: (B, N)
        abs_dot = torch.abs(dot_product)
        loss = -abs_dot
        return loss


class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, v_A, v_C):

        dot_product = torch.sum(v_A * v_C, dim=2)  # shape: (B, N)
        abs_dot = torch.abs(dot_product)
        loss = abs_dot
        return loss



def get_loss(end_points):
    objectness_loss, end_points = compute_objectness_loss(end_points)
    rot_loss, end_points = compute_VC_loss(end_points)
    offset_loss, end_points = compute_offset_loss(end_points)
    f_loss, end_points = feed_back_loss(end_points)
    loss = 8*objectness_loss + 2*rot_loss + 0.05*offset_loss + 0.1*f_loss  #2*rot,0.05*off

    end_points['overall_loss'] = loss
    return loss, end_points

def feed_back_loss(end_points):
    initial_approach = end_points['initial_approach_pred']
    initial_binormal = end_points['initial_binormal_pred']
    final_approach = end_points['approach_pred']
    final_binormal = end_points['binormal_pred']

    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    batch_scores = end_points['batch_scores']  # （B，N）
    batch_mask = end_points['batch_mask']

    objectness_label = torch.gather(objectness_label, 1, fp2_inds)
    objectness_mask = objectness_label.bool()  # (B, Ns)
    graspable_mask = batch_mask
    scores_mask = (batch_scores != 0)
    loss_mask = (objectness_mask & graspable_mask & scores_mask).float()
    f_a =  F.mse_loss(initial_approach, final_approach.detach())
    f_b = F.mse_loss(initial_binormal, final_binormal.detach())
    f_loss = torch.sum(f_a * loss_mask) / (loss_mask.sum() + 1e-6) + torch.sum(f_b * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['feed_back_loss'] = f_loss
    return f_loss,end_points



def compute_objectness_loss(end_points):
    # criterion = nn.BCELoss(reduction='mean')
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    objectness_pred = end_points['objectness_pred'].squeeze(1)
    objectness_label = end_points['objectness_label']   #(B,N)
    fp2_inds = end_points['fp2_inds'].long()
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)
    loss = criterion(objectness_pred, objectness_label.float())
    objectness_pred = (objectness_pred > 0.5).float()
    # print(objectness_pred)
    end_points['objectness_loss'] = loss
    end_points['objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()

    return loss, end_points

def compute_VC_loss(end_points):
    e_disloss = E_disLoss()
    cosineloss = CosineLoss()
    orthogonalloss = OrthogonalLoss()
    batch_approach_vector = end_points['batch_approach_vector']
    batch_anti_vector = end_points['batch_anti_vector']
    batch_scores = end_points['batch_scores']
    scores_mask = (batch_scores != 0)
    batch_mask = end_points['batch_mask']  #(B,N)
    # batch_mask_c = end_points['batch_mask_c']
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()

    objectness_label = torch.gather(objectness_label, 1, fp2_inds)   # (B, Ns)
    objectness_mask = objectness_label.bool()  # (B, Ns)
    graspable_mask = batch_mask
    loss_mask = (objectness_mask & graspable_mask & scores_mask).float()  # (B, Ns)
    # loss_mask = (objectness_mask & graspable_mask).float()
    mask = objectness_mask & graspable_mask & scores_mask
    rot_pred = end_points['rot_pred']   #(B, 6, N)
    g_v_pred = rot_pred[:, :3, :].permute(0, 2, 1).contiguous()  #(B,N,3)
    g_c_pred = rot_pred[:, 3:, :].permute(0, 2, 1).contiguous()  #(B,N,3)
    g_v = batch_approach_vector        #(B,N,3)
    g_c = batch_anti_vector  #(B,N,3)

    g_c = enforce_direction_symmetry(g_c)   #(B,N,3)


    #loss
    g_v_loss = torch.sum((5*e_disloss(g_v_pred,g_v) + cosineloss(g_v_pred,g_v) + orthogonalloss(g_v_pred,g_c_pred))*loss_mask)/(loss_mask.sum() + 1e-6)
    g_c_loss = torch.sum((5*e_disloss(g_c_pred,g_c) + cosineloss(g_c_pred,g_c) + orthogonalloss(g_v_pred,g_c_pred))*loss_mask)/(loss_mask.sum() + 1e-6)
    rot_loss = (g_v_loss + g_c_loss)
    end_points['rot_loss'] = rot_loss


    return rot_loss,end_points


def compute_offset_loss(end_points):
    batch_depth = end_points['batch_depth']
    batch_width = end_points['batch_width']
    batch_scores = end_points['batch_scores']  #（B，N）
    batch_mask = end_points['batch_mask']
    # batch_mask_c = end_points['batch_mask_c']
    offsets_pred = end_points['offsets_pred']

    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()

    objectness_label = torch.gather(objectness_label, 1, fp2_inds)
    objectness_mask = objectness_label.bool()  # (B, Ns)
    graspable_mask = batch_mask
    scores_mask = (batch_scores != 0)
    loss_mask = (objectness_mask & graspable_mask & scores_mask).float()  # (B, Ns)
    # loss_mask = (objectness_mask & graspable_mask).float()
    mask = objectness_mask & graspable_mask & scores_mask
    depth_pred = offsets_pred[:, 0, :]   #（B，N）
    width_pred = offsets_pred[:, 1, :]
    score_pred = offsets_pred[:, 2, :]
    # print(f'depth:{depth_pred[mask]}')
    # print(f'width:{width_pred[mask]}')
    # print(f'score:{score_pred[mask]}')
    depth_label = batch_depth*100  #（B，N）
    width_label = batch_width*100
    score_label = batch_scores

    grasp_width_loss = F.mse_loss(width_pred.float(), width_label.float())
    grasp_width_loss = torch.sum(grasp_width_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    grasp_depth_loss = F.mse_loss(depth_pred.float(), depth_label.float())
    grasp_depth_loss = torch.sum(grasp_depth_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    grasp_score_loss = F.mse_loss(score_pred.float(), score_label.float())
    grasp_score_loss = torch.sum(grasp_score_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    offset_loss = grasp_depth_loss+grasp_width_loss+ grasp_score_loss  #center

    end_points['offset_loss'] = offset_loss

    return offset_loss,end_points

