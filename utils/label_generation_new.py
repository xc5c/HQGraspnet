

import os
import sys
import torch
from knn_cuda import KNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'knn'))
# from all_code.graspnet.knn.knn_modules import knn
from all_code.graspnet.utils.loss_utils import GRASP_MAX_WIDTH, batch_viewpoint_params_to_matrix, \
    transform_point_cloud, generate_grasp_views


def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    clouds = end_points['input_xyz']  # (B, N, 3)
    seed_xyzs = end_points['fp2_xyz']  # (B, Ns, 3)
    batch_size, num_samples, _ = seed_xyzs.size()

    knn = KNN(k=1, transpose_mode=True)
    batch_contact_points = []
    batch_grasp_points = []
    batch_approach_vector = []
    batch_anti_vector = []
    batch_width = []
    batch_depth = []
    batch_scores = []
    batch_mask = []
    batch_mask_c = []
    for i in range(len(clouds)):
        seed_xyz = seed_xyzs[i]  # (Ns, 3)
        poses = end_points['object_poses_list'][i]  # [(3, 4),]

        # get merged grasp points for label computation
        grasp_points_merged = []
        contact_points_merged = []
        approach_vector_merged = []
        anti_vector_merged = []
        width_merged = []
        depth_merged = []
        scores_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx]
            # contact_points = end_points['contact_points_list'][i][obj_idx]
            approach_vector = end_points['approach_vector_list'][i][obj_idx]
            anti_vector = end_points['anti_vector_list'][i][obj_idx]
            width =  end_points['width_list'][i][obj_idx]
            depth = end_points['depth_list'][i][obj_idx]
            scores = end_points['scores_list'][i][obj_idx]
            num_grasp_points = grasp_points.size(0)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            # contact_points_trans = transform_point_cloud(contact_points,pose,'3x4')
            approach_vector_trans = transform_point_cloud(approach_vector, pose[:3, :3], '3x3')
            anti_vector_trans = transform_point_cloud(anti_vector, pose[:3, :3], '3x3')
            # add to list
            grasp_points_merged.append(grasp_points_trans)
            # contact_points_merged.append(contact_points_trans)
            approach_vector_merged.append(approach_vector_trans)
            anti_vector_merged.append(anti_vector_trans)
            width_merged.append(width)
            depth_merged.append(depth)
            scores_merged.append(scores)
        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)
        # contact_points_merged = torch.cat(contact_points_merged, dim=0)
        approach_vector_merged = torch.cat(approach_vector_merged, dim=0)
        anti_vector_merged = torch.cat(anti_vector_merged, dim=0)
        width_merged = torch.cat(width_merged, dim=0)
        depth_merged = torch.cat(depth_merged, dim=0)
        scores_merged = torch.cat(scores_merged,dim=0)
        # print(grasp_points_merged)
        # compute nearest neighbors
        seed_xyz_ = seed_xyz.unsqueeze(0)  # (1, Ns, 3)
        grasp_points_merged_ = grasp_points_merged.unsqueeze(0)  # (1, Np_total, 3)

        dist, nn_inds = knn(ref=grasp_points_merged_, query=seed_xyz_)
        nn_inds = nn_inds.squeeze(0).squeeze(-1).long()  # (Ns, )
        dist = dist.squeeze(0).squeeze(-1)
        mask = dist <= 0.01


        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
        approach_vector_merged = torch.index_select( approach_vector_merged, 0, nn_inds)
        anti_vector_merged = torch.index_select(anti_vector_merged, 0, nn_inds)
        width_merged = torch.index_select(width_merged, 0, nn_inds)
        depth_merged = torch.index_select(depth_merged, 0, nn_inds)
        scores_merged = torch.index_select(scores_merged, 0, nn_inds)


        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_contact_points.append(contact_points_merged)
        batch_approach_vector.append(approach_vector_merged)
        batch_anti_vector.append(anti_vector_merged)
        batch_width.append(width_merged)
        batch_depth.append(depth_merged)
        batch_scores.append(scores_merged)
        batch_mask.append(mask)
        # batch_mask_c.append(mask_c)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    # batch_contact_points = torch.stack(batch_contact_points, 0)
    batch_approach_vector = torch.stack(batch_approach_vector, 0)
    batch_anti_vector = torch.stack(batch_anti_vector, 0)
    batch_width = torch.stack(batch_width, 0)
    batch_depth = torch.stack(batch_depth, 0)
    batch_scores = torch.stack(batch_scores, 0)
    batch_mask = torch.stack(batch_mask,0)
    # batch_mask_c = torch.stack(batch_mask_c, 0)



    end_points['batch_grasp_point'] = batch_grasp_points
    # end_points['batch_contact_points'] = batch_contact_points
    end_points['batch_approach_vector'] = batch_approach_vector
    end_points['batch_anti_vector'] = batch_anti_vector
    end_points['batch_width'] = batch_width
    end_points['batch_depth'] = batch_depth
    end_points['batch_scores'] = batch_scores
    end_points['batch_mask'] = batch_mask
    # end_points['batch_mask_c'] = batch_mask_c


    return end_points
