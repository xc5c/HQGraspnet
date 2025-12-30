
import open3d as o3d
import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
# from torch._six import container_abcs
import collections.abc as container_abcs

from torch.onnx.symbolic_opset9 import arange
from torch.utils.data import Dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from all_code.graspnet.utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points


def orthogonalize_vectors(approach_batch, anti_batch):

    approach_norm = np.linalg.norm(approach_batch, axis=1, keepdims=True)
    ortho_approach = approach_batch / (approach_norm + 1e-8)

    dot_products = np.einsum('ij,ij->i', anti_batch, ortho_approach)
    projections = np.einsum('i,ij->ij', dot_products, ortho_approach)
    ortho_anti = anti_batch - projections

    anti_norm = np.linalg.norm(ortho_anti, axis=1, keepdims=True)
    ortho_anti = ortho_anti / (anti_norm + 1e-8)
    return ortho_approach, ortho_anti

class GraspNetDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(190))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'simple_datasets_force', 'collision_model_label', f'{x}.npz'))
                # collision_labels = np.load(os.path.join(root,'simple_expand_datasets','collision_model_label', f'{x}.npz'))
                # collision_labels = np.load(os.path.join(root, 'simple_datasets_contact', 'collision_label', f'{x}.npz'))
                self.collision_labels[x.strip()] = {}
                for i, key in enumerate(collision_labels.files):
                    self.collision_labels[x.strip()][i] = collision_labels[key]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)


    def augment_data(self, point_clouds, colors,object_poses_list):
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        if self.split == 'train':
            full_rot_angle = np.random.random() * 2 * np.pi  # 0-360度
            fc, fs = np.cos(full_rot_angle), np.sin(full_rot_angle)
            full_rot_mat = np.array([[1, 0, 0],
                                     [0, fc, -fs],
                                     [0, fs, fc]])
            point_clouds = transform_point_cloud(point_clouds, full_rot_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(full_rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, colors,object_poses_list



    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)

        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        contact_points_list = []
        anti_vector_list = []
        approach_vector_list = []
        width_list = []
        depth_list = []
        miu_list = []
        grasp_scores_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            # points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            end_points,grasp_points,approach_vector,width,depth,miu,score_1,score_2 = self.grasp_labels[obj_idx]
            width[width < 0.08] += 0.01
            end_point1 = end_points[:, 0, :]
            contact_points = end_point1
            end_point2 = end_points[:, 1, :]
            anti_vector = (end_point2 - end_point1) / np.linalg.norm(end_point2 - end_point1)
            approach_vector, anti_vector = orthogonalize_vectors( approach_vector, anti_vector )
            # grasp_points = grasp_points.squeeze(axis=1)     #simple_datasets用
            scores = np.array([x + 0.001 * y for x, y in zip(score_1, score_2)])
            collision = self.collision_labels[scene][i]

            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], grasp_points,
                                                             poses[:, :, i], th=0.01)
                # visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], contact_points,
                #                                              poses[:, :, i], th=0.01)
                anti_vector = anti_vector[visible_mask]
                grasp_points = grasp_points[visible_mask]
                contact_points = end_point1[visible_mask]
                approach_vector = approach_vector[visible_mask]
                width = width[visible_mask]
                depth = depth[visible_mask]
                miu = miu[visible_mask]
                scores = scores[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(grasp_points), min(max(int(len(grasp_points) / 4), 300), len(grasp_points)), replace=False)
            grasp_points_list.append(grasp_points[idxs])
            contact_points_list.append(contact_points[idxs])
            anti_vector_list.append(anti_vector[idxs])
            approach_vector_list.append(approach_vector[idxs])
            width_list.append(width[idxs])
            depth_list.append(depth[idxs])
            miu_list.append(miu[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)

        if self.augment:
            cloud_sampled, color_sampled, object_poses_list = self.augment_data(cloud_sampled, color_sampled, object_poses_list)

        ret_dict = {}
        cloud_with_color = np.concatenate([cloud_sampled, color_sampled], axis=1)
        ret_dict['point_clouds_colors'] = cloud_with_color.astype(np.float32)
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['contact_points_list'] = contact_points_list
        ret_dict['anti_vector_list'] = anti_vector_list
        ret_dict['approach_vector_list'] = approach_vector_list
        ret_dict['width_list'] = width_list
        ret_dict['depth_list'] = depth_list
        ret_dict['scores_list'] = grasp_scores_list

        return ret_dict


def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18: continue
        valid_obj_idxs.append(i + 1)  # here align with label png
        # label = np.load(os.path.join(root, 'simple_datasets', 'grasp_label','simple_graspdataset_{}.npy'.format(str(i))), allow_pickle=True)
        # label = np.load(os.path.join(root,'simple_expand_datasets', 'grasp_label', 'simple_expand_graspdataset_{}.npy'.format(str(i))),allow_pickle=True)
        label = np.load(os.path.join(root, 'simple_datasets_force', 'grasp_label','simple_expand_graspdataset_{}.npy'.format(str(i))), allow_pickle=True)
        # label = np.load(os.path.join(root, 'simple_datasets_contact', 'grasp_label','sampled_graspdataset_contact_{}.npy'.format(str(i))), allow_pickle=True)

        all_end_points = []
        all_grasp_points = []
        all_approach_vectors = []
        all_widths = []
        all_depths = []
        all_mius = []
        all_scores_1 = []
        all_scores_2 = []


        for label_dict in label:

            all_end_points.append(label_dict['end_points'])
            all_grasp_points.append(label_dict['grasp_point'])
            all_approach_vectors.append(label_dict['approach_vector'])
            all_widths.append(label_dict['width'])
            all_depths.append(label_dict['depth'])
            all_mius.append(label_dict['miu'])
            all_scores_1.append(label_dict['score_1'])
            all_scores_2.append(label_dict['score_2'])


        grasp_labels[i + 1] = (
            np.array(all_end_points),
            np.array(all_grasp_points),
            np.array(all_approach_vectors),
            np.array(all_widths),
            np.array(all_depths),
            np.array(all_mius),
            np.array(all_scores_1),
            np.array(all_scores_2)
        )

    return valid_obj_idxs, grasp_labels


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))




