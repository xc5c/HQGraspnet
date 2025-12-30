# from encodings.idna import sace_prefix

import torch
import cv2
import sys
sys.path.append('/home/xcc/PycharmProjects')
from all_code.graspnet.models.backbone_f import Pointnet2Backbone
import scipy.io as scio
import numpy as np
from PIL import Image
import open3d as o3d
import torch.nn
from all_code.generate_grasp_label.grasp import Grasp,GraspGroup
from all_code.graspnet.utils.collision_detector import ModelFreeCollisionDetector
import collections.abc as container_abcs
from torch.utils.data import DataLoader
from all_code.graspnet.dataset.improve_dataset  import load_grasp_labels, GraspNetDataset
from graspnetAPI import GraspNetEval

def get_scene_name(scene_id):
    return 'scene_{:04d}'.format(scene_id)

class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed



def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed

        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg > 0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
    mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
    mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask



def single_clouds(depth_file_path, color_file_path, meta_path=None, seg_file_path = None, num_points=20000, sample=True):

    if meta_path:
        meta = scio.loadmat(meta_path)
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1],
                        intrinsic[0][2], intrinsic[1][2], factor_depth)
    else:
        camera = CameraInfo(
            width=640,
            height=480,
            fx=379.936,
            fy=379.936,
            cx=321.159,
            cy=240.710,
            scale=1000.0  # 深度图缩放因子 (毫米转米)
        )

    depth = np.array(Image.open(depth_file_path))
    color = np.array(Image.open(color_file_path), dtype=float) / 255.0

    seg = np.array(Image.open(seg_file_path)) if seg_file_path is not None else None
    # mask = (seg>0)

    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    depth_mask = (depth > 0)

    if seg is not None:
        workspace_mask = get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0.02)
        mask = depth_mask & workspace_mask
        mask = (seg >0)
    else:
        mask = depth_mask



    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if sample:

        if len(cloud_masked) >= num_points:

            idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
        else:

            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)


        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        return cloud_sampled, color_sampled

    else:

        return cloud_masked, color_masked






def orthogonalize_vectors_torch(approach_batch, anti_batch):

    approach_norm = torch.norm(approach_batch, p=2, dim=2, keepdim=True)
    ortho_approach = approach_batch / (approach_norm + 1e-8)

    dot_products = torch.einsum('bni,bni->bn', anti_batch, ortho_approach)
    projections = torch.einsum('bn,bni->bni', dot_products, ortho_approach)
    ortho_anti = anti_batch - projections

    anti_norm = torch.norm(ortho_anti, p=2, dim=2, keepdim=True)
    ortho_anti = ortho_anti / (anti_norm + 1e-8)

    return ortho_approach, ortho_anti


def load_model(model_path, device="cuda:0"):


    net = Pointnet2Backbone(input_feature_dim=0,is_training=False,is_demo=True).to(device)

    raw_state_dict = torch.load(model_path, map_location=device)

    new_state_dict = {}
    for key, value in raw_state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        new_key = new_key.replace("mlp_layer", "mlp_module.layer")

        new_state_dict[new_key] = value.to(device)

    missing_keys, unexpected_keys = net.load_state_dict(new_state_dict)



    return net



def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))



def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass




if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = '/home/xcc/data_1Billion'

    model_path = "/home/xcc/下载/model_epoch_100.pth"
    net = load_model(model_path)

    TEST_DATASET = GraspNetDataset(root, valid_obj_idxs=None, grasp_labels=None, camera='kinect', split='test',
                                    num_points=20000, remove_outlier=True, augment=True,load_label=False)
    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False,
                                  num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)


    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):

        scene_id = 100 + batch_idx // 256
        ann_id = batch_idx % 256
        # print(batch_idx)

        raw_cloud, raw_color = TEST_DATASET.get_data(batch_idx, return_raw_cloud=True)

        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        colors = batch_data_label['cloud_colors']
        pointcloud = batch_data_label['point_clouds']


        pointcloud= pointcloud.cpu().numpy()[0]

        from scipy.linalg import inv
        pointcloud = torch.from_numpy(pointcloud).float().unsqueeze(0).to(device)

        with torch.no_grad():
            net.eval()
            features, num_seeds, end_points = net(pointcloud)

            objectness_pred = end_points['objectness_pred'].squeeze(1)
            objectness_mask = (objectness_pred > 0.5)
            objectness_mask = objectness_mask.cpu().numpy().squeeze(0)

            rot_pred = end_points['rot_pred']  # (B, 6, N)
            g_v_pred = rot_pred[:, :3, :].permute(0, 2, 1).contiguous()  # (B,N,3)
            g_c_pred = rot_pred[:, 3:, :].permute(0, 2, 1).contiguous()  # (B,N,3)
            offsets_pred = end_points['offsets_pred']
            # print(offsets_pred)
            # print(rot_pred)
            depth_pred = offsets_pred[:, 0, :]/100# （B，N）
            width_pred = offsets_pred[:, 1, :]/100
            score_pred = offsets_pred[:, 2, :]
            fp2_inds = end_points['fp2_inds'].long()
            # print(depth_pred)

            grasp_points = torch.gather(
                input=pointcloud,
                dim=1,
                index=fp2_inds.unsqueeze(-1).expand(-1, -1, 3).long()
            )
            # print(g_v_pred.shape,g_c_pred.shape)
            g_v_pred, g_c_pred = orthogonalize_vectors_torch(g_v_pred, g_c_pred)


            z = torch.cross(g_v_pred, g_c_pred, dim=2)
            z_norm = torch.norm(z, p=2, dim=2, keepdim=True)
            z = z / (z_norm + 1e-8)

            scene_transform = torch.stack([
                g_v_pred,
                g_c_pred,
                z
            ], dim=3)    #(B,N,3,3)


            scene_pcd = o3d.geometry.PointCloud()
            # scene_pcd.points = o3d.utility.Vector3dVector(pointcloud.cpu().numpy())
            # scene_pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
            pointcloud_np = pointcloud.cpu().numpy()[0].astype(np.float64)
            colors_np = colors.cpu().numpy()[0].astype(np.float64)

            scene_pcd.points = o3d.utility.Vector3dVector(pointcloud_np)
            scene_pcd.colors = o3d.utility.Vector3dVector(colors_np)


            scene_grippers = []
            grasp_arrays = []
            for k in range(grasp_points.shape[1]):

                # center_k = scene_center[k].cpu().numpy()  # (3,)
                grasp_points_k = grasp_points[0, k].cpu().numpy()
                transform_k = scene_transform[0, k].cpu().numpy()  # (3, 3)
                width_k = width_pred[0, k].cpu().item()
                depth_k = depth_pred[0, k].cpu().item()



                if width_k >0.08:
                    width_k=0.08
                # print(depth_k)
                score_k = score_pred[0, k].cpu().item()
                height_K = 0.02
                grasp_array = np.concatenate([
                    np.array([score_k, width_k, height_K, depth_k], dtype=np.float64),
                    transform_k.reshape(-1).astype(np.float64),
                    grasp_points_k.astype(np.float64),
                    np.array([0], dtype=np.float64)
                ])
                grasp_arrays.append(grasp_array)


            grasp_group = GraspGroup(np.stack(grasp_arrays, axis=0))
            detector = ModelFreeCollisionDetector(scene_pcd.points, voxel_size=0.01)#0.005
            collision_mask = detector.detect(grasp_group, approach_dist=0.05, collision_thresh=0.01)
            score_mask = (grasp_group.scores>0.0)
            safe_grasp_group = grasp_group[ ~collision_mask &objectness_mask &score_mask]
            # safe_grasp_group = safe_grasp_group.nms(0.03, 30.0 / 180 * np.pi)
            # safe_grasp_group.sort_by_score()
            # safe_grasp_group = safe_grasp_group[:50]
            global_safe_geometries = safe_grasp_group.to_open3d_geometry_list()
            print(len(safe_grasp_group))

            # top_grasp_group = safe_grasp_group[np.argsort(safe_grasp_group.scores)[::-1][:100]]
            # global_safe_geometries = top_grasp_group.to_open3d_geometry_list()

            geometries_to_draw = [scene_pcd] + global_safe_geometries
            o3d.visualization.draw_geometries(geometries_to_draw)

        #
        #     scene_name = get_scene_name(scene_id)
        #     import os
        #     save_dir = os.path.join('./eval_results_100', scene_name, 'kinect')
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     save_path = os.path.join(save_dir, f"{ann_id:04d}.npy")
        #     safe_grasp_group.save_npy(save_path)
        #
        #     print(f'Saved grasps for scene {scene_id}, ann {ann_id} to {save_path}')
        #
        # ge = GraspNetEval(root=root, camera='kinect', split='test')
        # res, ap = ge.eval_novel('./eval_results_100', proc=2)
        # friction_coefficients = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        #
        # ap_by_friction = np.mean(res, axis=(0, 1, 2))
        #
        # ap_08 = ap_by_friction[3]
        # ap_04 = ap_by_friction[1]
        #
        # print(f"Miu 0.8: AP = {100 * ap_08:.2f}%")
        # print(f"Miu 0.4: AP = {100 * ap_04:.2f}%")







