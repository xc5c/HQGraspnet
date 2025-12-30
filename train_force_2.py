import numpy as np
import torch
import os
import torch.optim as optim
from tqdm import tqdm
import sys

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.append('/xcc/home/PycharmProjects')
from all_code.graspnet.models.backbone_f import Pointnet2Backbone
import collections.abc as container_abcs
from torch.utils.data import DataLoader
from all_code.graspnet.dataset.improve_dataset  import load_grasp_labels, GraspNetDataset
from all_code.graspnet.models.loss_new import get_loss
# from all_code.graspnet.utils.label_generation_new import process_grasp_labels


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]


root = '/home/xcc/data_1Billion'
valid_obj_idxs, grasp_labels = load_grasp_labels(root)
TRAIN_DATASET = GraspNetDataset(root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,remove_outlier=True,
                                 augment=True,load_label=True)

TRAIN_DATALOADER = DataLoader(
    TRAIN_DATASET,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    worker_init_fn=my_worker_init_fn,
    collate_fn=collate_fn
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Pointnet2Backbone(input_feature_dim=0).to(device)

optimizer = optim.Adam(net.parameters(), lr=0.005)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.0005
)


def train_one_epoch(epoch):
    net.train()
    total_loss = 0.0
    total_batches = 0

    train_loader = tqdm(enumerate(TRAIN_DATALOADER),
                        total=len(TRAIN_DATALOADER),
                        desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, batch_data_label in train_loader:
        total_batches += 1
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        pointcloud = batch_data_label['point_clouds']
        features, num_seeds, end_points = net(pointcloud, batch_data_label)

        # Compute loss and gradients, update parameters.
        # end_points = process_grasp_labels(end_points)
        loss, end_points = get_loss(end_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        train_loader.set_postfix({'loss': loss.item()})

    avg_train_loss = total_loss / total_batches
    return avg_train_loss


if __name__ == '__main__':
    checkpoint_dir = "check_points_d1"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_train_loss = float('inf')
    train_loss_history = []
    total_epochs = 100

    for epoch in range(total_epochs):
        train_loss = train_one_epoch(epoch)

        scheduler.step()

        train_loss_history.append(train_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{total_epochs} | LR: {current_lr:.5f} | Train Loss: {train_loss:.4f}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'best_train_loss': best_train_loss,
                'learning_rate': current_lr  # 保存当前学习率
            }, os.path.join(checkpoint_dir, "best_checkpoint.pth"))

            torch.save(net.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best model saved at epoch {epoch + 1} with train loss {train_loss:.4f}")

    final_checkpoint = {
        'epoch': total_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'best_train_loss': best_train_loss,
        'history': train_loss_history,
        'final_learning_rate': current_lr
    }
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, "final_checkpoint.pth"))
    torch.save(net.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    print("Training completed. Final model saved.")