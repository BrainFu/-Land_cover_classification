import logging
import numpy as np
import pandas as pd
from osgeo import gdal
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, cohen_kappa_score
from Unet_model import UNet
gdal.UseExceptions()
########################################
# 配置参数
########################################
IMAGE_PATH = r"E:\kunming\km_msimg\km232_vwm_zscore.tif"
LABEL_IMAGE_PATH = r"E:\kunming\km_msimg\gisa_extract_vwm.tif"
SAMPLES_CSV = r"E:\kunming\km_sample\isa_sa9630\samples_points.csv"

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PATCH_SIZE = 32
PATIENCE = 5

MODEL_SAVE_PATH = r'E:\kunming\model\model_best_multibranch100.pth'
LOG_FILE = r'E:\kunming\script_filedirectory\training_log.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)]
)

########################################
# 数据集定义
########################################
class MultiSourceDataset(Dataset):
    def __init__(self, image_path, sample_points_path, label_image_path, patch_size=32, augment=False):
        self.image_path = image_path
        self.sample_points = pd.read_csv(sample_points_path)
        self.label_image_path = label_image_path
        self.patch_size = patch_size
        self.augment = augment

        # 打开影像
        self.image = gdal.Open(self.image_path)
        self.label_image = gdal.Open(self.label_image_path)
        self.image_width = self.image.RasterXSize
        self.image_height = self.image.RasterYSize

        # 获取 NoData 值
        self.no_data_val = self.label_image.GetRasterBand(1).GetNoDataValue() or -9999
        print(f"NoData value used for labels: {self.no_data_val}")

        # 数据增强
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
        ])

    def __len__(self):
        return len(self.sample_points)

    def __getitem__(self, idx):
        pixel_x = int(self.sample_points.iloc[idx]['x'])
        pixel_y = int(self.sample_points.iloc[idx]['y'])

        # 提取影像 Patch
        image_patch = self.get_patch(self.image, pixel_x, pixel_y)
        radar = torch.tensor(image_patch[0:4], dtype=torch.float32)
        optical = torch.tensor(image_patch[4:8], dtype=torch.float32)
        terrain = torch.tensor(image_patch[8], dtype=torch.float32).unsqueeze(0)

        # 提取标签 Patch
        label_patch = self.get_patch(self.label_image, pixel_x, pixel_y, single_band=True)
        label = torch.tensor(self.process_label(label_patch), dtype=torch.long)

        # 检查标签值
        # unique_labels = torch.unique(label)
        # print("Unique label values in this batch:", unique_labels)

        # 数据增强
        if self.augment:
            radar, optical, terrain, label = self.apply_augmentation(radar, optical, terrain, label)

        return radar, optical, terrain, label

    def get_patch(self, gdal_data, x, y, single_band=False):
        half_patch = self.patch_size // 2
        x_start = max(x - half_patch, 0)
        y_start = max(y - half_patch, 0)
        x_end = min(x + half_patch, self.image_width)
        y_end = min(y + half_patch, self.image_height)

        if single_band:
            patch = np.full((self.patch_size, self.patch_size), self.no_data_val, dtype=np.float32)
            data_array = gdal_data.GetRasterBand(1).ReadAsArray(x_start, y_start, x_end - x_start, y_end - y_start)
            patch[:data_array.shape[0], :data_array.shape[1]] = data_array
        else:
            band_count = gdal_data.RasterCount
            patch = np.zeros((band_count, self.patch_size, self.patch_size), dtype=np.float32)
            for i in range(band_count):
                band = gdal_data.GetRasterBand(i + 1)
                data_array = band.ReadAsArray(x_start, y_start, x_end - x_start, y_end - y_start)
                patch[i, :data_array.shape[0], :data_array.shape[1]] = data_array
        return patch

    def process_label(self, label_patch):
        label_patch[label_patch == self.no_data_val] = 2  # 将 NoData 值设为类别 2
        label_patch[label_patch == 3] = 2  # 将标签值为 3 的像素也设为类别 2
        # 设置有效类别标签
        label_patch[label_patch == 0] = 0  # 非不透水面
        label_patch[label_patch == 1] = 1  # 不透水面
        # 将其他任何值设为 2
        mask = (label_patch != 0) & (label_patch != 1) & (label_patch != 2)
        label_patch[mask] = 2
        return label_patch

    def apply_augmentation(self, radar, optical, terrain, label):
        if torch.rand(1).item() > 0.5:
            radar = radar.flip(-1)
            optical = optical.flip(-1)
            terrain = terrain.flip(-1)
            label = label.flip(-1)
        return radar, optical, terrain, label

########################################
# 模型定义
########################################


class MultiBranchUNet(nn.Module):
    def __init__(self, out_channels):
        super(MultiBranchUNet, self).__init__()

        # 定义每个分支的 UNet 子网络
        self.radar_unet = UNet(input_channels=4, out_channels=out_channels)  # 雷达数据 4 通道
        self.optical_unet = UNet(input_channels=4, out_channels=out_channels)  # 光学数据 4 通道
        self.terrain_unet = UNet(input_channels=1, out_channels=out_channels)  # 地形数据 1 通道

        # 最终融合层
        self.final_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)  # 融合三个分支的输出

    def forward(self, radar, optical, terrain):
        # 各分支独立通过对应的 UNet 子网络
        radar_output = self.radar_unet(radar)  # 输出形状 (B, out_channels, H, W)
        optical_output = self.optical_unet(optical)  # 输出形状 (B, out_channels, H, W)
        terrain_output = self.terrain_unet(terrain)  # 输出形状 (B, out_channels, H, W)

        # 融合三个分支的输出
        fused = torch.cat([radar_output, optical_output, terrain_output], dim=1)  # (B, out_channels*3, H, W)

        # 通过最终卷积层生成最终输出
        output = self.final_conv(fused)  # (B, out_channels, H, W)

        return output


##### 早停定义 #####

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5, save_path=MODEL_SAVE_PATH):
        """
        早停机制，用于防止过拟合。
        :param patience: 容忍验证损失不下降的 epoch 数
        :param min_delta: 验证损失改进的最小值，低于此值视为没有改进
        :param save_path: 保存最佳模型的路径
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss):
        """
        检查是否需要早停。
        :param val_loss: 当前验证集损失
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            # 验证损失有改进
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 验证损失没有改进
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model(self, model):
        """
        保存当前的最佳模型。
        :param model: PyTorch 模型
        """
        torch.save(model.state_dict(), self.save_path)
        logging.info(f"Model saved with validation loss: {self.best_loss:.7f}")

        try:
            torch.save(model.state_dict(), self.save_path)
            logging.info(f"Model saved with validation loss: {self.best_loss:.7f}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")


def move_to_device(*tensors, device):
    """
    将一个或多个张量转移到指定设备（CPU 或 GPU）。
    :param tensors: 一个或多个 PyTorch 张量
    :param device: 目标设备，如 'cuda' 或 'cpu'
    :return: 转移到目标设备的张量
    """
    return tuple(tensor.to(device) for tensor in tensors)

########################################
# 训练函数
########################################
def train_model(model, train_loader, val_loader, device, optimizer, criterion, scaler, num_epochs, patience, writer=None):
    early_stopping = EarlyStopping(patience=patience, save_path=MODEL_SAVE_PATH)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_labels = []
        all_preds = []

        for radar, optical, terrain, labels in train_loader:
            radar, optical, terrain, labels = move_to_device(radar, optical, terrain, labels, device=device)
            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(radar, optical, terrain)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            # 计算分类指标
            preds = torch.argmax(outputs, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        # print(f"all_preds type: {type(all_preds)}, shape: {all_preds.shape}")
        # print(f"all_labels type: {type(all_labels)}, shape: {all_labels.shape}")

        # 传递 all_preds 和 all_labels 到 calculate_metrics
        precision, recall, f1, iou, kappa = calculate_metrics(all_preds, all_labels)


        # 记录到 TensorBoard
        if writer is not None:
            writer.add_scalar(f'train/Loss', avg_train_loss, epoch)
            writer.add_scalar(f'train/Precision', precision, epoch)
            writer.add_scalar(f'train/Recall', recall, epoch)
            writer.add_scalar(f'train/F1', f1, epoch)
            writer.add_scalar(f'train/IoU', iou, epoch)
            writer.add_scalar(f'train/Kappa', kappa, epoch)

        # 验证阶段
        avg_val_loss, val_precision, val_recall, val_f1, val_iou, val_kappa = validate_model(
            model, val_loader, device, criterion, writer, epoch
        )

        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | "
            f"Train Precision: {precision:.6f} | Train Recall: {recall:.6f} | "
            f"Train F1: {f1:.6f} | Train IoU: {iou:.6f} | Train Kappa: {kappa:.6f}"
        )

        # 早停检查
        logging.info(f"Validation Loss at Epoch {epoch + 1}: {avg_val_loss:.6f}")
        early_stopping(avg_val_loss)
        if avg_val_loss < early_stopping.best_loss or early_stopping.best_loss is None:
            early_stopping.save_model(model)
            logging.info(f"Model saved at Epoch {epoch + 1} with validation loss: {avg_val_loss:.6f}")
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

        else:
            if epoch % 5 == 0:  # 示例：每 5 个 epoch 保存一次模型作为后备
                early_stopping.save_model(model)
                logging.info(f"Model saved at Epoch {epoch + 1} as fallback.")

    return model



def validate_model(model, val_loader, device, criterion, writer=None, epoch=None):
    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for radar, optical, terrain, labels in val_loader:
            radar, optical, terrain, labels = move_to_device(radar, optical, terrain, labels, device=device)
            outputs = model(radar, optical, terrain)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算分类指标
            preds = torch.argmax(outputs, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # 计算验证集指标
    avg_val_loss = val_loss / len(val_loader)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # print(f"all_preds type: {type(all_preds)}, shape: {all_preds.shape}")
    # print(f"all_labels type: {type(all_labels)}, shape: {all_labels.shape}")

    # 传递 all_preds 和 all_labels 到 calculate_metrics
    precision, recall, f1, iou, kappa = calculate_metrics(all_preds, all_labels)

    # 记录到 TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar(f'val/Loss', avg_val_loss, epoch)
        writer.add_scalar(f'val/Precision', precision, epoch)
        writer.add_scalar(f'val/Recall', recall, epoch)
        writer.add_scalar(f'val/F1', f1, epoch)
        writer.add_scalar(f'val/IoU', iou, epoch)
        writer.add_scalar(f'val/Kappa', kappa, epoch)

    return avg_val_loss, precision, recall, f1, iou, kappa



def calculate_metrics(preds, labels):
    """
    计算精确率、召回率、F1 分数、IoU 和 Kappa 系数
    假设 preds 和 labels 已经是 NumPy 数组
    """
    # 过滤有效的预测与标签
    valid_mask = labels != -1
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    iou = jaccard_score(labels, preds, average='weighted')
    kappa = cohen_kappa_score(labels, preds)

    return precision, recall, f1, iou, kappa




########################################
# 主流程
########################################
if __name__ == "__main__":
    writer = SummaryWriter(log_dir='../runs/multi_branch_unet')
    dataset = MultiSourceDataset(IMAGE_PATH, SAMPLES_CSV, LABEL_IMAGE_PATH, PATCH_SIZE, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiBranchUNet(out_channels=3).to(device)  # 传递 out_channels 参数
    # 设置 ignore_index=-1，忽略 NoData 类别
    criterion = nn.CrossEntropyLoss(ignore_index=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = amp.GradScaler()

    trained_model = train_model(
        model, train_loader, val_loader, device, optimizer, criterion, scaler, NUM_EPOCHS, PATIENCE
    )
    logging.info("Training completed.")

