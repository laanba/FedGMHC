import torch
import copy
from torch.utils.data import DataLoader, Subset, Dataset

from demo import MobileNetV2UNet
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Client:
    def __init__(self, client_id, dataset, indices, device):
        self.client_id = client_id
        # 根据分配的索引获取本地子集
        self.train_loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=True)
        self.device = device

    def local_train(self, model, epochs=1, lr=0.01):
        model.train()
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

        # 返回训练后的状态字典（本地模型参数）
        return model.state_dict()


class CamVidDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'train')
        self.mask_dir = os.path.join(root_dir, 'train_labels')
        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        # 标签通常是单通道的索引图，使用 "L" 或直接转 numpy
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            # 注意：分割任务中，图像和掩码必须同步变换（如旋转、裁剪）
            # 这里简单演示只做基础转换
            image = self.transform(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

def federated_aggregate(global_model, client_weights, client_lens):
    """
    global_model: 当前全局模型
    client_weights: 列表，存储每个客户端上传的 state_dict
    client_lens: 列表，存储每个客户端的数据量
    """
    total_data = sum(client_lens)
    global_dict = copy.deepcopy(client_weights[0])

    # 初始化全局字典为 0
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] * (client_lens[0] / total_data)

    # 加权平均其余客户端参数
    for i in range(1, len(client_weights)):
        fraction = client_lens[i] / total_data
        for key in global_dict.keys():
            global_dict[key] += client_weights[i][key] * fraction

    # 更新全局模型
    global_model.load_state_dict(global_dict)
    return global_model


def distribute_data(dataset, num_clients, is_non_iid=True):
    """简单模拟 Non-IID：每个客户端只获得特定类别的样本"""
    indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    client_data_indices = []

    if not is_non_iid:
        # IID: 随机打乱平分
        np.random.shuffle(indices)
        client_data_indices = np.array_split(indices, num_clients)
    else:
        # Non-IID: 按标签排序后分块，使每个客户端只拥有 1-2 个类别的图片
        label_idx = np.argsort(labels)
        client_data_indices = np.array_split(label_idx, num_clients)

    return client_data_indices


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 加载你之前的 MobileNetV2UNet 或简单模型
    global_model = MobileNetV2UNet(num_classes=10).to(device)

    # 2. 模拟数据分配 (假设使用 CIFAR10)
    from torchvision import datasets, transforms
    train_dataset = CamVidDataset('./data', transform=transforms.ToTensor())
    num_images = len(train_dataset)
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    # 3. 假设 5 个客户端，平分数据 (IID 模式)
    user_groups = np.array_split(indices, 5)

    # 3. 联邦训练轮数
    for round in range(20):
        print(f"--- Round {round + 1} ---")
        local_weights = []
        local_lens = []

        # 模拟客户端并行执行（此处用循环代替）
        for i in range(5):
            # 实例化客户端
            client = Client(i, train_dataset, user_groups[i], device)
            # 全局模型分发给客户端
            local_model = copy.deepcopy(global_model)
            # 客户端本地训练
            weights = client.local_train(local_model)

            local_weights.append(weights)
            local_lens.append(len(user_groups[i]))

        # 4. 服务端聚合
        global_model = federated_aggregate(global_model, local_weights, local_lens)

    print("Federated Learning Training Complete!")


if __name__ == "__main__":
    main()