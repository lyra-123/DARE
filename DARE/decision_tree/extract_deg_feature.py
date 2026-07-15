import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
from ultralytics import YOLO
import warnings
import csv
import os
import h5py

warnings.filterwarnings('ignore')
csv_files = ['/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc_info/DETRAC_desc.csv',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc_info/DSEC_desc.csv',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc_info/LMOT_desc.csv',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc_info/D²-City_desc.csv',]
dataset_dir = ['/mnt/mydisk/lyra/RL_Dataset/DETRAC/images',
                '/mnt/mydisk/lyra/RL_Dataset/DSEC/images',
                '/mnt/mydisk/lyra/RL_Dataset/LMOT/images',
                '/mnt/mydisk/lyra/RL_Dataset/D²-City/images',]


def load_video_indices(csv_file):
    indices = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            indices.append(int(row[0]))
    return indices

def get_selected_video_dirs(sub_dataset_dir, csv_file):
    all_videos = sorted(os.listdir(sub_dataset_dir))
    selected_indices = load_video_indices(csv_file)

    selected_dirs = []
    for new_id, old_id in enumerate(selected_indices):
        if old_id >= len(all_videos):
            raise IndexError(
                f"CSV index {old_id} out of range for {sub_dataset_dir}"
            )
        video_name = all_videos[old_id]
        video_path = os.path.join(sub_dataset_dir, video_name)
        selected_dirs.append((new_id, video_path))

    return selected_dirs

def write_features_to_h5(h5_path, video_features_dict):
    """
    video_features_dict:
        {
          0: [num_chunks, C],
          1: [num_chunks, C],
          ...
        }
    """
    with h5py.File(h5_path, 'w') as f:
        for vid, feats in video_features_dict.items():
            grp = f.create_group(f'{vid}')
            grp.create_dataset(
                'features',
                data=feats,
                compression='gzip'
            )

class CPAFeatureExtractor:
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
        self.model = yolo_model.model

        self.features = {
            'layer1': None,
            'layer2': None,
            'layer3': None
        }
        self.hooks = []

    def register_hooks(self):
        print("\n" + "=" * 60)
        print("Registering hooks to CPA modules...")
        print("=" * 60)

        hook_registered = False

        for name, module in self.model.named_modules():
            if name == 'model.0.prompt1':
                hook = module.register_forward_hook(self._get_hook('layer1'))
                self.hooks.append(hook)
                print(f"✓ Registered hook: {name} -> layer1 (dim=8)")
                hook_registered = True

            elif name == 'model.0.prompt2':
                hook = module.register_forward_hook(self._get_hook('layer2'))
                self.hooks.append(hook)
                print(f"✓ Registered hook: {name} -> layer2 (dim=16)")
                hook_registered = True

            elif name == 'model.0.prompt3':
                hook = module.register_forward_hook(self._get_hook('layer3'))
                self.hooks.append(hook)
                print(f"✓ Registered hook: {name} -> layer3 (dim=32)")
                hook_registered = True

        if not hook_registered:
            raise RuntimeError("Failed to register any hooks! Check model structure.")

        print(f"\n✓ Total {len(self.hooks)} hooks registered successfully!")
        return True

    def _get_hook(self, layer_name):
        """创建hook函数"""

        def hook(module, input, output):
            # 保存输出特征
            self.features[layer_name] = output.detach().cpu()
            # print(f"  [{layer_name}] Captured feature shape: {output.shape}")

        return hook

    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("\n✓ All hooks removed")

    def get_feature(self, layer_name):
        """获取指定层的特征"""
        return self.features.get(layer_name, None)


# ============ 数据集 ============
class DegradationDataset(Dataset):
    """退化图像数据集"""

    def __init__(self, image_dir, img_size=640, fps=10, chunk_seconds=2):
        self.image_dir = Path(image_dir)
        self.img_size = img_size

        self.fps = fps
        self.chunk_size = fps * chunk_seconds

        self.images = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.JPG', '*.PNG']:
            self.images.extend(list(self.image_dir.glob(ext)))
        self.images = sorted(self.images)

        self.num_chunks = len(self.images) // self.chunk_size
        if self.num_chunks == 0:
            raise ValueError(f"Not enough frames in {self.image_dir}")

    def __len__(self):
        return self.num_chunks

    def letterbox(self, image, new_shape=(640, 640), color=(114, 114, 114)):
        h0, w0 = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / h0, new_shape[1] / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        top = round(dh - 0.1)
        bottom = round(dh + 0.1)
        left = round(dw - 0.1)
        right = round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

    def __getitem__(self, idx):
        img_path = self.images[idx * self.chunk_size]

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        img = self.letterbox(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        return img


def extract_features_from_layer(yolo_model, dataloader, layer_name='layer1', device='cuda'):
    model = yolo_model.model
    model.eval()
    model = model.to(device)

    extractor = CPAFeatureExtractor(yolo_model)
    extractor.register_hooks()

    all_features = []

    print(f"\nExtracting features from {layer_name}...")
    print("-" * 60)

    with torch.no_grad():
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)

            try:
                _ = model(images)
            except Exception as e:
                print(f"\n⚠ Warning at batch {batch_idx}: {e}")
                continue

            features = extractor.get_feature(layer_name)

            if features is None:
                print(f"\n⚠ Warning: No features at batch {batch_idx}")
                continue

            if len(features.shape) == 4:
                features = torch.mean(features, dim=[2, 3])
            elif len(features.shape) == 3:
                features = torch.mean(features, dim=2)

            all_features.append(features.numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"  Progress: {batch_idx + 1:3d}/{len(dataloader):3d} batches", end='\r')

    print(f"  Progress: {len(dataloader):3d}/{len(dataloader):3d} batches - Done!  ")

    extractor.remove_hooks()

    if not all_features:
        raise RuntimeError("No features were extracted!")

    all_features = np.concatenate(all_features, axis=0)

    print(f"  Extracted features shape: {all_features.shape}")

    return all_features

def main():
    model_path = '/home/ubuntu/lyra/MPC/bifpn_cpa+x/bifpn_cpa+x.pt'  # 修改为你的模型路径
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 数据配置
    img_size = 640
    batch_size = 8
    dataset_fps = [25, 20, 20, 25]
    for idx, (csv_file, dataset_root) in enumerate(zip(csv_files, dataset_dir)):
        # dataset_name = Path(dataset_root).name
        dataset_name = os.path.basename(csv_file).replace('_desc.csv', '')
        if dataset_name == 'DETRAC' or dataset_name == 'DSEC' or dataset_name == 'LMOT':
            start_idx = 0
        else:
            start_idx = -55
        print(f'\nProcessing dataset: {dataset_name}')

        selected_videos = get_selected_video_dirs(dataset_root, csv_file)[start_idx:]

        video_features = {}

        for new_vid, video_path in selected_videos:
            print(f'  Video {new_vid}: {video_path}')

            dataset = DegradationDataset(
                image_dir=video_path,
                img_size=img_size,
                fps=dataset_fps[idx],
                chunk_seconds=2,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=False
            )

            feats = extract_features_from_layer(
                model,
                dataloader,
                layer_name='layer1',
                device=device
            )

            # feats.shape = [num_chunks, C]
            video_features[new_vid] = feats

        h5_path = f'{dataset_name}_layer1.h5'
        write_features_to_h5(h5_path, video_features)

        print(f'✓ Saved to {h5_path}')


if __name__ == '__main__':
    main()