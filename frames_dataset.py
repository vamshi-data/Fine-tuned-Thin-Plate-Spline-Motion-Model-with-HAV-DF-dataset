import os
import glob
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from skimage import io, img_as_float32
import cv2


class FramesDataset(Dataset):
    """
    Loads data for TPS-based motion models.
    Works for both:
      - Directories of frames (/dataset/frames/person_001/0001.png ...)
      - Single .mp4 videos
    Returns: dict {"source": Tensor, "driving": Tensor}
    """

    def __init__(self, root_dir, frame_shape=(384, 384, 3), id_sampling=True, max_frames=64, **kwargs):
        self.root_dir = root_dir
        self.frame_shape = frame_shape
        self.id_sampling = id_sampling
        self.max_frames = max_frames

        # Collect videos and frame folders
        self.video_list = glob.glob(os.path.join(root_dir, "*.mp4"))
        self.folder_list = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
        self.names = [os.path.splitext(os.path.basename(v))[0] for v in self.video_list] + [
            os.path.basename(f) for f in self.folder_list
        ]

        print(f"[Dataset] Found {len(self.video_list)} videos and {len(self.folder_list)} frame folders.")
        if len(self.names) == 0:
            raise RuntimeError(f"No valid data found in {root_dir}. Expecting .mp4 or frame folders.")

    def __len__(self):
        return len(self.names)

    def _read_video_frames(self, video_path):
        """Reads frames from a .mp4 file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_shape[1], self.frame_shape[0]))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Empty video: {video_path}")

        # Sample uniformly
        idx = np.linspace(0, len(frames) - 1, num=min(self.max_frames, len(frames))).astype(int)
        frames = [frames[i] for i in idx]
        return np.stack(frames, axis=0)

    def _read_image_folder(self, folder_path):
        """Reads frames from an image directory"""
        images = sorted(glob.glob(os.path.join(folder_path, "*.png")) +
                        glob.glob(os.path.join(folder_path, "*.jpg")) +
                        glob.glob(os.path.join(folder_path, "*.jpeg")))
        if len(images) == 0:
            raise ValueError(f"No images found in folder: {folder_path}")

        # Limit max frames
        images = images[:self.max_frames]
        frames = []
        for img_path in images:
            img = io.imread(img_path)
            if img.ndim == 2:  # grayscale
                img = np.stack([img] * 3, axis=-1)
            img = cv2.resize(img, (self.frame_shape[1], self.frame_shape[0]))
            frames.append(img)
        return np.stack(frames, axis=0)

    def __getitem__(self, idx):
        """Return source and driving frame tensors"""
        name = self.names[idx]
        video_path = os.path.join(self.root_dir, name + ".mp4")
        folder_path = os.path.join(self.root_dir, name)

        if os.path.exists(video_path):
            frames = self._read_video_frames(video_path)
        elif os.path.isdir(folder_path):
            frames = self._read_image_folder(folder_path)
        else:
            raise FileNotFoundError(f"No .mp4 or frame folder found for {name}")

        if frames.shape[0] < 2:
            raise ValueError(f"Not enough frames for sample {name}")

        # Pick source & driving frames
        source_idx, driving_idx = random.sample(range(frames.shape[0]), 2)
        source = img_as_float32(frames[source_idx].transpose((2, 0, 1)))
        driving = img_as_float32(frames[driving_idx].transpose((2, 0, 1)))

        source = torch.from_numpy(source).float()
        driving = torch.from_numpy(driving).float()

        return {"source": source, "driving": driving, "name": name}


# ✅ Optional repeater (for longer epochs)
class DatasetRepeater(Dataset):
    def __init__(self, dataset_params, times=1):
        self.dataset = FramesDataset(**dataset_params)
        self.times = times

    def __len__(self):
        return len(self.dataset) * self.times

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
