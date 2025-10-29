import sys, os, yaml, logging, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------
# Project Setup
# --------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'modules'))

from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector as KeypointDetector
from frames_dataset import FramesDataset as VideoDataset

# --------------------------------------------
# Logging
# --------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --------------------------------------------
# Paths
# --------------------------------------------
CONFIG_PATH = "/content/drive/MyDrive/DiffTED_project/TPS/config/ted-384.yaml"
DATASET_ROOT = "/content/drive/MyDrive/DiffTED_project/DiffTED/dataset/frames"
CKPT_DIR = "/content/drive/MyDrive/DiffTED_project/TPS/checkpoints"
PRETRAINED_PATH = "/content/drive/MyDrive/DiffTED_project/DiffTED/ted.pth.tar"

os.makedirs(CKPT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------
# Load Config
# --------------------------------------------
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
params = config["model_params"]
train_params = config["train_params"]

BATCH_SIZE = train_params["batch_size"]
EPOCHS = train_params["num_epochs"]
BASE_LR = train_params["base_lr"]
CKPT_FREQ = train_params["checkpoint_freq"]

# --------------------------------------------
# Dataset
# --------------------------------------------
train_dataset = VideoDataset(DATASET_ROOT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
logging.info(f"[Dataset] Loaded {len(train_dataset)} samples from {DATASET_ROOT}")

# --------------------------------------------
# Initialize Models
# --------------------------------------------
dense_motion_network = DenseMotionNetwork(**params["dense_motion_params"]).to(device)
inpainting_network = InpaintingNetwork(**params["generator_params"]).to(device)
keypoint_detector = KeypointDetector(**params["keypoint_detector_params"]).to(device)

# --------------------------------------------
# Load Pretrained
# --------------------------------------------
if os.path.exists(PRETRAINED_PATH):
    ckpt = torch.load(PRETRAINED_PATH, map_location=device)
    dense_motion_network.load_state_dict(ckpt.get("dense_motion_network", {}), strict=False)
    inpainting_network.load_state_dict(ckpt.get("inpainting_network", {}), strict=False)
    keypoint_detector.load_state_dict(ckpt.get("keypoint_detector", {}), strict=False)
    logging.info(f"Loaded pretrained weights from {PRETRAINED_PATH}")

# --------------------------------------------
# Training Setup
# --------------------------------------------
criterion = nn.L1Loss()
optimizer = optim.AdamW(inpainting_network.parameters(), lr=BASE_LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")

def save_checkpoint(epoch):
    ckpt_path = os.path.join(CKPT_DIR, f"finetune_epoch_{epoch+1}.pth.tar")
    state = {
        "epoch": epoch,
        "dense_motion_network": dense_motion_network.state_dict(),
        "inpainting_network": inpainting_network.state_dict(),
        "keypoint_detector": keypoint_detector.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, ckpt_path)
    torch.save(state, os.path.join(CKPT_DIR, "latest.pth.tar"))
    logging.info(f"✅ Saved checkpoint: {ckpt_path}")

# --------------------------------------------
# Training Loop
# --------------------------------------------
for epoch in range(EPOCHS):
    inpainting_network.train()
    dense_motion_network.eval()
    keypoint_detector.eval()

    running_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        source = batch["source"].to(device)
        driving = batch["driving"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            kp_s = keypoint_detector(source)
            kp_d = keypoint_detector(driving)
            motion = dense_motion_network(source, kp_d, kp_s)
            pred = inpainting_network(source, motion)

            # --- Fixed output extraction ---
            if isinstance(pred, dict):
                if epoch == 0 and batch_idx == 0:
                    logging.info(f"Model output keys: {list(pred.keys())}")
                if "prediction" in pred:
                    pred_img = pred["prediction"]
                elif "generated" in pred:
                    pred_img = pred["generated"]
                else:
                    pred_img = next(v for v in pred.values() if isinstance(v, torch.Tensor) and v.ndim == 4)
            else:
                pred_img = pred

            loss = criterion(pred_img, driving)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1}/{EPOCHS}: Avg Loss = {avg_loss:.6f}")

    if (epoch + 1) % CKPT_FREQ == 0:
        save_checkpoint(epoch)

logging.info("🎯 Fine-tuning complete!")
