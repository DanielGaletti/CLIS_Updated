# pretrain_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = 'train'
MODEL_SAVE_PATH = 'auxiliary'
MODEL_NAME = 'm2.pth'

IMG_HEIGHT = 256
IMG_WIDTH = 352
NUM_CLASSES = 21
BATCH_SIZE = 16 
EPOCHS = 50
LEARNING_RATE = 1e-4

# --- 2. CARREGADOR DE DADOS (DATASET) ---
class VOCDataset(Dataset):
    def __init__(self, data_dir, height, width):
        self.data_dir = data_dir
        self.height = height
        self.width = width
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.samples = [os.path.splitext(f)[0] for f in image_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        img_path = os.path.join(self.data_dir, sample_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.data_dir, sample_name + '.csv')
        mask = np.loadtxt(mask_path, delimiter=',').astype(np.int32)
        mask[mask == 255] = 0 # Ignorar o rótulo 255
        mask = np.clip(mask, 0, NUM_CLASSES - 1) # Garantir que todos os rótulos são válidos
        mask = Image.fromarray(mask.astype(np.uint8))

        image = TF.resize(image, (self.height, self.width), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (self.height, self.width), interpolation=Image.NEAREST)

        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image_tensor, mask_tensor

# --- 3. ARQUITETURA DO MODELO (U-NET) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1); diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1); return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(); self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64); self.down1 = Down(64, 128); self.down2 = Down(128, 256)
        self.down3 = Down(256, 512); self.down4 = Down(512, 1024); self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256); self.up3 = Up(256, 128); self.up4 = Up(128, 64); self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1); return self.outc(x)

# --- 4. FUNÇÃO DE TREINO ---
def train_model():
    print("--- Iniciando o Pré-treino com PyTorch e CUDA ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando o dispositivo: {device}")

    dataset = VOCDataset(data_dir=DATA_DIR, height=IMG_HEIGHT, width=IMG_WIDTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Dados carregados: {len(dataset)} amostras.")

    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Iniciando o treino...")
    for epoch in range(EPOCHS):
        model.train(); running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, masks)
            loss.backward(); optimizer.step(); running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

    print("\nTreino concluído!")
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)
    final_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    torch.save(model.state_dict(), final_model_path)
    print(f"Modelo salvo com sucesso em: {final_model_path}")

if __name__ == '__main__':
    train_model()