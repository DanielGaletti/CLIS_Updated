# pretrain_deeplabv3_earlystop.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# ---------------- CONFIGURAÇÃO ----------------
DATA_DIR = 'train'
MODEL_SAVE_PATH = 'auxiliary'
MODEL_NAME = 'deeplabv3_resnet50_best.pth'

IMG_HEIGHT = 256
IMG_WIDTH = 352
NUM_CLASSES = 21
BATCH_SIZE = 8
EPOCHS = 100  # máximo de épocas
LEARNING_RATE = 1e-4

PATIENCE = 10  # early stopping: para se não melhorar em 10 épocas
MIN_DELTA = 1e-4  # melhoria mínima para considerar progresso

# ---------------- DATASET ----------------
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
        mask_path = os.path.join(self.data_dir, sample_name + '.csv')

        image = Image.open(img_path).convert('RGB')
        mask = np.loadtxt(mask_path, delimiter=',').astype(np.int32)
        mask[mask == 255] = 0
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Resize
        image = TF.resize(image, (self.height, self.width), interpolation=Image.BILINEAR)
        mask = TF.resize(Image.fromarray(mask.astype(np.uint8)), (self.height, self.width), interpolation=Image.NEAREST)

        # To tensor
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        return image_tensor, mask_tensor

# ---------------- MODELO ----------------
def get_deeplabv3_model(num_classes):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

# ---------------- TREINO ----------------
def train_model():
    print("--- Treino DeepLabV3-ResNet50 com Early Stopping ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    dataset = VOCDataset(data_dir=DATA_DIR, height=IMG_HEIGHT, width=IMG_WIDTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Dados carregados: {len(dataset)} amostras.")

    model = get_deeplabv3_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)

        # Early stopping
        if best_loss - epoch_loss > MIN_DELTA:
            best_loss = epoch_loss
            epochs_no_improve = 0
            # Salvar checkpoint
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
            print(f"Melhor modelo salvo com loss {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping ativado após {epoch+1} épocas sem melhoria.")
                break

    print("\nTreino finalizado!")

if __name__ == '__main__':
    train_model()
