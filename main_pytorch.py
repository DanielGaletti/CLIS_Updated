# main_pytorch.py (v3 - Aprendizagem Agressiva)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import utilIO

# --- ARQUITETURA DO MODELO (Sem alterações) ---
# ... (todo o código da U-Net fica aqui, exatamente como antes) ...
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


# --- FUNÇÕES DE MODELO (Com iterações aumentadas) ---
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return mask

def finetune_on_clicks(model, optimizer, image_tensor, user_clicks, device, iterations=20): # AUMENTADO para 20
    if not user_clicks['pos'] and not user_clicks['neg']:
        print("Nenhum clique para refinar. A saltar o fine-tuning.")
        return
        
    model.train()
    click_mask = torch.full((image_tensor.shape[1], image_tensor.shape[2]), -1, dtype=torch.long)
    
    for p in user_clicks['pos']: click_mask[p[0], p[1]] = 1
    for p in user_clicks['neg']: click_mask[p[0], p[1]] = 0
        
    click_mask = click_mask.to(device)
    image_tensor = image_tensor.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print(f"A refinar o modelo com {len(user_clicks['pos'])} cliques positivos e {len(user_clicks['neg'])} negativos...")
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(image_tensor.unsqueeze(0))
        loss = criterion(outputs, click_mask.unsqueeze(0))
        loss.backward()
        optimizer.step()
    print("Refinamento concluído.")

# --- LÓGICA PRINCIPAL DA APLICAÇÃO (Com taxa de aprendizagem aumentada) ---
def run_interactive_session():
    # Configurações
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES = 256, 352, 21
    MODEL_PATH = 'auxiliary/m2.pth'
    IMAGE_PATH = 'img/2007_000033.jpg'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"A usar o dispositivo: {device}")
    
    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Modelo m2.pth carregado com sucesso.")

    # --- A CORREÇÃO PRINCIPAL ESTÁ AQUI ---
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # AUMENTADO DE 1e-5 PARA 1e-4 (10x maior)

    image_pil = Image.open(IMAGE_PATH).convert('RGB')
    image_tensor = utilIO.prepare_image_for_model(image_pil, IMG_HEIGHT, IMG_WIDTH)
    display_image = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    user_clicks = {'pos': [], 'neg': []}

    while True:
        segmentation_mask = predict(model, image_tensor, device)
        
        fig, ax = plt.subplots(); ax.imshow(mark_boundaries(np.array(display_image), segmentation_mask))
        ax.set_title("Clique (esq=objeto, dir=fundo) e feche a janela para refinar.")
        
        temp_clicks = {'pos': [], 'neg': []}
        def onclick(event):
            if event.xdata is None or event.ydata is None: return
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            if event.button == 1:
                temp_clicks['pos'].append((iy, ix)); ax.plot(ix, iy, 'go', markersize=8)
            elif event.button == 3:
                temp_clicks['neg'].append((iy, ix)); ax.plot(ix, iy, 'ro', markersize=8)
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)
        
        if not temp_clicks['pos'] and not temp_clicks['neg']:
            print("Nenhum clique novo. A terminar a sessão."); break
        
        user_clicks['pos'].extend(temp_clicks['pos'])
        user_clicks['neg'].extend(temp_clicks['neg'])
        
        finetune_on_clicks(model, optimizer, image_tensor, user_clicks, device)

if __name__ == '__main__':
    run_interactive_session()