import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import torchvision
import utilIO

# --- FUNÇÕES AUXILIARES ---
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))['out']
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return mask

def resize_clicks(clicks, orig_h, orig_w, new_h, new_w):
    """Mapeia as coordenadas dos cliques de uma dimensão para outra."""
    scale_r, scale_c = new_h / orig_h, new_w / orig_w
    def _map(pt): return (int(round(pt[0]*scale_r)), int(round(pt[1]*scale_c)))
    return {
        'pos': [_map(p) for p in clicks['pos']],
        'neg': [_map(p) for p in clicks['neg']]
    }

def finetune_on_clicks(model, optimizer, image_tensor, user_clicks, object_class_id, device, iterations=20, teacher_model=None):
    if not user_clicks['pos'] and not user_clicks['neg']:
        print("Nenhum clique para refinar. A saltar o fine-tuning.")
        return

    model.train()
    
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    
    model.apply(set_bn_eval)
    
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    H, W = image_tensor.shape[1], image_tensor.shape[2]
    
    clicks_for_finetune = resize_clicks(user_clicks, user_clicks['orig_h'], user_clicks['orig_w'], H, W)
    
    click_mask = torch.full((H, W), -1, dtype=torch.long)
    
    for p in clicks_for_finetune['pos']: click_mask[p[0], p[1]] = object_class_id
    for p in clicks_for_finetune['neg']: click_mask[p[0], p[1]] = 0 
        
    click_mask = click_mask.to(device)
    image_tensor = image_tensor.to(device)

    criterion_clicks = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_distillation = nn.MSELoss()
    
    lambda_distillation = 0.8 

    print(f"A refinar o modelo com {len(user_clicks['pos'])} cliques positivos e {len(user_clicks['neg'])} negativos...")
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(image_tensor.unsqueeze(0))['out']
        
        loss_clicks = criterion_clicks(outputs, click_mask.unsqueeze(0))
        
        if teacher_model:
            with torch.no_grad():
                teacher_outputs = teacher_model(image_tensor.unsqueeze(0))['out']
            loss_distillation = criterion_distillation(outputs, teacher_outputs)
            total_loss = loss_clicks + lambda_distillation * loss_distillation
        else:
            total_loss = loss_clicks

        total_loss.backward()
        optimizer.step()
    print("Refinamento concluído.")

# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
def run_interactive_session():
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES = 256, 352, 21

    MODEL_PATH = None 
    IMAGE_PATH = 'test/000001.jpg'
    OBJECT_CLASS_ID = 1 
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"A usar o dispositivo: {device}")
    
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
        num_classes=NUM_CLASSES
    ).to(device)
    teacher_model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
        num_classes=NUM_CLASSES
    ).to(device)
    teacher_model.eval()

    print("Modelos DeepLabV3-ResNet50 (aluno e professor) carregados com sucesso.")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=2e-5,
        weight_decay=1e-4
    )

    image_pil = Image.open(IMAGE_PATH).convert('RGB')
    orig_w, orig_h = image_pil.size
    image_tensor = utilIO.prepare_image_for_model(image_pil, IMG_HEIGHT, IMG_WIDTH)
    display_image = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    user_clicks = {'pos': [], 'neg': [], 'orig_h': orig_h, 'orig_w': orig_w} # Armazena as dimensões originais

    while True:
        segmentation_mask = predict(model, image_tensor, device)
        
        clicks_for_display = resize_clicks(user_clicks, user_clicks['orig_h'], user_clicks['orig_w'], IMG_HEIGHT, IMG_WIDTH)

        fig, ax = plt.subplots()
        marked_image = mark_boundaries(np.array(display_image), segmentation_mask)
        ax.imshow(marked_image)
        for click_y, click_x in clicks_for_display['pos']:
            ax.plot(click_x, click_y, 'go', markersize=8)
        for click_y, click_x in clicks_for_display['neg']:
            ax.plot(click_x, click_y, 'ro', markersize=8)

        ax.set_title("Clique (esq=objeto, dir=fundo) e feche a janela para refinar.")
        
        temp_clicks = {'pos': [], 'neg': []}
        def onclick(event):
            if event.xdata is None or event.ydata is None: return
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            
            scale_x, scale_y = user_clicks['orig_w'] / IMG_WIDTH, user_clicks['orig_h'] / IMG_HEIGHT
            orig_x, orig_y = int(round(ix * scale_x)), int(round(iy * scale_y))

            if event.button == 1:
                temp_clicks['pos'].append((orig_y, orig_x)); ax.plot(ix, iy, 'go', markersize=8)
            elif event.button == 3:
                temp_clicks['neg'].append((orig_y, orig_x)); ax.plot(ix, iy, 'ro', markersize=8)
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)
        
        if not temp_clicks['pos'] and not temp_clicks['neg']:
            print("Nenhum clique novo. A terminar a sessão."); break
        
        user_clicks['pos'].extend(temp_clicks['pos'])
        user_clicks['neg'].extend(temp_clicks['neg'])
        
        # Passa os cliques na dimensão original para o fine-tuning
        finetune_on_clicks(model, optimizer, image_tensor, user_clicks, OBJECT_CLASS_ID, device, teacher_model=teacher_model)

if __name__ == '__main__':
    run_interactive_session()