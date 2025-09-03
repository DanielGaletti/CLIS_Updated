# main_interactive.py
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import torchvision
import utilIO
from copy import deepcopy

# --- FUNÇÕES AUXILIARES ---
def get_deeplab_model(num_classes=21):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None, progress=False, num_classes=num_classes, aux_loss=True
    )
    return model

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))['out']
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return mask

def resize_clicks(clicks, orig_h, orig_w, new_h, new_w):
    scale_r, scale_c = new_h / orig_h, new_w / orig_w
    def _map(pt): return (int(round(pt[0]*scale_r)), int(round(pt[1]*scale_c)))
    return {'pos': [_map(p) for p in clicks['pos']], 'neg': [_map(p) for p in clicks['neg']]}

def finetune_on_clicks(model, optimizer, image_tensor, user_clicks, object_class_id, device, iterations=15, teacher_model=None):
    if not user_clicks['pos'] and not user_clicks['neg']:
        print("Nenhum clique para refinar.")
        return
        
    model.train()
    def set_bn_eval(m):
        if isinstance(m, nn.BatchNorm2d): m.eval()
    model.apply(set_bn_eval)

    H, W = image_tensor.shape[1], image_tensor.shape[2]
    clicks_for_finetune = resize_clicks(user_clicks, user_clicks['orig_h'], user_clicks['orig_w'], H, W)

    # Máscara para a loss dos cliques (como antes)
    click_mask = torch.full((H, W), -1, dtype=torch.long, device=device)
    all_clicks = []
    for p in clicks_for_finetune['pos']:
        click_mask[p[0], p[1]] = object_class_id
        all_clicks.append(p)
    for p in clicks_for_finetune['neg']:
        click_mask[p[0], p[1]] = 0
        all_clicks.append(p)

    # *** NOVO: Criar a Máscara de Destilação Focada ***
    # Começa por dar peso total à orientação do professor em toda a imagem
    distillation_mask = torch.ones_like(image_tensor, device=device, dtype=torch.float32)[0:1, :, :]
    
    # Define o tamanho da "zona de silêncio" à volta de cada clique
    click_region_size = 15 
    
    # Para cada clique, reduz o peso da destilação nessa área
    for y, x in all_clicks:
        y_min = max(0, y - click_region_size // 2)
        y_max = min(H, y + click_region_size // 2 + 1)
        x_min = max(0, x - click_region_size // 2)
        x_max = min(W, x + click_region_size // 2 + 1)
        
        # O professor "ficará em silêncio" (peso baixo) nesta região
        distillation_mask[:, y_min:y_max, x_min:x_max] = 0.1 

    image_tensor = image_tensor.to(device)
    criterion_clicks = nn.CrossEntropyLoss(ignore_index=-1)
    
    lambda_distillation = 10.0
    
    print(f"A refinar o modelo com {len(user_clicks['pos'])} cliques positivos e {len(user_clicks['neg'])} negativos...")
    
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(image_tensor.unsqueeze(0))['out']
        loss_clicks = criterion_clicks(outputs, click_mask.unsqueeze(0))
        
        with torch.no_grad():
            teacher_outputs = teacher_model(image_tensor.unsqueeze(0))['out']
        
        # *** NOVO: Calcular a loss de destilação usando a máscara ***
        # Calculamos a diferença de quadrados (MSE) e aplicamos a nossa máscara de pesos
        squared_diff = (outputs - teacher_outputs) ** 2
        loss_distillation = torch.mean(distillation_mask * squared_diff)

        total_loss = loss_clicks + lambda_distillation * loss_distillation
        
        total_loss.backward()
        optimizer.step()
        
    print("Refinamento concluído.")

def run_interactive_session():
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES = 256, 352, 21
    BASE_MODEL_PATH = 'auxiliary/m2_deeplab.pth'
    
    IMAGE_SEQUENCE = [
        {'path': 'test_data/2007_001149.jpg', 'class_id': 18}, 
        {'path': 'test_data/2007_001154.jpg', 'class_id': 18},
        {'path': 'test_data/2007_000033.jpg', 'class_id': 1}, 
    ]
    
    device = torch.device('cuda')
    print(f"A usar o dispositivo: {device}")
    
    model = get_deeplab_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    
    teacher_model = deepcopy(model)
    teacher_model.eval()

    print("Modelo 'm2_deeplab.pth' (aluno e professor) carregado com sucesso.")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=5e-6, 
        weight_decay=1e-4
    )

    for i, image_info in enumerate(IMAGE_SEQUENCE):
        print(f"\n--- A processar imagem {i+1}: {image_info['path']} ---")
        image_path = image_info['path']
        object_class_id = image_info['class_id']

        try:
            image_pil = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERRO: A imagem '{image_path}' não foi encontrada. Verifique a pasta 'test_data'.")
            continue

        orig_w, orig_h = image_pil.size
        image_tensor = utilIO.prepare_image_for_model(image_pil, IMG_HEIGHT, IMG_WIDTH)
        display_image = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
        user_clicks = {'pos': [], 'neg': [], 'orig_h': orig_h, 'orig_w': orig_w}

        while True:
            segmentation_mask = predict(model, image_tensor, device)
            clicks_for_display = resize_clicks(user_clicks, user_clicks['orig_h'], user_clicks['orig_w'], IMG_HEIGHT, IMG_WIDTH)

            fig, ax = plt.subplots()
            ax.imshow(mark_boundaries(np.array(display_image), segmentation_mask))
            for click_y, click_x in clicks_for_display['pos']:
                ax.plot(click_x, click_y, 'go', markersize=8)
            for click_y, click_x in clicks_for_display['neg']:
                ax.plot(click_x, click_y, 'ro', markersize=8)
            ax.set_title("Clique (esq=objeto, dir=fundo) e feche para refinar. Feche sem clicar para ir para a próxima imagem.")
            
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
                print("A avançar para a próxima imagem..."); break
            
            user_clicks['pos'].extend(temp_clicks['pos'])
            user_clicks['neg'].extend(temp_clicks['neg'])
            
            finetune_on_clicks(model, optimizer, image_tensor, user_clicks, object_class_id, device, teacher_model=teacher_model)

if __name__ == '__main__':
    run_interactive_session()