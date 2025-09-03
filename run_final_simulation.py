# run_final_simulation.py
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import torchvision
import utilIO
import pandas as pd
import os
from copy import deepcopy

# --- FUNÇÕES AUXILIARES ---
def get_deeplab_model(num_classes=21):
    """Cria a arquitetura do modelo, mas não carrega pesos pré-treinados da internet."""
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None, # Não baixa os pesos, apenas a arquitetura
        progress=False, 
        num_classes=num_classes,
        aux_loss=True
    )
    return model

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))['out']
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return mask

def finetune_on_clicks(model, optimizer, image_tensor, user_clicks, object_class_id, device, iterations=15):
    # Esta função está correta e não precisa de alterações
    model.train()
    def set_bn_eval(m):
        if isinstance(m, nn.BatchNorm2d): m.eval()
    model.apply(set_bn_eval)
    click_mask = torch.full((image_tensor.shape[1], image_tensor.shape[2]), -1, dtype=torch.long)
    max_y, max_x = click_mask.shape[0] - 1, click_mask.shape[1] - 1
    for p in user_clicks['pos']:
        y = max(0, min(int(p[0]), max_y)); x = max(0, min(int(p[1]), max_x))
        click_mask[y, x] = object_class_id
    for p in user_clicks['neg']:
        y = max(0, min(int(p[0]), max_y)); x = max(0, min(int(p[1]), max_x))
        click_mask[y, x] = 0
    click_mask = click_mask.to(device); image_tensor = image_tensor.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(image_tensor.unsqueeze(0))['out']
        loss = criterion(outputs, click_mask.unsqueeze(0))
        loss.backward()
        optimizer.step()

def calculate_iou(pred_mask, gt_mask, class_id):
    if class_id is None or class_id == 0: return np.nan
    pred_object = (pred_mask == class_id); gt_object = (gt_mask == class_id)
    if np.sum(gt_object) == 0: return np.nan
    intersection = np.logical_and(pred_object, gt_object).sum()
    union = np.logical_or(pred_object, gt_object).sum()
    return intersection / union if union > 0 else 0.0

def run_single_experiment(image_path, gt_mask_path, object_class_id, simulated_clicks, base_model_path=None, model_to_use=None):
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES = 256, 352, 21
    device = torch.device('cpu') # Os experimentos são rápidos e podem rodar na CPU
    
    # *** CORREÇÃO PRINCIPAL: Carrega o SEU modelo treinado ***
    if model_to_use is None:
        model = get_deeplab_model(num_classes=NUM_CLASSES).to(device)
        if base_model_path:
            print(f"INFO: A carregar o modelo base de '{base_model_path}'")
            model.load_state_dict(torch.load(base_model_path, map_location=device))
        else:
            raise ValueError("É necessário um modelo base para o primeiro passo do experimento.")
    else:
        model = model_to_use
        
    image_pil = Image.open(image_path).convert('RGB')
    image_tensor = utilIO.prepare_image_for_model(image_pil, IMG_HEIGHT, IMG_WIDTH)
    gt_mask_pil = Image.open(gt_mask_path).resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
    gt_mask = np.array(gt_mask_pil); gt_mask[gt_mask == 255] = 0
    initial_iou = calculate_iou(predict(model, image_tensor, device), gt_mask, object_class_id)
    
    if simulated_clicks['pos'] or simulated_clicks['neg']:
        if initial_iou is not None and not np.isnan(initial_iou) and initial_iou > 0.85:
            learning_rate = 2e-5 # Refinamento gentil
        else:
            learning_rate = 1e-4 # Refinamento normal
        print(f"INFO: IoU inicial ({initial_iou:.4f}). Usando taxa de aprendizagem: {learning_rate}")
        for param in model.backbone.parameters(): param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        finetune_on_clicks(model, optimizer, image_tensor, simulated_clicks, object_class_id, device)
        
    final_iou = calculate_iou(predict(model, image_tensor, device), gt_mask, object_class_id)
    return initial_iou, final_iou, model

def run_dissertation_experiments(experiments_config, base_model_path):
    results_list = []
    if not os.path.exists('results'): os.makedirs('results')
    for exp_name, config in experiments_config.items():
        print(f"\n===== EXECUTANDO: {exp_name} =====")
        if config['type'] == 'efficiency':
            config['params']['base_model_path'] = base_model_path
            initial_iou, final_iou, _ = run_single_experiment(**config['params'])
            gain = ((final_iou - initial_iou) / initial_iou) * 100 if initial_iou is not None and not np.isnan(initial_iou) and initial_iou > 0 else np.nan
            results_list.append({'Experiment': exp_name, 'IoU_Inicial': initial_iou, 'IoU_Final': final_iou, 'Ganho_%': gain})
            print(f"Resultado: IoU Inicial={initial_iou:.4f}, IoU Final={final_iou:.4f}, Ganho={gain:+.2f}%")
        elif config['type'] == 'continual':
            # O modelo de controle é sempre o seu modelo base
            config['params_control']['base_model_path'] = base_model_path
            iou_control, _, _ = run_single_experiment(**config['params_control'])
            
            # O modelo experimental primeiro aprende com a imagem A
            config['params_exp_part1']['base_model_path'] = base_model_path
            _, _, refined_model = run_single_experiment(**config['params_exp_part1'])
            
            # E depois é testado na imagem B
            iou_experimental, _, _ = run_single_experiment(model_to_use=deepcopy(refined_model), **config['params_exp_part2'])
            
            results_list.append({'Experiment': exp_name, 'IoU_Controle': iou_control, 'IoU_Experimental': iou_experimental})
            print(f"Resultado: IoU Inicial (Controle)={iou_control:.4f}, IoU Inicial (Experimental)={iou_experimental:.4f}")
            
    df = pd.DataFrame(results_list)
    df.to_csv('results/dissertation_final_results.csv', index=False)
    print("\n✅ Experimentos concluídos. Resultados salvos em 'results/dissertation_final_results.csv'")
    print(df)

if __name__ == '__main__':
    # O caminho para o modelo que você acabou de treinar
    BASE_MODEL_PATH = 'auxiliary/m2_deeplab.pth'
    
    EXPERIMENTS = {
        'Exp1_Aviao': {
            'type': 'efficiency',
            'params': {
                # CORREÇÃO: Usando a imagem de avião correta
                'image_path': 'test_data/2007_000033.jpg',
                'gt_mask_path': 'test_data/2007_000033.png',
                'object_class_id': 1, # ID para avião
                'simulated_clicks': {'pos': [(150, 200), (180, 280)], 'neg': [(50, 50), (220, 100)]}
            }
        },
        'Exp2_Continual_Aviao': {
            'type': 'continual',
            'params_control': {
                'image_path': 'test_data/2007_000738.jpg',
                'gt_mask_path': 'test_data/2007_000738.png',
                'object_class_id': 1,
                'simulated_clicks': {'pos':[],'neg':[]}
            },
            'params_exp_part1': {
                'image_path': 'test_data/2007_000033.jpg',
                'gt_mask_path': 'test_data/2007_000033.png',
                'object_class_id': 1,
                'simulated_clicks': {'pos': [(150, 200)], 'neg': [(50, 50)]}
            },
            'params_exp_part2': {
                'image_path': 'test_data/2007_000738.jpg',
                'gt_mask_path': 'test_data/2007_000738.png',
                'object_class_id': 1,
                'simulated_clicks': {'pos':[],'neg':[]}
            }
        },
        'Exp3_Sofa': {
            'type': 'efficiency',
            'params': {
                'image_path': 'test_data/2007_001149.jpg',
                'gt_mask_path': 'test_data/2007_001149.png',
                'object_class_id': 18, # ID para sofá
                'simulated_clicks': {'pos': [(150, 150)], 'neg': [(50, 50)]}
            }
        }
    }
    
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"\nERRO: O modelo treinado '{BASE_MODEL_PATH}' não foi encontrado.")
        print("Por favor, descarregue o modelo treinado do Google Drive e coloque-o na pasta 'auxiliary'.")
    else:
        run_dissertation_experiments(EXPERIMENTS, BASE_MODEL_PATH)