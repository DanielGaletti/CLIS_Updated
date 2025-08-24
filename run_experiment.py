import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import torchvision
import utilIO

# ----------------------------
# Funções auxiliares
# ----------------------------
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))['out']
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return mask

def resize_clicks(clicks, orig_h, orig_w, new_h, new_w):
    scale_r, scale_c = new_h / orig_h, new_w / orig_w
    def _map(pt): return (int(round(pt[0]*scale_r)), int(round(pt[1]*scale_c)))
    return {
        'pos': [_map(p) for p in clicks['pos']],
        'neg': [_map(p) for p in clicks['neg']]
    }

def finetune_on_clicks(model, optimizer, image_tensor, user_clicks, object_class_id, device, iterations=10):
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
    click_mask = torch.full((H, W), -1, dtype=torch.long)

    for p in user_clicks['pos']:
        click_mask[p[0], p[1]] = object_class_id
    for p in user_clicks['neg']:
        click_mask[p[0], p[1]] = 0

    click_mask = click_mask.to(device)
    image_tensor = image_tensor.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(image_tensor.unsqueeze(0))['out']
        loss = criterion(outputs, click_mask.unsqueeze(0))
        loss.backward()
        optimizer.step()

def calculate_iou(pred_mask, gt_mask, class_id):
    valid = (gt_mask != 255)
    pred_object = (pred_mask == class_id) & valid
    gt_object   = (gt_mask == class_id) & valid

    intersection = np.logical_and(pred_object, gt_object).sum()
    union = np.logical_or(pred_object, gt_object).sum()

    if union == 0:
        return None
    return intersection / union

def evaluate_all_classes(pred_mask, gt_mask, num_classes=21):
    results = {}
    for class_id in range(num_classes):
        iou = calculate_iou(pred_mask, gt_mask, class_id)
        if iou is not None:
            results[class_id] = iou
    return results

def find_simulated_clicks(initial_mask, gt_mask, object_class_id, num_clicks=15):
    """
    Encontra cliques positivos e negativos comparando a máscara inicial com a ground truth.
    Retorna uma lista de cliques para simulação.
    """
    false_negatives = np.where(
        (gt_mask == object_class_id) & (initial_mask != object_class_id)
    )

    false_positives = np.where(
        (initial_mask == object_class_id) & (gt_mask != object_class_id) & (gt_mask != 255)
    )
    
    pos_clicks = []
    if len(false_negatives[0]) > 0:
        indices = np.random.choice(len(false_negatives[0]), min(len(false_negatives[0]), num_clicks), replace=False)
        for i in indices:
            pos_clicks.append((false_negatives[0][i], false_negatives[1][i]))

    neg_clicks = []
    if len(false_positives[0]) > 0:
        indices = np.random.choice(len(false_positives[0]), min(len(false_positives[0]), num_clicks), replace=False)
        for i in indices:
            neg_clicks.append((false_positives[0][i], false_positives[1][i]))

    return {'pos': pos_clicks, 'neg': neg_clicks}


# ----------------------------
# Função principal
# ----------------------------
def run_single_experiment(image_path, gt_mask_path, object_class_id):
    print(f"--- A EXECUTAR EXPERIMENTO PARA: {image_path} ---")

    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES = 256, 352, 21
    MODEL_PATH = 'auxiliary/m2_deeplab.pth'

    device = torch.device('cpu')
    
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        num_classes=NUM_CLASSES
    ).to(device)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict)

    image_pil = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image_pil.size
    image_tensor = utilIO.prepare_image_for_model(image_pil, IMG_HEIGHT, IMG_WIDTH)

    gt_mask_pil = Image.open(gt_mask_path)
    gt_mask_pil = gt_mask_pil.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
    gt_mask = np.array(gt_mask_pil)
    print("Valores únicos no GT:", np.unique(gt_mask)) 

    initial_mask = predict(model, image_tensor, device)
    initial_iou = calculate_iou(initial_mask, gt_mask, object_class_id)
    print("Predição inicial:", np.unique(initial_mask))
    print(f"IoU Inicial para a classe {object_class_id}: {initial_iou:.4f}")

    simulated_clicks = find_simulated_clicks(initial_mask, gt_mask, object_class_id, num_clicks=5)
    
    if not simulated_clicks['pos'] and not simulated_clicks['neg']:
        print("\nSem erros de segmentação encontrados. Nenhum refinamento necessário.")
        final_iou = initial_iou
        final_mask = initial_mask
    else:
        clicks_resized = resize_clicks(simulated_clicks, orig_h, orig_w, IMG_HEIGHT, IMG_WIDTH)
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-5)
        
        finetune_on_clicks(model, optimizer, image_tensor, clicks_resized, object_class_id, device)

        final_mask = predict(model, image_tensor, device)
        final_iou = calculate_iou(final_mask, gt_mask, object_class_id)

    print("\n--- RELATÓRIO DO EXPERIMENTO ---")
    print(f"Imagem: {image_path}")
    print(f"IoU Inicial: {initial_iou:.4f}")
    print(f"IoU Final:   {final_iou:.4f}")
    if initial_iou is not None and initial_iou > 0:
        gain = ((final_iou - initial_iou) / initial_iou) * 100
        print(f"Ganho de Performance: {gain:+.2f}%")
        
    print("\n--- Avaliação por classe ---")
    initial_results = evaluate_all_classes(initial_mask, gt_mask)
    final_results = evaluate_all_classes(final_mask, gt_mask)

    for cls, iou in final_results.items():
        initial_iou_cls = initial_results.get(cls, 0.0)
        final_iou_cls = final_results[cls]
        gain = ((final_iou_cls - initial_iou_cls) / initial_iou_cls * 100) if initial_iou_cls > 0 else 0.0
        print(f"Classe {cls:2d} | Inicial: {initial_iou_cls:.4f} | Final: {final_iou_cls:.4f} | Δ {gain:+.2f}%")

    print("--------------------------------\n")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(mark_boundaries(np.array(image_pil.resize((IMG_WIDTH, IMG_HEIGHT))), initial_mask))
    axes[0].set_title(f'Inicial (IoU: {initial_iou:.4f})')
    axes[1].imshow(mark_boundaries(np.array(image_pil.resize((IMG_WIDTH, IMG_HEIGHT))), final_mask))
    axes[1].set_title(f'Final (IoU: {final_iou:.4f})')
    axes[2].imshow(gt_mask, cmap='gray')
    axes[2].set_title('Ground Truth')
    plt.show()

if __name__ == '__main__':
    TEST_IMAGE_PATH = 'test_data/2007_000256.jpg'
    TEST_GT_MASK_PATH = 'test_data/2007_000256.png'
    OBJECT_CLASS_ID = 0

    run_single_experiment(TEST_IMAGE_PATH, TEST_GT_MASK_PATH, OBJECT_CLASS_ID)
