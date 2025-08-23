# utilInteract.py (Versão Final e Limpa para PyTorch/NumPy)

import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops

def get_superpixels(image_np, sp_num, sp_compactness):
    """Calcula superpíxeis numa imagem."""
    return slic(image_np, n_segments=sp_num, compactness=sp_compactness, start_label=1)

def refine_mask_with_superpixels(mask_np, superpixels):
    """
    Refina uma máscara de segmentação garantindo que as regiões de superpíxeis
    sejam consistentemente rotuladas como objeto ou fundo.
    """
    refined_mask = np.copy(mask_np)
    regions = regionprops(superpixels)

    for props in regions:
        # Encontra as coordenadas de todos os píxeis neste superpíxel
        coords = props.coords
        
        # Obtém os rótulos da máscara de previsão para estes píxeis
        labels_in_sp = mask_np[coords[:, 0], coords[:, 1]]
        
        # Encontra o rótulo mais comum (moda)
        if len(labels_in_sp) > 0:
            most_common_label = np.bincount(labels_in_sp).argmax()
            
            # Pinta todo o superpíxel com o rótulo mais comum
            refined_mask[coords[:, 0], coords[:, 1]] = most_common_label
            
    return refined_mask

# O código antigo que usava TensorFlow foi removido e substituído por
# lógica pura de NumPy e scikit-image, que são compatíveis com o nosso
# novo ambiente PyTorch.