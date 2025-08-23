# utilIO.py (Versão Final e Limpa para PyTorch)

from PIL import Image
import torchvision.transforms.functional as TF

# A única função de que o nosso novo projeto precisa.
# Removemos todo o código antigo que dependia de TensorFlow/Keras.
def prepare_image_for_model(image_pil, height, width):
    """Prepara uma imagem PIL para ser usada pelo modelo PyTorch."""
    # Redimensionar
    image = TF.resize(image_pil, (height, width))
    # Converter para Tensor
    image_tensor = TF.to_tensor(image)
    # Normalizar (usando os mesmos valores do pré-treino)
    image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image_tensor

# As outras funções como getTrainData, etc., foram removidas
# porque a sua funcionalidade já está dentro dos nossos novos scripts PyTorch.