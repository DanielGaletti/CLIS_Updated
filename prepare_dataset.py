# prepare_dataset.py
import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
# Caminhos para as pastas originais do PASCAL VOC
VOC_DIR = os.path.join('data/VOCdevkit', 'VOC2012')
IMAGE_DIR = os.path.join(VOC_DIR, 'JPEGImages')
MASK_DIR = os.path.join(VOC_DIR, 'SegmentationClass')

# Pasta de destino para o nosso conjunto de treino
TRAIN_DIR = 'train'

def create_training_dataset():
    """
    Cria uma pasta 'train' com pares de imagens (.jpg) e suas máscaras (.csv).
    Apenas os ficheiros que têm um par correspondente são processados.
    """
    print("--- A preparar o dataset de treino ---")

    # 1. Cria a pasta de treino, limpando-a se já existir
    if os.path.exists(TRAIN_DIR):
        print(f"A limpar a pasta de treino existente: '{TRAIN_DIR}'")
        shutil.rmtree(TRAIN_DIR)
    os.makedirs(TRAIN_DIR)
    print(f"Pasta de treino criada em: '{TRAIN_DIR}'")

    # 2. Encontra todos os ficheiros de máscara disponíveis
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
    
    # Pega o nome base (sem extensão) para encontrar os JPEGs correspondentes
    sample_names = [os.path.splitext(f)[0] for f in mask_files]
    
    print(f"Encontradas {len(sample_names)} máscaras em '{MASK_DIR}'. A verificar os pares de imagens...")

    files_processed = 0
    
    # Usa o tqdm para uma barra de progresso visual
    for sample_name in tqdm(sample_names, desc="A processar imagens"):
        # Define os caminhos de origem
        image_path_src = os.path.join(IMAGE_DIR, sample_name + '.jpg')
        mask_path_src = os.path.join(MASK_DIR, sample_name + '.png')

        # 3. Verifica se a imagem JPEG correspondente realmente existe
        if not os.path.exists(image_path_src):
            # Se não existir, avisa e salta para a próxima
            # print(f"Aviso: Imagem '{image_path_src}' não encontrada para a máscara '{mask_path_src}'. A saltar.")
            continue

        # Define os caminhos de destino na pasta 'train'
        image_path_dest = os.path.join(TRAIN_DIR, sample_name + '.jpg')
        mask_path_dest = os.path.join(TRAIN_DIR, sample_name + '.csv')

        # 4. Copia o ficheiro de imagem
        shutil.copy(image_path_src, image_path_dest)

        # 5. Converte a máscara PNG para CSV
        try:
            mask_image = Image.open(mask_path_src)
            mask_array = np.array(mask_image)
            
            # Salva a matriz como um ficheiro CSV
            np.savetxt(mask_path_dest, mask_array, delimiter=',', fmt='%d')
            files_processed += 1
            
        except Exception as e:
            print(f"ERRO ao processar a máscara '{mask_path_src}': {e}")

    print("\n--- Processo Concluído ---")
    print(f"Foram processados e salvos com sucesso {files_processed} pares de imagem/máscara na pasta '{TRAIN_DIR}'.")

if __name__ == '__main__':
    create_training_dataset()