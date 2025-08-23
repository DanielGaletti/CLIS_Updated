import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

VOC_IMAGES_PATH = os.path.join('data', 'VOCdevkit', 'VOC2012', 'JPEGImages')

VOC_MASKS_PATH = os.path.join('data', 'VOCdevkit', 'VOC2012', 'SegmentationClass')

TRAIN_LIST_FILE = os.path.join('data', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt')

OUTPUT_TRAIN_PATH = 'train'


def create_training_data():
    """
    Lê a lista de treino do PASCAL VOC, copia as imagens .jpg correspondentes
    e converte as máscaras de segmentação .png para o formato .csv.
    """
    print("--- Iniciando a preparação dos dados de treino ---")

    print(f"Lendo a lista de ficheiros de: {TRAIN_LIST_FILE}")
    try:
        with open(TRAIN_LIST_FILE, 'r') as f:
            train_filenames = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"ERRO: O ficheiro de lista '{TRAIN_LIST_FILE}' não foi encontrado.")
        print("Verifique se o seu dataset PASCAL VOC está na pasta 'data/'.")
        return

    print(f"Encontrados {len(train_filenames)} ficheiros no conjunto de treino.")

    if not os.path.exists(OUTPUT_TRAIN_PATH):
        print(f"Criando a pasta de destino: '{OUTPUT_TRAIN_PATH}'")
        os.makedirs(OUTPUT_TRAIN_PATH)

    files_copied = 0
    files_converted = 0

    for filename_base in tqdm(train_filenames, desc="Processando ficheiros"):
        
        jpg_source_path = os.path.join(VOC_IMAGES_PATH, filename_base + '.jpg')
        jpg_dest_path = os.path.join(OUTPUT_TRAIN_PATH, filename_base + '.jpg')
        
        if os.path.exists(jpg_source_path):
            shutil.copyfile(jpg_source_path, jpg_dest_path)
            files_copied += 1
        else:
            print(f"Aviso: Imagem não encontrada: {jpg_source_path}")
            continue 

        png_source_path = os.path.join(VOC_MASKS_PATH, filename_base + '.png')
        csv_dest_path = os.path.join(OUTPUT_TRAIN_PATH, filename_base + '.csv')

        if os.path.exists(png_source_path):
            mask_image = Image.open(png_source_path)
            mask_array = np.array(mask_image.convert('L'), dtype=np.int8)
            
            np.savetxt(csv_dest_path, mask_array, fmt='%d', delimiter=',')
            files_converted += 1
        else:
            print(f"Aviso: Máscara de segmentação não encontrada: {png_source_path}")

    print("\n--- Preparação Concluída! ---")
    print(f"Total de imagens (.jpg) copiadas: {files_copied}")
    print(f"Total de máscaras (.csv) convertidas: {files_converted}")
    print(f"A sua pasta '{OUTPUT_TRAIN_PATH}' está pronta para o treino.")

if __name__ == '__main__':
    create_training_data()