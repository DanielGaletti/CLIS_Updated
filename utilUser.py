# utilUser.py (versão moderna)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get_user_clicks(image_pil, current_mask):
    """
    Mostra uma imagem e recolhe os cliques do utilizador.
    Clique esquerdo para pontos positivos (objeto).
    Clique direito para pontos negativos (fundo).
    Fecha a janela para terminar a recolha de cliques.
    """
    print("\n--- Interação com o Utilizador ---")
    print("Na janela que vai abrir:")
    print("  - Clique com o botão ESQUERDO para marcar o OBJETO.")
    print("  - Clique com o botão DIREITO para marcar o FUNDO.")
    print("  - FECHE a janela para o modelo aprender com os seus cliques.")

    fig, ax = plt.subplots()
    ax.imshow(image_pil)
    ax.set_title("Clique para corrigir a segmentação")
    
    clicks = {'pos': [], 'neg': []}

    def onclick(event):
        # matplotlib (y, x), mas nós trabalhamos com (linha, coluna), que é (y, x)
        ix, iy = int(round(event.xdata)), int(round(event.ydata))

        # Botão esquerdo = 1 (positivo), Botão direito = 3 (negativo)
        if event.button == 1:
            clicks['pos'].append((iy, ix))
            print(f"Clique positivo adicionado em: (linha={iy}, coluna={ix})")
            ax.plot(ix, iy, 'go') # 'g' para green
        elif event.button == 3:
            clicks['neg'].append((iy, ix))
            print(f"Clique negativo adicionado em: (linha={iy}, coluna={ix})")
            ax.plot(ix, iy, 'ro') # 'r' para red
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)

    # Redimensionar os cliques para as dimensões do modelo (256, 352)
    original_w, original_h = image_pil.size
    model_h, model_w = current_mask.shape
    
    resized_clicks = {'pos': [], 'neg': []}
    for p in clicks['pos']:
        new_y = int((p[0] / original_h) * model_h)
        new_x = int((p[1] / original_w) * model_w)
        resized_clicks['pos'].append((new_y, new_x))

    for p in clicks['neg']:
        new_y = int((p[0] / original_h) * model_h)
        new_x = int((p[1] / original_w) * model_w)
        resized_clicks['neg'].append((new_y, new_x))

    return resized_clicks