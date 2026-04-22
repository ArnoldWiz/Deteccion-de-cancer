import cv2
import os
from tqdm import tqdm

input_dir = "dataset_original"
output_dir = "dataset_procesado"

IMG_SIZE = 224
GAUSSIAN_KERNEL = (5, 5)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Crear CLAHE
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

def procesar_imagen(ruta):
    # Leer imagen en escala de grises
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # 1. Redimensionar
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 2. Filtro Gaussiano (reducir ruido)
    img = cv2.GaussianBlur(img, GAUSSIAN_KERNEL, 0)

    # 3. Umbralización para aislar pulmones (simple)
    _, mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(img, mask)

    # 4. CLAHE (mejorar contraste)
    img = clahe.apply(img)

    # 5. Normalización 
    img = img / 255.0
    img = (img * 255).astype("uint8")

    return img

def procesar_dataset():
    clases = os.listdir(input_dir)

    for clase in clases:
        ruta_clase = os.path.join(input_dir, clase)
        salida_clase = os.path.join(output_dir, clase)

        os.makedirs(salida_clase, exist_ok=True)

        imagenes = os.listdir(ruta_clase)

        print(f"Procesando clase: {clase}")

        for img_nombre in tqdm(imagenes):
            ruta_img = os.path.join(ruta_clase, img_nombre)

            img_procesada = procesar_imagen(ruta_img)

            if img_procesada is not None:
                salida_img = os.path.join(salida_clase, img_nombre)
                cv2.imwrite(salida_img, img_procesada)

if __name__ == "__main__":
    procesar_dataset()