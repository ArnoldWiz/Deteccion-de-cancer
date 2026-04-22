
import os
import cv2
import tensorflow as tf
from tqdm import tqdm

input_dir = "dataset_original"
output_dir = "dataset_expandido"
target_folder = "iniciando"

AUMENTOS_POR_IMAGEN = 3  

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

def guardar_imagen(img, ruta):
    img = img.numpy()
    img = (img * 255).astype("uint8")
    cv2.imwrite(ruta, img)

def expandir_dataset():
    ruta_entrada = os.path.join(input_dir, target_folder)
    ruta_salida = os.path.join(output_dir, target_folder)

    if not os.path.isdir(ruta_entrada):
        raise FileNotFoundError(f"No existe la carpeta de entrada: {ruta_entrada}")

    os.makedirs(ruta_salida, exist_ok=True)

    print(f"\nProcesando carpeta: {target_folder}")

    for img_nombre in tqdm(os.listdir(ruta_entrada)):
        ruta_img = os.path.join(ruta_entrada, img_nombre)

        img = cv2.imread(ruta_img)

        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0

        nombre_base = os.path.splitext(img_nombre)[0]

        ruta_original = os.path.join(
            ruta_salida, f"{nombre_base}_orig.png"
        )
        cv2.imwrite(ruta_original, cv2.imread(ruta_img))

        for i in range(AUMENTOS_POR_IMAGEN):
            aug = data_augmentation(tf.expand_dims(img, 0))
            aug_img = aug[0]

            ruta_aug = os.path.join(
                ruta_salida,
                f"{nombre_base}_aug_{i}.png"
            )

            guardar_imagen(aug_img, ruta_aug)

if __name__ == "__main__":
    expandir_dataset()