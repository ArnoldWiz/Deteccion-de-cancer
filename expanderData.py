# =========================================
# 🔄 EXPANSIÓN DE DATASET (SIN PREPROCESAR)
# =========================================

import os
import cv2
import tensorflow as tf
from tqdm import tqdm

# RUTAS
input_dir = "dataset_original"
output_dir = "dataset_expandido"

# PARÁMETROS
AUMENTOS_POR_IMAGEN = 3  # cuántas nuevas por imagen

# DATA AUGMENTATION (SUAVE PARA IMÁGENES MÉDICAS)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# FUNCIÓN PARA GUARDAR IMAGEN
def guardar_imagen(img, ruta):
    img = img.numpy()
    img = (img * 255).astype("uint8")
    cv2.imwrite(ruta, img)

# PROCESAR DATASET
def expandir_dataset():
    clases = ["sano", "enfermo"]

    for clase in clases:
        ruta_clase = os.path.join(input_dir, clase)
        salida_clase = os.path.join(output_dir, clase)

        os.makedirs(salida_clase, exist_ok=True)

        print(f"\nProcesando clase: {clase}")

        for img_nombre in tqdm(os.listdir(ruta_clase)):
            ruta_img = os.path.join(ruta_clase, img_nombre)

            # Leer imagen tal cual (color original)
            img = cv2.imread(ruta_img)

            if img is None:
                continue

            # Convertir a RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalizar para augmentation
            img = img / 255.0

            nombre_base = os.path.splitext(img_nombre)[0]

            # 🔹 Guardar ORIGINAL (sin tocar)
            ruta_original = os.path.join(
                salida_clase, f"{nombre_base}_orig.png"
            )
            cv2.imwrite(ruta_original, cv2.imread(ruta_img))

            # 🔹 Generar aumentos
            for i in range(AUMENTOS_POR_IMAGEN):
                aug = data_augmentation(tf.expand_dims(img, 0))
                aug_img = aug[0]

                ruta_aug = os.path.join(
                    salida_clase,
                    f"{nombre_base}_aug_{i}.png"
                )

                guardar_imagen(aug_img, ruta_aug)

# EJECUTAR
if __name__ == "__main__":
    expandir_dataset()