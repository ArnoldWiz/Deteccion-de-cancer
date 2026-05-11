"""
Organizador de dataset YOLO: divide imágenes y labels en train/val/test.

Estructura esperada ANTES:
├── images/
│   ├── img_00001.jpg
│   ├── img_00002.jpg
│   └── ...
└── labels/
    ├── img_00001.txt
    ├── img_00002.txt
    └── ...

Estructura creada DESPUÉS:
dataset_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

Uso:
    python organizar_dataset.py --input . --output dataset_yolo --train 0.7 --val 0.2 --test 0.1
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Organizar dataset para YOLO")
    p.add_argument("--input", "-i", default=".", help="Carpeta con 'images/' y 'labels/' subdirectorios")
    p.add_argument("--output", "-o", default="dataset_yolo", help="Carpeta de salida")
    p.add_argument("--train", type=float, default=0.7, help="Proporción train (0-1)")
    p.add_argument("--val", type=float, default=0.2, help="Proporción val (0-1)")
    p.add_argument("--test", type=float, default=0.1, help="Proporción test (0-1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    
    input_root = Path(args.input)
    images_src = input_root / "images"
    labels_src = input_root / "labels"
    
    if not images_src.exists():
        print(f"Error: {images_src} no existe")
        return
    if not labels_src.exists():
        print(f"Error: {labels_src} no existe")
        return
    
    output_root = Path(args.output)
    
    # Crear estructura
    splits = {"train": args.train, "val": args.val, "test": args.test}
    for split in splits:
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)
    
    # Obtener lista de imágenes
    image_files = sorted([f for f in images_src.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    
    if not image_files:
        print(f"No se encontraron imágenes en {images_src}")
        return
    
    random.seed(args.seed)
    random.shuffle(image_files)
    
    # Calcular índices
    n = len(image_files)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    assignments = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    total_copied = 0
    for split, files in assignments.items():
        for img_file in files:
            label_file = labels_src / (img_file.stem + ".txt")
            
            # Copiar imagen
            dst_img = output_root / "images" / split / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Copiar label si existe
            if label_file.exists():
                dst_lbl = output_root / "labels" / split / label_file.name
                shutil.copy2(label_file, dst_lbl)
                total_copied += 1
    
    print(f"Dataset reorganizado:")
    print(f"  Train: {len(train_files)} imágenes")
    print(f"  Val:   {len(val_files)} imágenes")
    print(f"  Test:  {len(test_files)} imágenes")
    print(f"  Total: {total_copied} pares imagen-label copiados")
    print(f"  Salida: {output_root.absolute()}")


if __name__ == "__main__":
    main()
