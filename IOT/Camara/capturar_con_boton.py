"""
Capturar fotos desde la cámara con un botón (GUI Tkinter).
Guarda imágenes redimensionadas (para YOLO) en una carpeta plana.

Controles GUI:
 - Botón "Capturar": guarda la imagen actual redimensionada
 - Botón "Salir": cierra la ventana y libera la cámara

Uso:
    python capturar_con_boton.py --output ../../dataset_capturas --width 640 --height 640

Requisitos:
    pip install opencv-python pillow
"""

import argparse
import sys
import cv2
import os
from pathlib import Path
from tkinter import Tk, Label, Button, LEFT, RIGHT, TOP
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


def parse_args():
    p = argparse.ArgumentParser(description="Capturar fotos con botón (GUI)")
    p.add_argument("--output", "-o", default="dataset_capturas", help="Carpeta donde guardar las imágenes")
    p.add_argument("--camera", type=int, default=0, help="ID de la cámara")
    p.add_argument("--width", type=int, default=640, help="Ancho objetivo (px) para guardar)")
    p.add_argument("--height", type=int, default=640, help="Alto objetivo (px) para guardar)")
    return p.parse_args()


def next_index(folder: Path) -> int:
    if not folder.exists():
        return 1
    files = list(folder.glob("img_*.jpg"))
    if not files:
        return 1
    max_i = 0
    for f in files:
        stem = f.stem  # img_00001
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            try:
                i = int(parts[-1])
                if i > max_i:
                    max_i = i
            except ValueError:
                pass
    return max_i + 1


class CameraApp:
    def __init__(self, camera_id, out_folder: Path, target_size):
        self.cap = cv2.VideoCapture(camera_id)
        self.out_folder = out_folder
        self.target_size = target_size
        self.frame = None
        self.index = next_index(self.out_folder)

        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara. Comprueba el ID o permisos.")

        self.root = Tk()
        self.root.title("Captura con botón")

        self.label = Label(self.root)
        self.label.pack()

        self.count_label = Label(self.root, text=f"Siguiente: img_{self.index:05d}.jpg")
        self.count_label.pack()

        btn_frame = Label(self.root)
        btn_frame.pack()

        self.capture_btn = Button(btn_frame, text="Capturar", command=self.capture)
        self.capture_btn.pack(side=LEFT)

        self.quit_btn = Button(btn_frame, text="Salir", command=self.quit)
        self.quit_btn.pack(side=RIGHT)

        # start update loop
        self.update()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            # try again later
            self.root.after(100, self.update)
            return

        # store original frame for saving, but display resized to fit label
        # resize to target size for saving
        h, w = self.target_size[1], self.target_size[0]
        frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        self.frame = frame_resized

        # convert BGR -> RGB for PIL
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        if Image is None or ImageTk is None:
            # fallback: convert to BGR->RGB then show via OpenCV window
            cv2.imshow('Captura - presione q en la ventana para salir', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.quit()
            self.root.after(30, self.update)
            return

        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        self.root.after(30, self.update)

    def capture(self):
        if self.frame is None:
            return
        ensure_dir(self.out_folder)
        filename = f"foto_{self.index:05d}.jpg"
        path = self.out_folder / filename
        # save as JPEG
        cv2.imwrite(str(path), self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Guardada -> {path}")
        self.index += 1
        self.count_label.config(text=f"Siguiente: foto_{self.index:05d}.jpg")

    def quit(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    out = Path(args.output)
    width = int(args.width)
    height = int(args.height)

    try:
        app = CameraApp(args.camera, out, (width, height))
        app.root.mainloop()
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
