import cv2
from ultralytics import YOLO

# 1. Cargar el modelo YOLOv8 pre-entrenado (versión 'n' de nano para mayor fluidez)
model = YOLO('yolov8n.pt')

# 2. Iniciar la captura de video (el 0 corresponde a tu cámara web principal)
cap = cv2.VideoCapture(0)

print("Iniciando cámara... Presiona 'q' en la ventana de video para detener el programa.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: No se pudo acceder a la cámara.")
        break

    # 3. Ejecutar la inferencia en el frame actual
    # classes=[2, 3, 5, 7] filtra exclusivamente: auto, moto, bus y camión
    # verbose=False silencia los logs por defecto de YOLO en la terminal
    results = model.predict(frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False)

    # 4. Obtener la cantidad de vehículos detectados
    conteo_vehiculos = len(results[0].boxes)
    
    # 5. Imprimir el conteo en la terminal 
    # El '\r' al final hace que la línea se sobreescriba, manteniendo la terminal limpia
    print(f"Vehículos detectados en este momento: {conteo_vehiculos}    ", end="\r")

    # --- Mostrar visualmente la detección (opcional pero útil para pruebas) ---
    annotated_frame = results[0].plot()
    cv2.imshow("Visión del Semáforo", annotated_frame)

    # 6. Condición para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos de la cámara y cerrar ventanas al terminar
cap.release()
cv2.destroyAllWindows()

print("\n\nPrograma finalizado correctamente.")