import cv2

print("Buscando cámaras activas con MSMF...")
for i in range(5):
    # Cambiamos DSHOW por MSMF
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF) 
    if cap.isOpened():
        success, frame = cap.read()
        if success:
            print(f"¡Cámara encontrada en el índice {i}!")
            cv2.imshow(f"Prueba {i}", frame)
            cv2.waitKey(2000) # Muestra la ventana 2 segundos
            cv2.destroyAllWindows()
    cap.release()
print("Búsqueda finalizada.")