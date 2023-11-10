import numpy as np
import cv2

##esto es para indicar que se usa la cáamra del dispositivo pero si fuera externa se pondria (1)
cap = cv2.VideoCapture(0) 

while (cap.isOpened()):
    # ret es para ver si hay video o no, mientras que frame es para capturar cada frame de la imagen
    ret, frame = cap.read()

    if ret:
        # mostramos la imagen 
        cv2.imshow('frame', frame)
    
        # Salimos precionando la tecla q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()