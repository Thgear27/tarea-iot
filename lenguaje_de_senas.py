import cv2
import mediapipe as mp #para el detector de manos 
import numpy as np

def Etiqueta(idx, mano, results):
    aux = None
    for _, clase in enumerate(results.multi_handedness):
      if clase.classification[0].index == idx:
        label = clase.classification[0].label
        texto = '{}'.format(label)

        coords = tuple(np.multiply(np.array(
           (mano.landmark[mp_manos.HandLandmark.WRIST].x, 
            mano.landmark[mp_manos.HandLandmark.WRIST].y)),
            [1920, 1080]).astype(int))
        
        aux = texto, coords
    return aux

def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_manos = mp.solutions.hands
change = True
change2 = False

cap = cv2.VideoCapture(0)
with mp_manos.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    #Sirve para que mediapipe trabaje correctamente con los colores
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape
    if results.multi_hand_landmarks: ##verificamos si hay resultados
        if len(results.multi_hand_landmarks): ##para detectar la cantidad de manos
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_manos.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                punta_dedo_indice = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                resto_dedo_indice = (int(hand_landmarks.landmark[6].x * image_width),
                                int(hand_landmarks.landmark[6].y * image_height))
                
                punta_dedo_pulgar = (int(hand_landmarks.landmark[4].x * image_width),
                                int(hand_landmarks.landmark[4].y * image_height))
                resto_dedo_pulgar = (int(hand_landmarks.landmark[2].x * image_width),
                                int(hand_landmarks.landmark[2].y * image_height))
                
                punta_dedo_medio = (int(hand_landmarks.landmark[12].x * image_width),
                                int(hand_landmarks.landmark[12].y * image_height))
                
                resto_dedo_medio = (int(hand_landmarks.landmark[10].x * image_width),
                                int(hand_landmarks.landmark[10].y * image_height))
                
                punta_dedo_anular = (int(hand_landmarks.landmark[16].x * image_width),
                                int(hand_landmarks.landmark[16].y * image_height))
                resto_dedo_anular = (int(hand_landmarks.landmark[14].x * image_width),
                                int(hand_landmarks.landmark[14].y * image_height))
                
                punta_dedo_menique = (int(hand_landmarks.landmark[20].x * image_width),
                                int(hand_landmarks.landmark[20].y * image_height))
                resto_dedo_menique = (int(hand_landmarks.landmark[18].x * image_width),
                                int(hand_landmarks.landmark[18].y * image_height))
                
                muñeca = (int(hand_landmarks.landmark[0].x * image_width),
                                int(hand_landmarks.landmark[0].y * image_height))
                
                if resto_dedo_pulgar[1] - punta_dedo_pulgar[1] > 0 and resto_dedo_pulgar[1] - punta_dedo_indice[1] < 0 \
                    and resto_dedo_pulgar[1] - punta_dedo_medio[1] < 0 and resto_dedo_pulgar[1] - punta_dedo_anular[1]<0 \
                    and resto_dedo_pulgar[1] - punta_dedo_menique[1] < 0:
                    print("GOOD")

                elif resto_dedo_pulgar[1] - punta_dedo_pulgar[1] < 0 and resto_dedo_pulgar[1] - punta_dedo_indice[1] > 0 \
                    and resto_dedo_pulgar[1] - punta_dedo_medio[1] > 0 and resto_dedo_pulgar[1] - punta_dedo_anular[1]>0 \
                    and resto_dedo_pulgar[1] - punta_dedo_menique[1] > 0:
                    print("BAD")
                   
                elif resto_dedo_pulgar[1] - punta_dedo_pulgar[1] > 0 and resto_dedo_indice[1] - punta_dedo_indice[1]>0 \
                    and resto_dedo_menique[1] - punta_dedo_menique[1] > 0:
                    print("I LOVE YOU")
                   
                    
                if Etiqueta(num, hand_landmarks, results) and len(results.multi_hand_landmarks)==2:
                    text,coords = Etiqueta(num, hand_landmarks, results)
                    #print(text, coords)
                    if text =="Right":
                      #text = "IZQUIERDA"
                      index_finger_tip_r = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                      #print(index_finger_tip)
                      change = True
                    if text =="Left":
                        #text = "DERECHA"
                        index_finger_tip_l = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                      
                        muñeca = (int(hand_landmarks.landmark[0].x * image_width),
                                int(hand_landmarks.landmark[0].y * image_height))

                        change2 = True

                    if change2 == True and change == True:
                        if distancia_euclidiana(index_finger_tip_l,  muñeca) < 170.0:
                            print("Que hora es?")
                
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()