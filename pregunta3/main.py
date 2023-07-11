import cv2
import mediapipe as mp

# Cargar el video
video = cv2.VideoCapture('video.mp4')

# Crear el detector de rostros de Mediapipe
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Variables para contar las vocales abiertas y cerradas
vocales_abiertas = 0
vocales_cerradas = 0

while True:
    # Leer un cuadro del video
    ret, frame = video.read()
    if not ret:
        break

    # Convertir el cuadro a RGB (Mediapipe utiliza el formato RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar los puntos clave faciales
    results = mp_face_mesh.process(frame_rgb)
    landmarks = results.multi_face_landmarks

    # Verificar si se detectaron landmarks del rostro y de la boca
    if landmarks:
        # Extraer los puntos clave de la boca

        face_landmarks = landmarks[0]

        #Al usar el índice de corte [50:55], se extraen los puntos clave desde el índice 50 hasta el índice 54
        mouth_landmarks = face_landmarks.landmark[50:55]

        # Calcular la apertura de la boca
        y_min = min([landmark.y for landmark in mouth_landmarks])
        y_max = max([landmark.y for landmark in mouth_landmarks])
        mouth_height = y_max - y_min

        # Determinar si la boca está abierta o cerrada
        umbral_apertura = 0.12
        boca_abierta = mouth_height > umbral_apertura # si la altura de la boca es mayor que el umbral entonces se ha hablado una vocal abierta sino se habló una vocal cerrada


        if boca_abierta:
            vocales_abiertas += 1 # se actualiza el contador
        else:
            vocales_cerradas += 1  # se actualiza el contador

        # Dibujar los puntos clave en el cuadro
        for landmark in mouth_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Mostrar el cuadro con los puntos clave faciales
    cv2.imshow('Video', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
video.release()
cv2.destroyAllWindows()

# Imprimir los resultados de las cantidades
print('Vocales abiertas:', vocales_abiertas)
print('Vocales cerradas:', vocales_cerradas)
