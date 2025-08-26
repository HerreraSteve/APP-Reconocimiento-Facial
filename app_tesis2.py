import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# --- Inicializar Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# --- Función EAR ---
def eye_aspect_ratio(eye_landmarks, frame_w, frame_h):
    coords = [(int(landmark.x * frame_w), int(landmark.y * frame_h)) for landmark in eye_landmarks]
    A = distance.euclidean(coords[1], coords[5])  # vertical
    B = distance.euclidean(coords[2], coords[4])  # vertical
    C = distance.euclidean(coords[0], coords[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear, coords

# --- Índices de landmarks de los ojos (FaceMesh) ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --- Parámetros ---
EAR_THRESH = 0.25       # umbral para considerar ojo cerrado
CLOSED_FRAMES = 15      # nº de frames consecutivos para considerar fatiga

# --- Variables de control ---
frame_counter = 0
fatiga_detectada = False

# --- Captura de video ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Ojo izquierdo
            left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
            left_ear, left_coords = eye_aspect_ratio(left_eye_landmarks, w, h)

            # Ojo derecho
            right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            right_ear, right_coords = eye_aspect_ratio(right_eye_landmarks, w, h)

            # EAR promedio
            ear_avg = (left_ear + right_ear) / 2.0

            # Dibujar landmarks de los ojos
            for (x, y) in left_coords + right_coords:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # --- Detección de parpadeo/fatiga ---
            if ear_avg < EAR_THRESH:
                frame_counter += 1
            else:
                frame_counter = 0
                fatiga_detectada = False

            # Si se mantienen cerrados por muchos frames
            if frame_counter >= CLOSED_FRAMES:
                fatiga_detectada = True

            # Mostrar estado en pantalla
            if fatiga_detectada:
                cv2.putText(frame, "ESTRES DETECTADO!", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                estado = "Cerrado" if ear_avg < EAR_THRESH else "Abierto"
                color = (0, 0, 255) if estado == "Cerrado" else (0, 255, 0)
                cv2.putText(frame, f"Ojos: {estado}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Deteccion Ojos / Estres", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
