import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# --- Inicializar Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# --- Función EAR (ojos) ---
def eye_aspect_ratio(eye_landmarks, frame_w, frame_h):
    coords = [(int(landmark.x * frame_w), int(landmark.y * frame_h)) for landmark in eye_landmarks]
    A = distance.euclidean(coords[1], coords[5])  # vertical
    B = distance.euclidean(coords[2], coords[4])  # vertical
    C = distance.euclidean(coords[0], coords[3])  # horizontal
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear, coords

# --- Función MAR (boca) ---
def mouth_aspect_ratio(mouth_landmarks, frame_w, frame_h):
    coords = [(int(landmark.x * frame_w), int(landmark.y * frame_h)) for landmark in mouth_landmarks]
    # Vertical: 13 (arriba) - 14 (abajo)
    A = distance.euclidean(coords[2], coords[3])
    # Horizontal: 61 (izq) - 291 (der)
    C = distance.euclidean(coords[0], coords[1])
    mar = A / C if C != 0 else 0
    return mar, coords

# --- Índices de landmarks ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 17]  # boca

# --- Parámetros ---
EAR_THRESH = 0.25        # umbral ojo cerrado
CLOSED_FRAMES = 15       # nº frames seguidos -> fatiga
MAR_THRESH = 0.6         # umbral boca abierta (bostezo)

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
            # --- Ojos ---
            left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            left_ear, left_coords = eye_aspect_ratio(left_eye_landmarks, w, h)
            right_ear, right_coords = eye_aspect_ratio(right_eye_landmarks, w, h)
            ear_avg = (left_ear + right_ear) / 2.0

            # Dibujar ojos
            for (x, y) in left_coords + right_coords:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # --- Boca ---
            mouth_landmarks = [face_landmarks.landmark[i] for i in MOUTH]
            mar, mouth_coords = mouth_aspect_ratio(mouth_landmarks, w, h)

            # Dibujar boca
            for (x, y) in mouth_coords:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # --- Detección ojos (fatiga) ---
            if ear_avg < EAR_THRESH:
                frame_counter += 1
            else:
                frame_counter = 0
                fatiga_detectada = False

            if frame_counter >= CLOSED_FRAMES:
                fatiga_detectada = True

            # --- Lógica combinada ---
            estado = ""
            if fatiga_detectada and mar > MAR_THRESH:
                estado = "Bostezo detectado (Cansancio)"
                color = (255, 0, 0)
            elif fatiga_detectada:
                estado = "Fatiga detectada!"
                color = (0, 0, 255)
            elif mar > MAR_THRESH:
                estado = "Boca abierta (estres/bostezo)"
                color = (0, 255, 255)
            else:
                estado = "Normal"
                color = (0, 255, 0)

            # Mostrar texto
            cv2.putText(frame, estado, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Deteccion Ojos y Boca / Estrés", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
