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

# --- Función Pose de Cabeza ---
def head_pose_estimation(face_landmarks, frame_w, frame_h):
    image_points = np.array([
        (face_landmarks.landmark[1].x * frame_w, face_landmarks.landmark[1].y * frame_h),   # Nariz
        (face_landmarks.landmark[152].x * frame_w, face_landmarks.landmark[152].y * frame_h), # Barbilla
        (face_landmarks.landmark[33].x * frame_w, face_landmarks.landmark[33].y * frame_h),  # Ojo izq
        (face_landmarks.landmark[263].x * frame_w, face_landmarks.landmark[263].y * frame_h),# Ojo der
        (face_landmarks.landmark[61].x * frame_w, face_landmarks.landmark[61].y * frame_h),  # Boca izq
        (face_landmarks.landmark[291].x * frame_w, face_landmarks.landmark[291].y * frame_h) # Boca der
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),     # Nariz
        (0.0, -63.6, -12.5), # Barbilla
        (-43.3, 32.7, -26.0),# Ojo izq
        (43.3, 32.7, -26.0), # Ojo der
        (-28.9, -28.9, -24.1),# Boca izq
        (28.9, -28.9, -24.1) # Boca der
    ])

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [angle[0] for angle in euler_angles]
    return pitch, yaw, roll

# --- Índices de landmarks ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 17]

# --- Parámetros ---
EAR_THRESH = 0.25
CLOSED_FRAMES = 15
MAR_THRESH = 0.6
PITCH_THRESH = 15  # umbral de cabeceo

# --- Variables ---
frame_counter = 0
fatiga_detectada = False

# --- Video ---
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

            for (x, y) in left_coords + right_coords:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # --- Boca ---
            mouth_landmarks = [face_landmarks.landmark[i] for i in MOUTH]
            mar, mouth_coords = mouth_aspect_ratio(mouth_landmarks, w, h)
            for (x, y) in mouth_coords:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # --- Cabeza ---
            pitch, yaw, roll = head_pose_estimation(face_landmarks, w, h)

            # --- Detección ojos (fatiga) ---
            if ear_avg < EAR_THRESH:
                frame_counter += 1
            else:
                frame_counter = 0
                fatiga_detectada = False

            if frame_counter >= CLOSED_FRAMES:
                fatiga_detectada = True

            # --- Lógica combinada ---
            estado = "Normal"
            color = (0, 255, 0)

            if fatiga_detectada and mar > MAR_THRESH:
                estado = "Bostezo detectado (Cansancio)"
                color = (255, 0, 0)
            elif fatiga_detectada:
                estado = "Fatiga detectada!"
                color = (0, 0, 255)
            elif mar > MAR_THRESH:
                estado = "Boca abierta (Estrés/Bostezo)"
                color = (0, 255, 255)
            elif pitch > PITCH_THRESH:
                estado = "Cabeceo detectado (Somnolencia)"
                color = (128, 0, 255)

            cv2.putText(frame, estado, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Deteccion Estrés / Fatiga / Somnolencia", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
