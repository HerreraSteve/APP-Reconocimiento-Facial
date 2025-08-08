import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# ==============================
# 1. Cargar modelo entrenado
# ==============================
# Por ahora simulemos un modelo, pero si ya tienes uno, descomenta:
# model = tf.keras.models.load_model('modelos/modelo_estres.h5')
model = tf.keras.models.load_model("modelos/modelo_mobilenetv2_estres.keras")
IMG_SIZE = (128, 128)

# Simulador de modelo: devuelve aleatorio
import random
def modelo_simulado(img):
    # Aquí podrías hacer predicción real: model.predict(...)
    return random.random()  # valor entre 0 y 1


# ==============================
# 2. Inicializar detector de rostro
# ==============================
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ==============================
# 3. Captura de video
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

# ==============================
# 4. Loop principal
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el frame.")
        break

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    # Para cada rostro detectado
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
            w_box, h_box = int(bboxC.width * w), int(bboxC.height * h)

            # Asegurarse que los límites son válidos
            x, y = max(0, x), max(0, y)
            w_box, h_box = max(0, w_box), max(0, h_box)

            # Recortar el rostro
            face_img = frame[y:y+h_box, x:x+w_box]
            if face_img.size != 0:
                # Preprocesar: redimensionar, normalizar
                face_resized = cv2.resize(face_img, IMG_SIZE)
                face_array = np.expand_dims(face_resized / 255.0, axis=0)  # shape (1,64,64,3)

                # ==============================
                # 5. Predicción con modelo
                # ==============================
                pred = model.predict(face_array)[0][0]
                # pred = modelo_simulado(face_array)  # usando el simulador

                # Determinar estado
                estado = "Estres" if pred > 0.5 else "Relajado"
                color = (0, 0, 255) if estado == "Estres" else (0, 255, 0)

                # Mostrar en pantalla
                cv2.putText(frame, f"{estado} ({pred:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Dibujar bounding box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 2)

    # Mostrar frame
    cv2.imshow("Deteccion de Estres", frame)

    # Salir con tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
