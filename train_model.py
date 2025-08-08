import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ============================
# 1. Parámetros
# ============================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

train_dir = "dataset/train"
val_dir = "dataset/val"

# ============================
# 2. Cargar dataset
# ============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Opcional: caché y prefetch para acelerar
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ============================
# 3. Definir capa de Data Augmentation
# ============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),       # voltear horizontalmente
    tf.keras.layers.RandomRotation(0.2),            # rotar ±20%
    tf.keras.layers.RandomZoom(0.1),                # hacer zoom ±10%
    tf.keras.layers.RandomTranslation(0.1, 0.1),    # mover imagen
])


# ============================
# 3. Crear la CNN
# ============================
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    #data_augmentation,  # <<--- aplicamos aumentos aquí
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # salida binaria
    
])

model.summary()

# ============================
# 4. Compilar modelo
# ============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================
# 5. Entrenar
# ============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ============================
# 6. Guardar modelo
# ============================
if not os.path.exists("modelos"):
    os.makedirs("modelos")

model.save("modelos/modelo_estres.h5")
print("✅ Modelo guardado en modelos/modelo_estres.h5")

# ============================
# 7. Graficar resultados
# ============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Entrenamiento')
plt.plot(epochs_range, val_acc, label='Validación')
plt.legend(loc='lower right')
plt.title('Precisión')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Entrenamiento')
plt.plot(epochs_range, val_loss, label='Validación')
plt.legend(loc='upper right')
plt.title('Pérdida')

plt.show()
