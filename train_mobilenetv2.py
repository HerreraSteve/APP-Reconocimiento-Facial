import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ============================
# 1. Parámetros
# ============================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
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

# ============================
# 3. Data Augmentation
# ============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# ============================
# 4. Preprocesamiento base MobileNetV2
# ============================
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# ============================
# 5. Modelo base MobileNetV2
# ============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # congelamos las capas base

# ============================
# 6. Construir modelo final
# ============================
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
#x = data_augmentation(inputs)               # aumentos
#x = preprocess_input(x)                      # preprocesamiento
x = preprocess_input(inputs)  
x = base_model(x, training=False)            # extractor de características
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # binaria
model = tf.keras.Model(inputs, outputs)

model.summary()

# ============================
# 7. Compilar
# ============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================
# 8. Entrenar
# ============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ============================
# 9. Guardar modelo
# ============================
if not os.path.exists("modelos"):
    os.makedirs("modelos")
model.save("modelos/modelo_mobilenetv2_estres.keras")
print("✅ Modelo guardado en modelos/modelo_mobilenetv2_estres.h5")

# ============================
# 10. Graficar resultados
# ============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Entrenamiento')
plt.plot(epochs_range, val_acc, label='Validación')
plt.legend(loc='lower right')
plt.title('Precisión (MobileNetV2 + Aug)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Entrenamiento')
plt.plot(epochs_range, val_loss, label='Validación')
plt.legend(loc='upper right')
plt.title('Pérdida (MobileNetV2 + Aug)')
plt.show()
