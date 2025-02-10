from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Fungsi untuk membangun model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Dua kelas: Organik & Non-Organik
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Siapkan data dengan augmentasi
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Hanya rescale untuk validasi

train_generator = train_datagen.flow_from_directory(
    'D:/Campus/python/Uas Pemodelan Simulasi/train_folder/train_data',  
    target_size=(128, 128),  
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'D:/Campus/python/Uas Pemodelan Simulasi/train_folder/val_data',  # Tambahkan folder validasi
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Bangun dan latih model
model = build_model()

# Training model dengan validasi
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Plot Training & Validation Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linestyle='-')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linestyle='-')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Simpan grafik sebagai file dan tampilkan
plt.savefig("training_plot.png")
plt.show()  # Menampilkan grafik

# Simpan model setelah grafik
model.save('model.h5')
print("Model berhasil dilatih dan disimpan sebagai 'model.h5'")
print("Grafik pelatihan telah disimpan sebagai 'training_plot.png'")
