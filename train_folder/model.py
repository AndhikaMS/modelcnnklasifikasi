from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

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

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Naik satu level ke 'uas projek'
train_dir = os.path.join(base_dir, "train_folder", "train_data")

val_dir = os.path.join(base_dir, "train_folder", "val_data")

# Siapkan data dengan augmentasi
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2
)


val_datagen = ImageDataGenerator(rescale=1./255)  # Hanya rescale untuk validasi

train_path = "./train_folder/train_data"
print("Cek apakah path ada:", os.path.exists(train_path))
print("Isi folder jika ada:", os.listdir(train_path) if os.path.exists(train_path) else "Folder tidak ditemukan")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  
    batch_size=32,
    class_mode='categorical'
)

val_path = "./train_folder/val_data"
print("Cek apakah path validasi ada:", os.path.exists(val_path))
print("Isi folder validasi jika ada:", os.listdir(val_path) if os.path.exists(val_path) else "Folder tidak ditemukan")
validation_generator = val_datagen.flow_from_directory(
    val_dir,
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
# Pastikan folder `simulasi/` ada sebelum menyimpan model
simulasi_path = os.path.join(os.path.dirname(__file__), "..", "simulasi")
os.makedirs(simulasi_path, exist_ok=True)  # Buat folder jika belum ada

# Simpan model ke dalam folder `simulasi/`
model_path = os.path.join(simulasi_path, "model.h5")
model.save(model_path)
print("Model berhasil dilatih dan disimpan sebagai 'model.h5'")
print("Grafik pelatihan telah disimpan sebagai 'training_plot.png'")
