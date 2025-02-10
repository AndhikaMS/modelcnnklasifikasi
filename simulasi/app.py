# app.py
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Muat model yang telah dilatih
model = load_model('model.h5')

# Fungsi untuk mempersiapkan gambar: resize ke 128x128, konversi ke array, normalisasi
def prepare_image(filepath):
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Pastikan ada file di request
    if 'file' not in request.files:
        return "Tidak ada file yang diunggah."
    file = request.files['file']
    if file.filename == '':
        return "Tidak ada file yang dipilih."
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Persiapkan gambar untuk prediksi (resize ke 128x128)
        img_array = prepare_image(filepath)
        prediction = model.predict(img_array)
        # Misal: kelas 0 = Organik, kelas 1 = Non-Organik (sesuai urutan folder)
        class_names = ['Non-Organik', 'Organik']
        predicted_class = class_names[np.argmax(prediction)]
        
        # Gunakan url_for agar gambar diakses dengan benar dari folder static
        image_url = url_for('static', filename='uploads/' + filename)
        return render_template('index.html', predicted_class=predicted_class, img_path=image_url)

if __name__ == '__main__':
    app.run(debug=True)
