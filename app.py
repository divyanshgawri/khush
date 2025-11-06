from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load class mapping and trained model
index_to_class = np.load('index_to_class.npy', allow_pickle=True).item()

with open('tomato_disease_cnn.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='❌ No file uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result='⚠️ No file selected.')

    try:
        # Save uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = Image.open(filepath).convert('RGB').resize((256, 256))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        class_name = index_to_class[class_index]
        confidence = float(np.max(predictions))

        result_text = f"🌿 <b>Predicted Class:</b> {class_name}<br>🔹 <b>Confidence:</b> {confidence:.2%}"

        # Render same page with image and result
        return render_template(
            'index.html',
            result=result_text,
            image_path=filepath
        )

    except Exception as e:
        return render_template('index.html', result=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
