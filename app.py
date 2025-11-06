from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model and class mapping
model = tf.keras.models.load_model('tomato_disease_cnn.keras')
index_to_class = np.load('index_to_class.npy', allow_pickle=True).item()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='No file uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result='No file selected.')

    try:
        # Preprocess image
        img = Image.open(file).convert('RGB').resize((256, 256))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        class_name = index_to_class[class_index]
        confidence = float(np.max(predictions))

        result_text = f"🌿 Predicted Class: {class_name}<br>🔹 Confidence: {confidence:.2%}"
        return render_template('index.html', result=result_text)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
