from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load class mapping
index_to_class = np.load('index_to_class.npy', allow_pickle=True).item()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tomato_disease_cnn.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

        # Run inference using TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction result
        class_index = np.argmax(predictions)
        class_name = index_to_class[class_index]
        confidence = float(np.max(predictions))

        result_text = f"🌿 <b>Predicted Class:</b> {class_name}<br>🔹 <b>Confidence:</b> {confidence:.2%}"

        return render_template('index.html', result=result_text, image_path=filepath)

    except Exception as e:
        return render_template('index.html', result=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
