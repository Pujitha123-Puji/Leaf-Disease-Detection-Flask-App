**Title of the Progect**: Leaf Disease Detection Flask App

**Objective**: The ”Leaf Disease Detection Flask App” aims to develop a user-friendly
web application for detecting plant leaf disease using machine learning techniques. The project focuses on creating a
robust system capable of accurately identifying various types of diseases affecting plant leaves. By leveraging machine
learning algorithms, the app will empower users, including farmers and gardeners, to quickly diagnose plant health
issues, thereby facilitating timely interventions to prevent crop losses and maintain plant vitality.

**Technologies Used**:
Python
Flask Framework
Machine Learning (TensorFlow/Keras)
OpenCV for image preprocessing
HTML/CSS for the front-end

**Flask Application Code (app.py)**:
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/leaf_model.h5'
model = load_model(MODEL_PATH)

# Target size of the images
IMG_SIZE = (128, 128)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Class labels (example)
        class_labels = ['Healthy', 'Powdery Mildew', 'Yellow Rust']
        result = class_labels[class_index]

        return render_template('result.html', result=result, confidence=confidence, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

**index.html: Upload Page**:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
    <title>Leaf Disease Detection</title>
</head>
<body>
    <h1>Leaf Disease Detection</h1>
    <form action="{{ url_for('predict') }}" method="POST" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Predict</button>
    </form>
</body>
</html>

**result.html: Result Page**:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
    <title>Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <img src="{{ image_path }}" alt="Uploaded Leaf">
    <p>Result: {{ result }}</p>
    <p>Confidence: {{ confidence }}%</p>
    <a href="{{ url_for('home') }}">Back</a>
</body>
</html>

**Style Sheet (styles.css)**:
body {
    font-family: Arial, sans-serif;
    text-align: center;
    margin: 50px;
}

h1 {
    color: #4CAF50;
}

form {
    margin: 20px auto;
}

button {
    padding: 10px 20px;
    background-color: #4CAF50;
    color: white;
    border: none;
    cursor: pointer;
}

button:hover {
    background-color: #45a049;
}
