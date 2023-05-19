from flask import Flask, render_template, request, flash
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load your model and classes here
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '../best_custom_model.h5')
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f'Error loading model: {e}')

classes = {0: 'cloudy', 1: 'desert', 2: 'green_area', 3: 'water'}  # Your classes

def predict_image_class(image_url):
    target_size = (256, 256)  # The target size for your images
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")
            image = image.resize(target_size)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            prediction = model.predict(image_array)
            return prediction
        else:
            flash('Invalid URL', 'error')
            return None
    except requests.exceptions.RequestException as e:
        flash(str(e), 'error')
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image_url = request.form['url']
        prediction = predict_image_class(image_url)
        if prediction is not None:
            predicted_classes = {classes[i]: prob for i, prob in enumerate(prediction.flatten())}
            return render_template('result.html', url=image_url, class_probs=predicted_classes)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
