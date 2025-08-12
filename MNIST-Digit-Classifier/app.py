# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io, base64

# -Create Flask app-
app = Flask(__name__)

# -Load the trained TensorFlow model-
# This file must be in the same folder as app.py
model = tf.keras.models.load_model('model.h5')

# ---- Route: Homepage ----
@app.route('/')
def index():
    # Render the HTML page with the drawing canvas
    return render_template('index.html')

# ---- Route: Prediction endpoint ----
@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON body: { "image": "data:image/png;base64,..." }
    data = request.get_json()
    img_b64 = data.get('image')                # the image as a base64 string

    # Split off the header (like "data:image/png;base64,")
    header, encoded = img_b64.split(',', 1)

    # Decode base64 to binary bytes
    img_bytes = base64.b64decode(encoded)

    # Read bytes into a PIL Image and convert to grayscale ('L')
    img = Image.open(io.BytesIO(img_bytes)).convert('L')

    # Resize to 28x28 pixels (the size MNIST expects)
    img = img.resize((28, 28), Image.ANTIALIAS)

    # Convert image to numpy array (shape: 28x28)
    arr = np.array(img).astype('float32')

    # Invert colors: canvas black-on-white -> MNIST white-on-black
    arr = 255 - arr

    # Normalize pixel values to [0,1]
    arr = arr / 255.0

    # Reshape to (1, 28, 28, 1) because TF expects a batch dimension and channel
    arr = arr.reshape(1, 28, 28, 1)

    # Run the model to get probabilities for each digit class
    pred = model.predict(arr)

    # Choose the class with highest probability
    pred_class = int(np.argmax(pred, axis=1)[0])

    # Return prediction as JSON
    return jsonify({'prediction': pred_class})

# ---- Run server when invoked directly ----
if __name__ == '__main__':
    # debug=True restarts the server when files change and prints errors to console
    app.run(debug=True)
