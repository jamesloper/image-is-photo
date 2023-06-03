from io import BytesIO
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
app = Flask(__name__)


@app.route("/")
def hello():
    return "ok"


@app.route("/classify")
def get_users():
    url = request.args.get('url')
    print('Classify URL:', url)
    image_data = requests.get(url).content

    # Preprocess the image
    image = Image.open(BytesIO(image_data))
    resized_image = image.resize((input_shape[1], input_shape[2]))
    if resized_image.mode != "RGB":
        resized_image = resized_image.convert("RGB")
    input_data = np.array(resized_image, dtype=np.float32)
    input_data = (input_data - 127.5) / 127.5  # Normalize the input image
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return jsonify({url: float(output_data[0])})
