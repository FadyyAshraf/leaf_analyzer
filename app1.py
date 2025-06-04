from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

app = Flask(__name__)

# Where your .tflite files live
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEAF_MODEL_PATH    = os.path.join(BASE_DIR, "leafNetV3_model.tflite")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "mobilevit_model17.pt")

# Load both TFLite interpreters once at startup
leaf_model = tf.lite.Interpreter(model_path=LEAF_MODEL_PATH)
leaf_model.allocate_tensors()

disease_model = tf.lite.Interpreter(model_path=DISEASE_MODEL_PATH)
disease_model.allocate_tensors()

def preprocess_image_mobilenet(image_bytes, target_size):
    """Resize, convert to array, expand dims and apply MobileNetV3 preprocess_input."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)  # (224,224)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)        # [1,224,224,3]
    arr = preprocess_input(arr)              # scales to [-1,1] exactly like Keras MobileNetV3
    return arr

def run_tflite(interpreter, input_data):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])[0]  # drop batch dim

def preprocess_image(image_bytes, input_shape):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    input_data = np.expand_dims(img_array, axis=0)
    return input_data

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify(success=False, message="No image file provided"), 400

    image_bytes = request.files["image"].read()

    # --- 1) Leaf Detection ---
    leaf_shape = leaf_model.get_input_details()[0]["shape"]   # e.g. [1,224,224,3]
    leaf_input = preprocess_image_mobilenet(image_bytes, (leaf_shape[2], leaf_shape[1]))
    leaf_output = run_tflite(leaf_model, leaf_input)          # e.g. [p_leaf, p_nonleaf]

    leaf_class      = int(np.argmax(leaf_output))             # 0 or 1
    leaf_confidence = float(leaf_output[leaf_class])

    # class 0 == “leaf” according to your mapping
    if leaf_class != 0:
        return jsonify({
            "success":     True,
            "stage":       "leaf_detection",
            "isLeaf":      False,
            "confidence":  leaf_confidence,
            "message":     "This is not a leaf",
            "results":     []
        })

    # --- 2) Disease Detection (only if leaf) ---
    # Disease detection preprocessing and inference (only if leaf)
    disease_input_details = disease_model.get_input_details()[0]
    disease_input_shape = disease_input_details['shape']
    disease_input = preprocess_image(image_bytes, disease_input_shape)
    disease_output = run_inference(disease_model, disease_input)[0]


    disease_labels = [
        "Apple: Apple scab", "Apple: Black rot", "Apple: Cedar apple rust",
        "Apple: healthy", "Grape: Esca (Black Measles)", "Pepper-bell: Bacterial-spot",
        "Pepper-bell: healthy", "Potato: Early blight", "Potato: Late blight",
        "Potato: healthy", "Strawberry: Leaf scorch", "Strawberry: healthy",
        "Tomato: Bacterial spot", "Tomato: Early blight", "Tomato: Late blight",
        "Tomato: Leaf Mold", "Tomato: healthy"
    ]

    # Top-5 by confidence (descending)
    top_idxs = np.argsort(disease_output)[::-1][:5]
    threshold = 0.75
    disease_results = []
    for idx in top_idxs:
        conf = float(disease_output[idx])
        if conf >= threshold:
            disease_results.append({"label": disease_labels[idx], "confidence": conf})

    # If none pass threshold, at least return the single top
    if not disease_results:
        top = top_idxs[0]
        disease_results.append({"label": disease_labels[top], "confidence": float(disease_output[top])})

    return jsonify({
        "success":        True,
        "stage":          "disease_detection",
        "isLeaf":         True,
        "leaf_confidence": leaf_confidence,
        "results":        disease_results,
        "message":        "Analysis complete"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
