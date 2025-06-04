from flask import Flask, request, jsonify
import tensorflow as tf
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEAF_MODEL_PATH    = os.path.join(BASE_DIR, "leafNetV3_model.tflite")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "converted_model.tflite")
MOBILEVIT_MODEL_PATH = os.path.join(BASE_DIR, "mobilevit_model17.pt")

# Load TFLite interpreters
leaf_model = tf.lite.Interpreter(model_path=LEAF_MODEL_PATH)
leaf_model.allocate_tensors()

disease_model = tf.lite.Interpreter(model_path=DISEASE_MODEL_PATH)
disease_model.allocate_tensors()

# Load MobileViT PyTorch model with weights_only=False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilevit_model = torch.load(MOBILEVIT_MODEL_PATH, map_location=device, weights_only=False)
mobilevit_model.eval()

# Preprocessing for MobileNetV3
def preprocess_image_mobilenet(image_bytes, target_size):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# Preprocessing for TFLite disease model
def preprocess_image(image_bytes, input_shape):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

# Preprocessing for MobileViT
mean = np.array([0.5, 0.5, 0.5])
std  = np.array([0.25, 0.25, 0.25])
common_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
def preprocess_image_mobilevit(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return common_transform(img).unsqueeze(0).to(device)  # [1,3,224,224]

# Inference functions
def run_tflite(interpreter, input_data):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])[0]

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def run_mobilevit(image_tensor):
    with torch.no_grad():
        outputs = mobilevit_model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()[0]  # [num_classes]

# Flask route
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify(success=False, message="No image file provided"), 400

    image_bytes = request.files["image"].read()

    # --- 1) Leaf Detection ---
    leaf_shape = leaf_model.get_input_details()[0]["shape"]
    leaf_input = preprocess_image_mobilenet(image_bytes, (leaf_shape[2], leaf_shape[1]))
    leaf_output = run_tflite(leaf_model, leaf_input)
    leaf_class = int(np.argmax(leaf_output))
    leaf_confidence = float(leaf_output[leaf_class])

    if leaf_class != 0:
        return jsonify({
            "success":     True,
            "stage":       "leaf_detection",
            "isLeaf":      False,
            "confidence":  leaf_confidence,
            "message":     "This is not a leaf",
            "results":     []
        })

    # --- 2) Disease Detection ---
    disease_labels = [
        "Apple: Apple scab", "Apple: Black rot", "Apple: Cedar apple rust",
        "Apple: healthy", "Grape: Esca (Black Measles)", "Pepper-bell: Bacterial-spot",
        "Pepper-bell: healthy", "Potato: Early blight", "Potato: Late blight",
        "Potato: healthy", "Strawberry: Leaf scorch", "Strawberry: healthy",
        "Tomato: Bacterial spot", "Tomato: Early blight", "Tomato: Late blight",
        "Tomato: Leaf Mold", "Tomato: healthy"
    ]

    # --- Run both models ---
    # 1. MobileNetV3 (TFLite)
    disease_input_shape = disease_model.get_input_details()[0]['shape']
    disease_input = preprocess_image(image_bytes, disease_input_shape)
    disease_output = run_inference(disease_model, disease_input)[0]

    top1_idx_mobilenet = np.argmax(disease_output)
    top1_conf_mobilenet = float(disease_output[top1_idx_mobilenet])
    label_mobilenet = disease_labels[top1_idx_mobilenet]

    # 2. MobileViT (PyTorch)
    image_tensor = preprocess_image_mobilevit(image_bytes)
    mobilevit_output = run_mobilevit(image_tensor)

    top1_idx_mobilevit = int(np.argmax(mobilevit_output))
    top1_conf_mobilevit = float(mobilevit_output[top1_idx_mobilevit])
    label_mobilevit = disease_labels[top1_idx_mobilevit]

    # Choose the best prediction
    if top1_conf_mobilevit > top1_conf_mobilenet:
        selected_label = label_mobilevit
        selected_confidence = top1_conf_mobilevit
    else:
        selected_label = label_mobilenet
        selected_confidence = top1_conf_mobilenet

    return jsonify({
        "success":         True,
        "stage":           "disease_detection",
        "isLeaf":          True,
        "leaf_confidence": leaf_confidence,
        "results": [{
            "label":      selected_label,
            "confidence": selected_confidence
        }],
        "message":         "Analysis complete"
    })

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
