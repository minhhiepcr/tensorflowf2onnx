import onnx
import tf2onnx
import tensorflow as tf

MODEL_TF_DIR = "model_tf"
MODEL_TF2ONNX_DIR = "model_onnx/model.onnx"

# Load tf model
model = tf.keras.models.load_model("model_tf")

# Convert tf model to onnx model using tf2onnx library.
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

# Save onnx model to onnx directory.
try:
    onnx.save(onnx_model, MODEL_TF2ONNX_DIR)
    print("Save onnx model successfully")
except Exception as ex:
    print("Failed to save onnx model")
    print(ex)
