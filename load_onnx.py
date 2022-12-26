import tensorflow as tf
import numpy as np
import onnxruntime as ort


# Load test dataset
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

# Load model
sess = ort.InferenceSession("model_onnx/model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# Inference result
result = sess.run([label_name], {input_name: x_test})

predict = np.argmax(result[0], axis=1)

# Calculate accuracy
count = 0
for idx in range(len(y_test)):
    count += 1 if y_test[idx] == predict[idx] else 0

print("Test accuracy: ", count / len(y_test))
