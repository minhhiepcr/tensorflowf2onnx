# TensorFlow to ONNX Runtime

This repo will guide you to build simple TensorFlow model and convert it to ONNX runtime for deploy purpose. 

## 1. Content

| Content   | Descriptions |
| ----------- | ----------- |
| Introduction | Introduction to TensorFlow framework and ONNX Runtime. |
| Build and train model using TensorFlow framework | Instruction to build and train TensorFlow model. |
| Convert TensorFlow model to ONNX runtime | Instruction to convert TensorFlow model to ONNX runtime and use it to inference. |
| Load ONNX model and use this model to inference result | How to load ONNX model and use to inference result. |

## 2. Introduction

### 2.1 What is TensorFlow framework?

TensorFlow is an open source framework developed by Google researchers to run machine learning, deep learning and other statistical and predictive analytics workloads.

### 2.2 What is ONNX Runtime?

ONNX Runtime is an open source project that is designed to accelerate machine learning across a wide range of frameworks, operating systems, and hardware platforms.

### 2.3 Why don't we need to convert to ONNX Runtime? 

For deploying purpose, ONNX Runtime can help you deploy everywhere you want such as frontend, backend, microcontroller. If you deploy your app using docker image/container, ONNX Runtime will reduce the docker image size instead of importing tensorflow or pytorch or cuda library and increase the inferece speed. 

## 3. Build and train model using TensorFlow framework

### 3.1 Prequisted

Step 1: Create python virtual environment

```sh
python3 -m venv venv
```

Step 2: Activate virtual environment

In Linux,

```sh
source venv/bin/activate
```

In Windows,
```sh
venv/Scripts/activate
```

Step 3: Install relating python packages
```sh
pip install tensorflow onnxruntime tf2onnx
```

### 3.2 Usage

Step 1: Import relating packages

```python
import tensorflow as tf
import numpy as np
```

Step 2: Load MNIST dataset from TensorFlow API.

```python
#Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the image to (0,1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Using np.float32 data type to train model.
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
```

Step 3: Build neural network using TensorFlow framework.

```python
# Init model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28), name='input'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax', name='output')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

Step 4: Train neural network model

```python
#Define number of epochs, the batch size, the folder to save model.
EPOCHS = 50
BATCH_SIZE = 64
MODEL_TF_DIR = "model_tf"

#Train model using built-in .fit() function
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test))

#After training model, we will save model to the given directory.
try:
    model.save(MODEL_TF_DIR)
except Exception as ex:
    print(ex)
```

The last ten epochs are shown here:

```
Epoch 41/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0047 - accuracy: 0.9984 - val_loss: 0.1717 - val_accuracy: 0.9741
Epoch 42/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0073 - accuracy: 0.9974 - val_loss: 0.1661 - val_accuracy: 0.9740
Epoch 43/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0058 - accuracy: 0.9981 - val_loss: 0.1773 - val_accuracy: 0.9737
Epoch 44/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0046 - accuracy: 0.9986 - val_loss: 0.1680 - val_accuracy: 0.9765
Epoch 45/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0094 - accuracy: 0.9972 - val_loss: 0.1862 - val_accuracy: 0.9737
Epoch 46/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0051 - accuracy: 0.9982 - val_loss: 0.1609 - val_accuracy: 0.9759
Epoch 47/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0033 - accuracy: 0.9990 - val_loss: 0.1952 - val_accuracy: 0.9721
Epoch 48/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0071 - accuracy: 0.9976 - val_loss: 0.1909 - val_accuracy: 0.9728
Epoch 49/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0043 - accuracy: 0.9985 - val_loss: 0.1829 - val_accuracy: 0.9752
Epoch 50/50
938/938 [==============================] - 2s 2ms/step - loss: 0.0053 - accuracy: 0.9981 - val_loss: 0.2248 - val_accuracy: 0.9687
```

## 4. Convert TensorFlow model to ONNX runtime

Step 1: Import relating packages

```python
import onnx
import tf2onnx
import tensorflow as tf
```

Step 2: Define global variables

```python
MODEL_TF_DIR = "model_tf"
MODEL_TF2ONNX_DIR = "model_onnx/model.onnx"
```

Step 3: Load tf model, convert onnx model and save onnx model.

```python
#Load tf model
model = tf.keras.models.load_model("model_tf")

#Convert tf model to onnx model using tf2onnx library.
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

#Save onnx model to onnx directory.
try:
    onnx.save(onnx_model, MODEL_TF2ONNX_DIR)
    print("Save onnx model successfully")
except Exception as ex:
    print("Failed to save onnx model")
    print(ex)
```

## 5. Load ONNX model and use this model to inference result.

Step 1: Import relating packages

```python
import tensorflow as tf
import numpy as np
import onnxruntime as ort
```

Step 2: Load test MNIST dataset

```python
#Load test dataset
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.astype(np.float32)
```

Step 3: Load converted onnx model and inference result

```python
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
```
