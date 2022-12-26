import tensorflow as tf
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the image to (0,1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Using np.float32 data type to train model.
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

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

# Define number of epochs, the batch size, the folder to save model.
EPOCHS = 50
BATCH_SIZE = 64
MODEL_TF_DIR = "model_tf"

# Train model using built-in .fit() function
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test))

# After training model, we will save model to the given directory.
try:
    model.save(MODEL_TF_DIR)
except Exception as ex:
    print(ex)
