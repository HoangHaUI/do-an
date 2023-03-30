# Python ≥3.5 is required
import load_image
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import sklearn
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exists in Colab.
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0 is required
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# Common imports

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures

EPOCHS =202
checkpoint_path = "runs/training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

def load_model(class_num):
    model = keras.models.Sequential()
    model.add(keras.layers.Resizing(
        244, 244, interpolation="bilinear", crop_to_aspect_ratio=False))
    model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                            input_shape=[224, 224, 3]))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.add(keras.layers.Dense(class_num))
    return model

# def start_train():

train_ds, val_ds, class_names = load_image.get_dataset()

model = load_model(len(class_names))




# opt = keras.optimizers.Adam(learning_rate=0.01)


# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True)

# model.compile(
#     optimizer= opt,
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS,
#     callbacks=[model_checkpoint_callback]
    
# )

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(EPOCHS)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
# plt.savefig("test.png")

# class_names=["chuan", "thieu_nhan"]

# testing_image = "./images/testing/z4203541157044_792a297f7aabc2382ed980e5c1cd7eb3.jpg"
# # Evaluate
# img = tf.keras.utils.load_img(
#     testing_image, target_size=(244, 244)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )


# # Load model 
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# Create a new model instance
model = load_model(3)

# Load the previously saved weights
model.load_weights(latest)

# model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
# # loss, acc = model.evaluate(test_images, test_labels, verbose=2)



from glob import glob
testing_image = glob("./images/nap_thieu/*.jpg")
for path in testing_image:
    # Evaluate
    img = tf.keras.utils.load_img(
        path, target_size=(244, 244)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    print(np.max(img_array[0]), np.min(img_array[0]))
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # _,_,class_names = get_dataset()

    class_names = ['chuan', 'thieu_nhan_dan']
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


