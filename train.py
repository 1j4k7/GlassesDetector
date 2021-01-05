import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Download the dataset (already on disk in this case)
data_dir = "./MeGlass_120x120"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Image count: " + str(image_count) + " in " + str(data_dir))

# Create a dataset
batch_size = 1024 # larger batch_size for imbalanced data
img_height = 120
img_width = 120

random_seed = tf.random.uniform(shape=[], minval=1, maxval=1000, dtype=tf.int64)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        label_mode='binary',
        seed=random_seed,
        image_size=(img_height,img_width),
        batch_size=batch_size)

for features, label in train_ds.take(1):
    print("Feature shape:", features.shape)
    print("Label shape: ", label.shape)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        label_mode='binary',
        seed=random_seed,
        image_size=(img_height,img_width),
        batch_size=batch_size)

# Configure the dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache(). shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(img_height,img_width,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    ])

# Set initial bias for imbalanced dataset
pos = 14832 # number of positive occurances total
neg = 33085 # number of negative occurances total
total = pos + neg
initial_bias = np.log([pos/neg])
output_bias = tf.keras.initializers.Constant(initial_bias)

# Calculate class weights for imbalanced dataset
weight_for_0 = (1 / neg)*(total)/2.0
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# Create the model
num_classes = 1

metrics = [
      #keras.metrics.TruePositives(name='tp'),
      #keras.metrics.FalsePositives(name='fp'),
      #keras.metrics.TrueNegatives(name='tn'),
      #keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      #keras.metrics.Precision(name='precision'),
      #keras.metrics.Recall(name='recall'),
      #keras.metrics.AUC(name='auc'),
]

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, bias_initializer=output_bias)])
    #layers.Dense(num_classes)])

# Compile the model
model.compile(optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics)

model.summary()

# Train the model
epochs = 500
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight)
#history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Save the trained model
model.save('./training/model.h5')

# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./training/figure.png')
plt.show()
