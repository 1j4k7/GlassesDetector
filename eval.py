import tensorflow as tf
import numpy as np
import pathlib
import sys

batch_size = 32
img_height = 120
img_width = 120

img_path = sys.argv[1]

img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height,img_width))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array,0)

data_dir = "./MeGlass_120x120"
data_dir = pathlib.Path(data_dir)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height,img_width),
        batch_size=batch_size)
class_names = train_ds.class_names

model = tf.keras.models.load_model('training/model.h5')

predictions = model.predict(img_array)
print(predictions)
#score = tf.nn.softmax(predictions[0])

#print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100*np.max(score)))
print("This image has glasses with a {:.2f} score.".format(predictions[0][0]))
