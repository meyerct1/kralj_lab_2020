#This file allows tf models to be viewed (protbuf .pb files)

import tensorflow as tf

model_path = '/Library/ML Data/Antibiotic videos/Models/diff_ augmented/saved_model.pb'
saved_model = tf.keras.models.load_model(model_path)

saved_model.summary()
