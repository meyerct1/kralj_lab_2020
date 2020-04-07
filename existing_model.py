#Eugene Miller
#Using an existing model from Google to retrain for bacterial-identification purposes.

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub

data_dir = "/Library/ML Data/kralj-lab.tmp/Data"        # Directory with data, seperated into subfolders by category
model_save_dir = "/Library/ML Data/kralj-lab.tmp/Models/"       # Directory where the model will be saved (saved_model format)
BATCH_SIZE = 128 #@param {type:"integer"}
do_data_augmentation = False #@param {type:"boolean"}   # True enables random resize/rotation of images, not very useful for our purposes
do_fine_tuning = False #@param {type:"boolean"}         # True enables fine tuning; transfer learning
dropout_rate = 0.2      # probability a neuron is deactivated for an training step

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# Using an existing trained image recognition model (MAKE SURE DATA IMAGES ARE OF THE APPROPRIATE RESOLUTION)
module_selection = ("inception_v3", 299) #@param ["(\"mobilenet_v2_100_224\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print(IMAGE_SIZE)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

# defining test/training datasets
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

# data augmentation code (data randomization) only applies to training set
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir,  subset="training", shuffle=True, **dataflow_kwargs)

print("Building model with", MODULE_HANDLE)

# model superparameters, dropout, regularization, activation layer
model = tf.keras.Sequential([
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=dropout_rate),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

# Compiler
model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
  metrics=['accuracy'])

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size

# Run
hist = model.fit(
    train_generator,
    epochs=15, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps).history

# Results
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
plt.show()

# Save
saved_model_path = model_save_dir
tf.saved_model.save(model, saved_model_path)
