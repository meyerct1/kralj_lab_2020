#Using an existing model from Google to retrain for bacterial-identification purposes.

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorboard.plugins.hparams import api as hp

#Try this
tf.keras.backend.clear_session()

print(tf.__version__)
acc_list = []
print(type(acc_list))
data_dir = "/Library/ML Data/kralj-lab.tmp/Data"        # Directory with data, seperated into subfolders by category
win_data_dir = r"C:\Users\eugmille\Desktop\kralj-lab.tmp\Data"
model_save_dir = "/Library/ML Data/kralj-lab.tmp/Models/"       # Directory where the model will be saved (saved_model format)
win_model_save_dir = r"C:\Users\eugmille\Desktop\kralj-lab.tmp\Models"
BATCH_SIZE = 128 #@param {type:"integer"}
do_data_augmentation = False #@param {type:"boolean"}   # True enables random resize/rotation of images, not very useful for our purposes
do_fine_tuning = False #@param {type:"boolean"}         # True enables fine tuning; transfer learning
dropout_rate = 0.2      # probability a neuron is deactivated for an training step
learn_rate = 0.005
momentum = 0.9
loss_label_smoothing = 0.1

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Using an existing trained image recognition model (MAKE SURE DATA IMAGES ARE OF THE APPROPRIATE RESOLUTION)
module_selection = ("inception_v3", 299) #@param ["(\"mobilenet_v2_100_224\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print(IMAGE_SIZE)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

# defining test/training datasets
def test_train_model(batch_size, dropout_rate, learn_rate, momentum, loss_label_smoothing):
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=batch_size,
                       interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        win_data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

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
        win_data_dir,  subset="training", shuffle=True, **dataflow_kwargs)

    print("Building model with", MODULE_HANDLE)

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
      optimizer=tf.keras.optimizers.SGD(lr=learn_rate, momentum=momentum),
      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=loss_label_smoothing),
      metrics=['accuracy'])

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    # Run
    hist = model.fit(
        train_generator,
        epochs=1, steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps).history


    print(hist['accuracy'][0])
    print(type(hist['accuracy']))
    acc_list.extend(hist['accuracy'])
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
    #saved_model_path = win_model_save_dir
    #tf.saved_model.save(model, saved_model_path)

test_train_model(36, dropout_rate, learn_rate, momentum, loss_label_smoothing)
print(acc_list)
print(acc_list.max())

