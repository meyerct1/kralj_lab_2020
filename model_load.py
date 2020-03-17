#This file allows tf models to be viewed (protbuf .pb files)

import tensorflow as tf

model_path = '/Library/ML Data/Antibiotic videos/Models/diff_augmented_5.10,15.20'
saved_model = tf.keras.models.load_model(model_path)

module_selection = ("inception_v3", 299) #@param ["(\"mobilenet_v2_100_224\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
handle_base, pixels = module_selection

MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)

saved_model.build((None,)+IMAGE_SIZE+(3,))
saved_model.summary()
# show me the model.
saved_model.show()
