#Eugene Miller
#Reads in a saved model trained on treated/untreated data for the antibiotic resistance classification project.

import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
import statistics


# Dimensions of model architecture ("inception_v3", 299)
img_width = 299
img_height = 299

# Path to model to be loaded
loaded = tf.keras.models.load_model('/Library/ML Data/kralj-lab.tmp/Models/Combined Data No Dropout')

print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)


print("NeuralNet has {} trainable variables: {}, ...".format(
          len(loaded.trainable_variables),
          ", ".join([v.name for v in loaded.trainable_variables[:5]])))

# Parameters for model metrics
loaded.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
               optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), metrics=['accuracy'])

# Gives model predictions for all folders in a given directory.
def predictions(rootdir):

    print(rootdir)

    d = os.listdir(rootdir)

    if '.DS_Store' in d:    # deletes .DS_Store file from dir list, if not done the program will crash
        ind = d.index('.DS_Store')
        del d[ind]

    l = len(d)
    i = 0

    print(d)

    total_treated = 0
    total_untreated = 0

    for dir in d:

        n = 0

        curr = rootdir + dir        # current path for loop iteration

        unteated_sum = 0
        treated_sum = 0

        untreated_list = []         # list of predictions that will be used to compute the overall mean for each subfolder
        treated_list = []

        for file in os.listdir(curr):

            if file.endswith(".png"):
                p = os.path.join(curr, file)

                test_image = image.load_img(p, target_size=(img_width, img_height))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                result = loaded.predict(test_image)

                unteated_sum = unteated_sum + result[0,0]
                treated_sum = treated_sum + result[0,1]

                n = n + 1

        if n == 0:
             print("no .png files in this folder")
             untreated_avg = 0
             treated_avg = 0

        else:
            untreated_avg = unteated_sum/n
            treated_avg = treated_sum/n

            treated_list.append(untreated_avg)
            untreated_list.append(treated_avg)

        total_untreated = total_untreated + untreated_avg
        total_treated = total_treated + treated_avg
        print(d[i] + '\n' + "treated avg: " + str(untreated_avg) +
                             " untreated avg: " + str(treated_avg) + '\n')

        i = i + 1

    print("total average treated: " + str(statistics.mean(treated_list)) + " untreated: " + str(statistics.mean(untreated_list)))

rootdir = '/Library/ML Data/kralj-lab.tmp/Testing/Treated Folders/'
predictions(rootdir)