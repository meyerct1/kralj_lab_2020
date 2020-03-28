#This file allows tf models to be viewed (protbuf .pb files)

import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image
from skimage.io import imread
import os
import statistics

img_width = 299
img_height = 299

loaded = tf.keras.models.load_model('/Library/ML Data/Bionicles.tmp/Models/New')

print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)


print("BioticNet has {} trainable variables: {}, ...".format(
          len(loaded.trainable_variables),
          ", ".join([v.name for v in loaded.trainable_variables[:5]])))

loaded.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
               optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), metrics=['accuracy'])

unteated_sum = 0
treated_sum = 0
rootdir = '/Library/ML Data/Bionicles.tmp/Testing/Treated Folders/'
i = 0
for subdir, dirs, files in os.walk(rootdir):

    n = 0

d = os.listdir(rootdir)

if '.DS_Store' in d:
    ind = d.index('.DS_Store')
    del d[ind]

len = len(d)
i = 0
print(d)
total_treated = 0
total_untreated = 0
for dir in d:

    n = 0

    curr = rootdir + dir

    unteated_sum = 0
    treated_sum = 0

    untreated_list = []
    treated_list = []

    for file in os.listdir(curr):

        if file.endswith(".png"):
            p = os.path.join(curr, file)

            test_image = image.load_img(p, target_size=(img_width, img_height))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = loaded.predict(test_image)

            img = Image.open(p)

            unteated_sum = unteated_sum + result[0,0]
            treated_sum = treated_sum + result[0,1]

            n = n + 1

    if n == 0:
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

total_treated_avg = total_treated/len
total_untreated_avg = total_untreated/len
print("total average treated: " + str(statistics.mean(treated_list)) + " untreated: " + str(statistics.mean(untreated_list)))
