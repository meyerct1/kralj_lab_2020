#This file allows tf models to be viewed (protbuf .pb files)

import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image
from skimage.io import imread
import os

#model_path = '/Library/ML Data/Antibiotic videos/Models/diff_augmented_5.10,15.20'
#saved_model = tf.keras.models.load_model(model_path)

#module_selection = ("inception_v3", 299) #@param ["(\"mobilenet_v2_100_224\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
#handle_base, pixels = module_selection

#MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
#IMAGE_SIZE = (pixels, pixels)

#saved_model.build((None,)+IMAGE_SIZE+(3,))
#saved_model.summary()
## show me the model.
#saved_model.show()

img_width = 299
img_height = 299

loaded = tf.keras.models.load_model('/Library/ML Data/Bionicles.tmp/Models/New')

print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)


print("BioticNet has {} trainable variables: {}, ...".format(
          len(loaded.trainable_variables),
          ", ".join([v.name for v in loaded.trainable_variables[:5]])))


#loaded.build()
#loaded.summary()

#background = Image.new('RGBA', png.size, (255,255,255))

#alpha_composite = Image.alpha_composite(background, png)
#alpha_composite.save('foo.jpg', 'JPEG', quality=80)

#png = Image.open(img_path)
#png.load() # required for png.split()
#color=(255, 255, 255)
#x = np.array(png)
#print((x.shape))
#r, g, b, a = np.rollaxis(x, axis=-1)
#r[a == 0] = color[0]
#g[a == 0] = color[1]
#b[a == 0] = color[2]
#x = np.dstack([r, g, b, a])
#png = Image.fromarray(x, 'RGBA')

#alpha = png.split()[-1]
#png.putalpha(alpha)

#background = Image.new("RGB", png.size, (255, 255, 255))
#background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

#png.save("predict1.png")
#print(type(png))
loaded.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
               optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), metrics=['accuracy'])

#test_image = image.load_img(img_path, target_size=(img_width, img_height))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis=0)
##test_image = test_image.reshape(img_width, img_height)
#result = loaded.predict(test_image)
#print(result[0,0])

unteated_sum = 0
treated_sum = 0
rootdir = '/Library/ML Data/Bionicles.tmp/Testing/Tht Treated Folders/'
i = 0
for subdir, dirs, files in os.walk(rootdir):
    #print(dirs)
        #t = open(s + " result.txt", "w+")
        #t.close()
    #print(dirs)
    #print(i)
    #print(dirs[i])
    n = 0
d = os.listdir(rootdir)
del d[2]

i = 0
print("test")
print(d)
for dir in d:
    #print(dir)
    n = 0
    #print(i)
    curr = rootdir + dir
    unteated_sum = 0
    treated_sum = 0
    for file in os.listdir(curr):
        #print(file)
        if file.endswith(".png"):
            p = os.path.join(curr, file)

            test_image = image.load_img(p, target_size=(img_width, img_height))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = loaded.predict(test_image)

            img = Image.open(p)
            #print(img.size)

            unteated_sum = unteated_sum + result[0,0]
            treated_sum = treated_sum + result[0,1]
            #print(treated_sum)

            n = n + 1

    if n == 0:
         untreated_avg = 0
         treated_avg = 0
    else:
         untreated_avg = unteated_sum/n
         treated_avg = treated_sum/n

    print(d[i] + '\n' + "treated avg: " + str(untreated_avg) +
                         " untreated avg: " + str(treated_avg) + '\n')
        #with open('result.txt', "a+") as output_file:
        #    output_file.write(s + '\n' + "untreated avg: " + untreated_avg +
        #                 " treatead avg: " + treated_avg + '\n')
    i = i + 1

