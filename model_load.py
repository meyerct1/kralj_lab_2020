#This file allows tf models to be viewed (protbuf .pb files)

import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image
from skimage.io import imread

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
img_path = '/Library/ML Data/Antibiotic videos/Data/Treated Diff/ 7_movie.avi - diff(11 - 16) resized.png'

img = Image.open('/Library/ML Data/Antibiotic videos/Data/Treated Diff/ 7_movie.avi - diff(11 - 16) resized.png')
print(img.size)

loaded = tf.keras.models.load_model('/Library/ML Data/Antibiotic videos/Models/Batch 2/0 to 30 rg = 5')

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

test_image = image.load_img(img_path, target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
#test_image = test_image.reshape(img_width, img_height)
result = loaded.predict(test_image)
print(loaded.predict(test_image))
print(result)

img2 = '/Library/ML Data/Antibiotic videos/Data/Untreated Diff/ 1_movie.avi - diff(11 - 16).png'

test_image = image.load_img(img2, target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = loaded.predict(test_image)
print(loaded.predict(test_image))
print(result)