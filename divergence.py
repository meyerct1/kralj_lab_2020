#Code to do non-ml classification
import cv2
from scipy.stats import entropy
import math
import os, sys

from image_preprocessing import diff_imager, image_chop, test_train_split

treated_read = 'C:\\Users\\eugmille\\Desktop\\kralj-lab.tmp\\Treated Frames'
untreated_read = 'C:\\Users\\eugmille\\Desktop\\kralj-lab.tmp\\Untreated Frames'

untreated_img_path = 'C:\\Users\\eugmille\\Desktop\\kralj-lab.tmp\\Relative Entropy\\Untreated'
treated_img_path = "C:\\Users\\eugmille\\Desktop\\kralj-lab.tmp\\Relative Entropy\\Treated"

untreated_format_dir = 'C:\\Users\\eugmille\\Desktop\\kralj-lab.tmp\\Relative Entropy\\Formatted Untreated'
treated_format_dir = 'C:\\Users\\eugmille\\Desktop\\kralj-lab.tmp\\Relative Entropy\\Formatted Treated'

def img_kl_divergence(set1, set2):
    arr = []
    spl_set1 = set1.split('\\')
    spl_set2 = set2.split('\\')
    name_set1 = spl_set1[-1]
    name_set2 = spl_set2[-1]
    for e in os.listdir(set1):
        if e.endswith('.png'):
            im1 = cv2.imread(set1 + '\\' + e, cv2.IMREAD_GRAYSCALE)
            im1 = im1/255
            for f in os.listdir(set2):
                if f.endswith('.png'):
                    im2 = cv2.imread(set2 + '\\' + f, cv2.IMREAD_GRAYSCALE)
                    im2 = im2/255
                    if entropy(sum(list(im1)) / float(len(list(im1))), sum(list(im2)) / float(len(list(im2)))) == float("inf"):
                        #If one value is zero and the other isn't
                        arr.append(5)
                    elif math.isnan(entropy(sum(list(im1)) / float(len(list(im1))), sum(list(im2)) / float(len(list(im2))))):
                        #If both values are zero.   
                        arr.append(0)
                    else:
                        #print(entropy(sum(list(im1)) / float(len(list(im1))), sum(list(im2)) / float(len(list(im2)))))  #Reduce images to singular value, calc entropy.
                        arr.append(entropy(sum(list(im1)) / float(len(list(im1))), sum(list(im2)) / float(len(list(im2)))))     #Add to list
    print("Mean " + name_set1 + " to " + name_set2 + ": " + str(sum(arr) / float(len(arr))))   # Calculate mean of list
    print("Max: "+ str(max(arr)))

img_kl_divergence(untreated_format_dir, treated_format_dir)
img_kl_divergence(treated_format_dir, untreated_format_dir)
img_kl_divergence(untreated_format_dir, untreated_format_dir)
img_kl_divergence(treated_format_dir, treated_format_dir)


#When is this function equal to inf?





