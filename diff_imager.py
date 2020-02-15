# Greyscale image diff code
import cv2
import os
import numpy as np
from PIL import Image
print(cv2.__version__)

read_dir = "/Library/ML Data/Antibiotic videos/Treated Frames/"
save_dir = "/Library/ML Data/Antibiotic videos/Treated Diff"

def diff_imager(read_dir, save_dir):

    first_frame = 15
    last_frame = 60
    for subdir, subdirList, fileList in os.walk(read_dir):
            print(subdir)
            if subdir.endswith(".avi"):
                dirname = os.fsdecode(subdir)
                img1 = cv2.imread(subdir + '/frame%d.jpg' % first_frame)
                img2 = cv2.imread(subdir + '/frame%d.jpg' % last_frame)

            # diff has the required difference data
            # try converting these to doubles -- to increase resolution.

                diff = np.abs(img1.astype(np.uint) - img2.astype(np.uint)).astype(np.uint8)
                print(type(diff))

            # Convert from array and save as image
                img = Image.fromarray(diff)

                names = dirname.split("/")
                savename = names[-1]
                img.save("%s %s - diff(%d - %d).png" % ((save_dir + "/"), savename, first_frame, last_frame))
            else:
                continue

diff_imager(read_dir,save_dir)