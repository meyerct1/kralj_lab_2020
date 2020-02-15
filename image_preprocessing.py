#Eugene Miller
#Preprocessing Image functions for Kralj-Lab project (ML training on bacterial antibiotic resistance).
########################################################

import cv2
import os
import numpy as np
from PIL import Image
from shutil import copyfile
import math
import glob
import random
import shutil
print(cv2.__version__)

#todo
# make sure everything runs smoothly, do a full run through test
## generalize functions one read and one save directory frame_extraction and test_train_split (done)


########################################################

# Directories (frame extraction)
treated_video_dir = "/Library/ML Data/Antibiotic videos/Treated AVIs"
untreated_video_dir = "/Library/ML Data/Antibiotic videos/Untreated AVIs"
treated_save_dir = "/Library/ML Data/Antibiotic videos/Treated Frames"
untreated_save_dir = "/Library/ML Data/Antibiotic videos/Untreated Frames"

# Directories (diff image calculation)
treated_read_dir = "/Library/ML Data/Antibiotic videos/Treated Frames"
treated_saved_dir = "/Library/ML Data/Antibiotic videos/Treated Diff"

untreated_read_dir = "/Library/ML Data/Antibiotic videos/Untreated Frames"
untreated_saved_dir = "/Library/ML Data/Antibiotic videos/Untreated Diff"

train_data = "/Library/ML Data/Antibiotic videos/Train"
test_data = "/Library/ML Data/Antibiotic videos/Test"

# Frame #s for difference images
first_frame = 15
last_frame = 60

########################################################

#Functions:

# Getting images code
def frame_extraction(video_dir, save_dir):
    # Treated directory
    directory = os.fsencode(video_dir)
    for file in os.listdir(directory):
         filename = os.fsdecode(file)
         fileloc = video_dir + "/" + filename # file location as string
         #filenoext = os.path.splitext(filename)[0]
         if filename.endswith(".avi"): # all .avi videos
           vidcap = cv2.VideoCapture(fileloc) # read video
           success, image = vidcap.read()
           count = 0
           success = True
           if not os.path.exists(save_dir + "/" + filename):
            os.makedirs(save_dir + "/" + filename)
           while success: # every time a new image is detected
             framename = "frame%d.jpg" % (count)
             save = save_dir + "/" +  filename + "/" + framename
             print(save)
             cv2.imwrite(save, image)  # save frame as JPEG file
             success, image = vidcap.read()
             print(framename)
             print('Read a new frame: ', success)
             count += 1
             continue
         else:
             continue

# Greyscale image diff code
def diff_imager(read_dir, save_dir, first_frame, last_frame):

    for subdir, subdirList, fileList in os.walk(read_dir):
            print(subdir) # for every subdirectory in the read_dir
            if subdir.endswith(".avi"): # if the folder ends with .avi, means it was created by frame_extraction

                dirname = os.fsdecode(subdir)

                # read in the desired first and last frames
                img1 = cv2.imread(subdir + '/frame%d.jpg' % first_frame)
                img2 = cv2.imread(subdir + '/frame%d.jpg' % last_frame)

            # try converting these to doubles -- to increase resolution.
                # diff has the required difference data
                diff = np.abs(img1.astype(np.uint) - img2.astype(np.uint)).astype(np.uint8)
                print(type(diff))

                # Convert from array and save as image
                img = Image.fromarray(diff)

                # Figure out which video the frames came from and add the save file name.
                names = dirname.split("/")
                savename = names[-1]
                img.save("%s %s - diff(%d - %d).png" % ((save_dir + "/"), savename, first_frame, last_frame))
            else:
                continue

# Splits the data into test and train datasets.
def test_train_split(origin_dir, train_dir, val_dir):

    # code for untreated directory
    directory = origin_dir.split("/")
    extname = directory[-1]

    if not os.path.exists(train_dir + "/" + extname + " train"):

        os.makedirs(train_dir + "/" + extname + " train")
    if not os.path.exists(val_dir + "/" + extname + " val"):
        os.makedirs(val_dir + "/" + extname + " val")

    file_list = [file for file in glob.glob(origin_dir + "**/*.png", recursive=True)] # list of all files in the directory
    random.shuffle(file_list) # randomizes the list
    print("n")
    print(len(file_list))
    file_count = len(file_list)
    train_sz = math.ceil(file_count * 0.8)
    test_sz = math.floor(file_count * 0.2)
    print("train")
    print(train_sz)
    print("test")
    print(test_sz)

    for i in range(train_sz):       # assigns files to train directory
        filename = file_list[i]
        names = filename.split("/")
        savename = names[-1]


        fileloc = origin_dir + "/" + savename  # file location as string
        filedest = train_dir + "/" + extname + " train/" + savename # file dest as string
        copyfile(fileloc, filedest) # copies file into new folder
        print(filedest)

    for j in range(test_sz):        # assigns files to test directory
        filename = file_list[j + train_sz - 1]      # need to account for indexing
        names = filename.split("/")
        savename = names[-1]

        directory = origin_dir.split("/")
        extname = directory[-1]


        fileloc = origin_dir + "/" + savename  # file location as string
        filedest = val_dir + "/" +  extname + " val/" + savename  # file dest as string
        print(fileloc)
        print(filedest)
        copyfile(fileloc, filedest)  # copies file into new folder
        print(filedest)

# Cleans up for testing purposes.
def cleaner_upper(dir, ext):
    if ext == "folder":
        filelist = [file for file in glob.glob(dir + "/*/", recursive=True)]
        for f in filelist:
            shutil.rmtree(f)
    else:
        filelist = [file for file in glob.glob(dir + "**/*.%s" % (ext), recursive=True)]
        for f in filelist:
            os.remove(f)

########################################################

# Main:

#frame_extraction(treated_video_dir, treated_save_dir)
#frame_extraction(untreated_video_dir, untreated_save_dir)

#diff_imager(treated_save_dir, treated_saved_dir, 15, 60)
#diff_imager(untreated_save_dir, untreated_saved_dir, 15, 60)

test_train_split(treated_saved_dir, train_data, test_data)
test_train_split(untreated_saved_dir, train_data, test_data)


#cleaner_upper("/Library/ML Data/Antibiotic videos/Untreated Data/train", "png")
#cleaner_upper("/Library/ML Data/Antibiotic videos/Treated Data/train", "png")
#cleaner_upper("/Library/ML Data/Antibiotic videos/Untreated Data/test", "png")
#cleaner_upper("/Library/ML Data/Antibiotic videos/Treated Data/test", "png")
#cleaner_upper("/Library/ML Data/Antibiotic videos/Treated Frames", "folder")
#cleaner_upper("/Library/ML Data/Antibiotic videos/Untreated Frames", "folder")
#cleaner_upper("/Library/ML Data/Antibiotic videos/Treated Diff", "png")
#cleaner_upper("/Library/ML Data/Antibiotic videos/Untreated Diff", "png")



