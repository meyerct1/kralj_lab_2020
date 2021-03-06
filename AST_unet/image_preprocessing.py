# Preprocessing/Data Manipulation functions for Kralj-Lab project (ML training on bacterial antibiotic resistance).
########################################################

import cv2
import numpy as np
from PIL import Image, ImageChops
import shutil
import math
import glob
import random
import os, sys

print(cv2.__version__)

#For segmenting images
import skimage.segmentation as seg
import skimage.filters as filters
from skimage.color import rgb2gray


# todo
###

########################################################

# Functions:

# Getting images code, takes all videos in video_dir, extracts frames and saves frames in a folder with the same name
# as the video in save_dir.
def frame_extraction(video_dir, save_dir,regexp = '.avi', trim=True, max_frame=np.inf):
    directory = os.fsencode(video_dir)    # video directory
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        fileloc = video_dir + os.sep + filename  # file location as string
        # filenoext = os.path.splitext(filename)[0]
        if filename.endswith(regexp):  # all .avi videos
            vidcap = cv2.VideoCapture(fileloc)  # read video
            success, image = vidcap.read()
            count = 0
            success = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Note CV2 reads in the data as BGR not RGB
            image = Image.fromarray(image,'RGB')
            if not os.path.exists(save_dir + os.sep + filename):
                os.makedirs(save_dir + os.sep + filename)
            while success:  # every time a new image is detected
                framename = "frame%d.png" % (count)    ###XXX Modified by CM to save as 3Channel images XXX#####
                save_name = save_dir + os.sep + filename + os.sep + framename

                #Moving trim to within frame extraction in order to reduce the amount of IO operations
                #Only calculate trim based on first image.
                if trim and count == 0:
                    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))  # looks at top left pixel to determine the border color
                    diff = ImageChops.difference(image, bg)
                    diff = ImageChops.add(diff, diff, 2.0, -1)
                    bbox = diff.getbbox()  # creates a mask
                    if bbox:
                        image = image.crop(bbox)
                elif trim and bbox:
                    image = image.crop(bbox)

                image.save(save_name)

                #Read in next image
                success, image = vidcap.read()
                #Stop at the maximum frame.
                if count>max_frame:
                    success = False
                if success:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image,'RGB')
                    count += 1

            print('Extracted:' + str(count) + ' frames into ' + save_dir + os.sep + filename)

        else:
            continue


# Greyscale image diff code for the hyperspectral difference images.
# Looks at images in read_dir generated during file extraction
# matching the frames specified (first_frame, last_frame). Then calculates the difference
# in greyscale pixel values between successive frames starting from
# and generates a new image to be saved in save_dir.
def diff_imager_hyper(read_dir, save_dir, total_time, max_little_delta,chopsize=256):
    for subdir, subdirList, fileList in os.walk(read_dir):
        if subdir.endswith(".avi"):  # if the folder ends with .avi, means it was created by frame_extraction
            #print(subdir)  # for every subdirectory in the read_dir

            dirname = os.fsdecode(subdir)
            img1 = cv2.imread(subdir + os.sep + 'frame%d.png' % 0)

            width, height, _ = img1.shape
            num_chop_im = 0
            for _ in range(0,width,chopsize):
                for _ in range(0,height,chopsize):
                    num_chop_im = num_chop_im + 1

            hyperspec_diff = np.zeros((chopsize**2,max_little_delta,total_time-max_little_delta,num_chop_im))
            for e1,little_delta in enumerate(np.linspace(1,total_time-max_little_delta,total_time-max_little_delta)):
                 for e2,fr in enumerate(np.arange(max_little_delta,total_time)):
                    # read in the desired first and last frames
                    img1 = cv2.imread(subdir + os.sep + 'frame%d.png' % fr)
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) #Note CV2 reads in the data as BGR not RGB
                    img2 = cv2.imread(subdir + os.sep + 'frame%d.png' % int(fr-little_delta))
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) #Note CV2 reads in the data as BGR not RGB
                    # try converting these to doubles -- to increase resolution.
                    # diff has the required difference data
                    try:
                        diff = np.abs(img1.astype(np.double) - img2.astype(np.double)).astype(np.uint8)
                    except ValueError:      # in case for some reason the images were trimmed improperly, skips the iteration
                        print("ValueError encountered")
                        continue
                    # Convert from array and save as image
                    img = Image.fromarray(diff,'RGB').convert('L')

                    cntr = 0
                    # Save Chops of original image
                    for x in range(0, width, chopsize):
                        for y in range(0, height, chopsize):
                            if x + chopsize < width and y + chopsize < height:  # checks if boundaries good
                                box = (x,
                                       y,
                                       x + chopsize,
                                       y + chopsize)
                            else:  # re-indexes box if boundaries bad (3 cases)
                                if x + chopsize > width and y + chopsize > height:
                                    box = (width - chopsize - 1,
                                           height - chopsize - 1,
                                           width - 1,
                                           height - 1)
                                else:
                                    if x + chopsize > width:
                                        box = (width - 1 - chopsize,
                                               y,
                                               width - 1,
                                               y + chopsize)
                
                                    else:  # (y + chopsize > height)
                                        box = (x,
                                               height - chopsize - 1,
                                               x + chopsize,
                                               height - 1)
                
                            hyperspec_diff[:,e1,e2,cntr] = np.array(img.crop(box)).ravel(order='F')


            # Figure out which video the frames came from and add the save file name.
            names = dirname.split(os.sep)
            savename = names[-1]
            for e1 in range(hyperspec_diff.shape[2]):
                for e2 in range(hyperspec_diff.shape[3]):
                    img = Image.fromarray(hyperspec_diff[:,:,e1,e2],mode='L')
                    fname = '%s %s - indx(%d - %d).png'%((save_dir+os.sep),savename,e1,e2)
                    img.save(fname)

        else:
            continue



# Greyscale image diff code. Looks at images in read_dir for images matching names generated by frame_extraction and
# matching the frames specified (first_frame, last_frame). Then calculates the difference in greyscale pixel values
# and generates a new image to be saved in save_dir.
def diff_imager(read_dir, save_dir, first_frame, last_frame,to_chop=True,chopsize=299):
    for subdir, subdirList, fileList in os.walk(read_dir):
        if subdir.endswith(".avi"):  # if the folder ends with .avi, means it was created by frame_extraction
            #print(subdir)  # for every subdirectory in the read_dir

            dirname = os.fsdecode(subdir)

            # read in the desired first and last frames
            img1 = cv2.imread(subdir + os.sep + 'frame%d.png' % first_frame)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) #Note CV2 reads in the data as BGR not RGB

            img2 = cv2.imread(subdir + os.sep + 'frame%d.png' % last_frame)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) #Note CV2 reads in the data as BGR not RGB

            # try converting these to doubles -- to increase resolution.
            # diff has the required difference data
            try:
                diff = np.abs(img1.astype(np.double) - img2.astype(np.double)).astype(np.uint8)
            except ValueError:      # in case for some reason the images were trimmed improperly, skips the iteration
                print("ValueError encountered")
                continue

            # Convert from array and save as image
            img = Image.fromarray(diff,'RGB').convert('L')
            # Figure out which video the frames came from and add the save file name.
            names = dirname.split(os.sep)
            savename = names[-1]
            if to_chop:
                fname = "%s - diff(%d - %d).png" % (savename, first_frame, last_frame)
                image_chop_direct(img,fname,save_dir,chopsize=chopsize)
            else:
                fname = "%s %s - diff(%d - %d).png" % ((save_dir + os.sep), savename, first_frame, last_frame)
                img.save(fname)


        else:
            continue

def image_chop_direct(img, fname, dest, chopsize):

    width, height = img.size
    #print(str(img.size))

    # Save Chops of original image
    for x in range(0, width, chopsize):
        for y in range(0, height, chopsize):
            if x + chopsize < width and y + chopsize < height:  # checks if boundaries good
                box = (x,
                       y,
                       x + chopsize,
                       y + chopsize)
            else:  # re-indexes box if boundaries bad (3 cases)
                if x + chopsize > width and y + chopsize > height:
                    box = (width - chopsize - 1,
                           height - chopsize - 1,
                           width - 1,
                           height - 1)
                else:
                    if x + chopsize > width:
                        box = (width - 1 - chopsize,
                               y,
                               width - 1,
                               y + chopsize)

                    else:  # (y + chopsize > height)
                        box = (x,
                               height - chopsize - 1,
                               x + chopsize,
                               height - 1)

            im = img.crop(box)
            #print('%s.x%03d.y%03d.png' % (dest + fname, x, y))
            im.save('%s.x%03d.y%03d.png' % (dest + fname, x, y))      # save as png


# Crops an image into multiple pieces of size (chopsize, chopsize)
def image_chop(infile, dest, chopsize):

    img = Image.open(infile)
    width, height = img.size
    #print(str(img.size))

    # get the name of the file
    split1 = infile.split(os.sep)
    split2 = split1[-1].split(".")
    fname = split2[0] + "." + split2[1]

    # Save Chops of original image
    for x in range(0, width, chopsize):
        for y in range(0, height, chopsize):
            if x + chopsize < width and y + chopsize < height:  # checks if boundaries good
                box = (x,
                       y,
                       x + chopsize,
                       y + chopsize)
            else:  # re-indexes box if boundaries bad (3 cases)
                if x + chopsize > width and y + chopsize > height:
                    box = (width - chopsize - 1,
                           height - chopsize - 1,
                           width - 1,
                           height - 1)
                else:
                    if x + chopsize > width:
                        box = (width - 1 - chopsize,
                               y,
                               width - 1,
                               y + chopsize)

                    else:  # (y + chopsize > height)
                        box = (x,
                               height - chopsize - 1,
                               x + chopsize,
                               height - 1)

            # print('%s %s' % (infile, box))
            im = img.crop(box)
            # print('%s.x%03d.y%03d.png' % (dest + fname, x, y))
            im.save('%s.x%03d.y%03d.png' % (dest + fname, x, y))      # save as png


# Splits the data into test and train datasets. Takes all images in origin_dir and splits 80/20 into train_dir and
# val_dir respectively. Folders are created (if they don't already exist) labeling the images as origninating from
# origin_dir.
def test_train_split(origin_dir, train_dir, val_dir):
    directory = origin_dir.split(os.sep)
    extname = directory[-1]

    if not os.path.exists(train_dir + os.sep + extname + " train"):
        os.makedirs(train_dir + os.sep + extname + " train")
    if not os.path.exists(val_dir + os.sep + extname + " val"):
        os.makedirs(val_dir + os.sep + extname + " val")

    file_list = [file for file in
                 glob.glob(origin_dir + "**/*.png", recursive=True)]  # list of all files in the directory
    random.shuffle(file_list)  # randomizes the list
    print("n")
    print(len(file_list))
    file_count = len(file_list)
    train_sz = math.ceil(file_count * 0.8)
    test_sz = math.floor(file_count * 0.2)
    print("train")
    print(train_sz)
    print("test")
    print(test_sz)

    for i in range(train_sz):  # assigns files to train directory
        filename = file_list[i]
        names = filename.split(os.sep)
        savename = names[-1]

        fileloc = origin_dir + os.sep + savename  # file location as string
        filedest = train_dir + os.sep + extname + " train/" + savename  # file dest as string
        shutil.copyfile(fileloc, filedest)  # copies file into new folder
        print(filedest)

    for j in range(test_sz):  # assigns files to test directory
        filename = file_list[j + train_sz - 1]  # need to account for indexing
        names = filename.split(os.sep)
        savename = names[-1]

        directory = origin_dir.split(os.sep)
        extname = directory[-1]

        fileloc = origin_dir + os.sep + savename  # file location as string
        filedest = val_dir + os.sep + extname + " val/" + savename  # file dest as string
        print(fileloc)
        print(filedest)
        shutil.copyfile(fileloc, filedest)  # copies file into new folder
        print(filedest)


# Cleans up for testing purposes. (deletes files and folders of a given extention)
def cleaner_upper(dir, ext):
    if ext == "folder":
        filelist = [file for file in glob.glob(dir + "/*/", recursive=True)]
        for f in filelist:
            shutil.rmtree(f)  # unlinks all the folders
    else:
        filelist = [file for file in glob.glob(dir + "**/*.%s" % (ext), recursive=True)]
        for f in filelist:
            os.remove(f)  # removes files


# Finds specific images in a folder (used for looking up specific frames)
def image_finder(origin_dir, dest_dir, frame_number):
    for subdir, subdirList, fileList in os.walk(origin_dir):
        print(subdir)  # for every subdirectory in the origin_dir
        if subdir.endswith(".avi"):  # if the folder ends with .avi, means it was created by frame_extraction
            dirname = os.fsdecode(subdir)
            img = cv2.imread(subdir + '/frame%d.jpg' % frame_number)
            img = Image.fromarray(img)
            names = dirname.split(os.sep)
            savename = names[-1]
            img.save("%s %s frame%d.png" % ((dest_dir + os.sep), savename, frame_number))
        else:
            continue


# Resize all images in a directory to a given size
def resize_all(path, size):
    dirs = os.listdir(path)
    for file in dirs:
        if file.endswith('.png') or file.endswith('.jpg'):
            print(file)
            im = Image.open(path + file)  # open with PIL library
            f, e = os.path.splitext(path + file)
            imResize = im.resize((size, size), Image.ANTIALIAS)  # resize the image
            imResize.save(f + ' resized.png', 'PNG', quality=90)
            print(imResize.size)  # test print
            print(f + ' resized.png')


# Moves .png files from one movie to a single folder
def files_to_folders(path, ext):
    num_files = len(os.listdir(path))
    while num_files != 0:
        i = 0    # arbitrary range that will capture all potential movie names (i.e. 8_movie, 22_movie...)
        for file in os.listdir(path):
            if file.endswith(".png") | file.endswith(".jpg"):
                print(path + os.sep + file)
                split = file.split(".")
                if (split[0] == (" " + str(i) + ext)):  # if file matches pattern
                    if not os.path.exists(path + os.sep + split[0]):  # create a new folder for the movie in none exists
                        os.makedirs(path + os.sep + split[0])
                    shutil.move(path + os.sep + file, path + os.sep + split[0])  # move file to the folder
                    num_files -= 1      # decrement loop variable
                    print(split[0])
                    print("s")
        i += 1



#  Removes borders of an image (note: overwrites original file)
def trim(path):
    im = Image.open(path)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))  # looks at top left pixel to determine the border color
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()  # creates a mask
    if bbox:
        cropped = im.crop(bbox)
        cropped.save(path)  # saves image


# Randomizes the names of all files in a directory (used for shuffling data before training a model)
# No two files will share a name, this operation cannot be undone
def randomize_names(path):
    # List files
    dirs = os.listdir(path)

    # Stores previously used names
    prevs = []

    for file in dirs:
        if file.endswith('.png') or file.endswith('.jpg'):

            rand = str(random.randint(100000, 999999)) # random six digit number as string

            # ensures names are not repeated
            while rand in prevs:
                rand = str(random.randint(100000, 999999))

            prevs.append(rand)  # saves to avoid repitition

            # Extract file extension
            split1 = file.split(os.sep)
            split2 = split1[-1].split(".")
            ext = "." + split2[-1]

            dir = path + os.sep

            os.rename(dir + file, dir + rand + ext)
            print(file + " was renamed " + dir + rand + ext)

