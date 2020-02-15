# kralj_lab_2020
E. Coli Antibiotic Resistance Identification Project

Doctor's offices need to make quick decisions when prescribing patients an antibiotic to deal with a potential infection. 
Because of this, doctors often prescribe an antibiotic that works a majority of the time, but may not actually be affective
against a specific patient's infection. 
This is because tests of bacterial samples take several hours to culture and identify.
By the time a classification can occur, the patient has already left the waiting area (>60min) with their possibly incorrect prescription and is unlikely to return to pick up a new drug based on lab results. 
To adress this problem, a machine learning framework is being developed in order to quickly identify bacterial samples based on their resistance to antibiotics. 


*main_script.py
  The file that runs the machine learning model. Currently using the Keras wrapper on Tensorflow.
  
*image_preprocessing.py
  Functions that help with processing images to feed into the model.
  
*diff_imagery.py
  Code that takes two greyscale images, calculates their difference in greyscale pixel values, and creates a corresponding images based on the difference. Also included in image_preprocessing.py.
