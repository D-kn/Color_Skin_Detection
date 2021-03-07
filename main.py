
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pixPeau = 0
pixNonPeau = 0

# print(image.shape)
img = None
mask = None


# dictionary to store value
dict = {}
dictNonPeau = {}


for i in range (32):
  for j in range (32):
    dict[(i,j)] = 0
    dictNonPeau[(i,j)] = 0


def count_pixel_skin(a_channel, b_channel, mask):
  binary = convert_to_binary(mask)
  x,y = binary.shape

  global pixPeau
  global pixNonPeau

  reduced_a = reduce(a_channel)
  reduced_b = reduce(b_channel)
  for i in range(x):
    for j in range(y):
      a = reduced_a[i][j]
      b = reduced_b[i][j]
      if binary[i][j] == 255:
        dict[(a,b)] += 1
        pixPeau += 1
      elif binary[i][j] == 0:
        dictNonPeau[(a, b)] += 1
        pixNonPeau += 1

# reduce to 32
def reduce (channel):
  x, y = channel.shape
  for i in range (x):
    for j in range (y):
      channel[i][j] = channel[i][j] // 8
  return channel

# convert to black and white
def convert_to_binary(mask):
  grayImage = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
  return blackAndWhiteImage

# counting pixel for skin
for filenameWithExtension in os.listdir():
  ext = filenameWithExtension[-3:]
  filename = filenameWithExtension[:-3]
  if ext == "jpg":
     img = cv2.imread(filenameWithExtension)
     mask = cv2.imread(filename + 'png')

     # convert to lab image
     lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
     l_channel, a_channel, b_channel = cv2.split(lab_image)

     # reducing a and b
     count_pixel_skin(a_channel, b_channel, mask)

#pourcentage des peaux et non peaux
pourcentagePeau = pixPeau / (pixPeau + pixNonPeau)
pourcentageNonPeau = 1 - pourcentagePeau

# sauvegarder les donnees analyses dans un fichier
f = open("savedData.txt", "w")
f.write(str(pourcentagePeau) + "\n")
f.write(str(dict))
f.write("\n")
f.write(str(dictNonPeau))
f.close()


# final