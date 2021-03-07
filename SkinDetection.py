import ast
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# reduce to 32
def reduce (channel):
  x, y = channel.shape
  for i in range (x):
    for j in range (y):
      channel[i][j] = channel[i][j] // 8
  return channel

with open('savedData.txt') as f:
    pourcentagePeau = float(f.readline())
    pourcentageNonPeau = 1 - pourcentagePeau

    dictionaryPeau = ast.literal_eval(f.readline())
    dictionaryNonPeau = ast.literal_eval(f.readline())

    # faire les figures
    # xdata, ydata, zdataPeau, zdataNonPeau = [], [], [], []
    # for i in range(32):
    #     for j in range(32):
    #         xdata.append(i)
    #         ydata.append(j)
    #         zdataPeau.append(dictionaryPeau[(i, j)])
    #         zdataNonPeau.append(dictionaryNonPeau[(i, j)])
    #
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    #
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax1.plot3D(xdata, ydata, zdataPeau, 'gray')
    # ax1.title.set_text('Peau')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # ax2.plot3D(xdata, ydata, zdataNonPeau, 'gray')
    # ax2.title.set_text('Non Peau')
    # plt.show()

    # demande le fichier a detecter
    Tk().withdraw()
    filename = askopenfilename()
    img = cv2.imread(filename)

    # mask
    filename = filename[:-3]
    print(filename)
    mask = cv2.imread(filename + 'png')

    # conversion en lab
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # reduction de a et b en 32 bits
    reduce_a = reduce(a_channel)
    reduce_b = reduce(b_channel)
    x,y = reduce_a.shape

    # construire resultat
    skinImage = np.zeros(img.shape, np.uint8)
    for i in range(x):
        for j in range(y):
            a = reduce_a[i][j]
            b = reduce_b[i][j]

            # naiveBayes
            p = dictionaryPeau[a,b] * pourcentagePeau
            np = dictionaryNonPeau[a,b] * pourcentageNonPeau
            nb = p/(p+np)
            # print(str(nb))

            if nb < 0.5:
                skinImage[i][j][0] = 0
                skinImage[i][j][1] = 0
                skinImage[i][j][2] = 0
            else:
                skinImage[i][j][0] = 255
                skinImage[i][j][1] = 255
                skinImage[i][j][2] = 255

    # afficher
    cv2.imshow("imageOriginal", img)
    cv2.imshow("mask", mask)
    cv2.imshow("resultat", skinImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()