import cv2
import numpy as np 
from math import exp
import time


class GaussianHeatmap:

  def __init__(self, GPR, border, imgSize=512) -> None:

       
      GrayscaleImage = np.zeros((imgSize,imgSize),np.uint8)
      print("Creating GrayScaleImage...")
      StartTime = time.time()

      # Method 1: meshgrid
      '''
      X,Y = np.mgrid[0:imgSize, 0:imgSize]
      XX = np.hstack((X.reshape(imgSize**2, -1), Y.reshape(imgSize**2, -1)))
      XX = XX/imgSize*border
      #print(XX.shape)
      YY = GPR.sample_y(XX.reshape(-1,2))
      #YYmin = min(YY)
      #YYmax = max(YY)
      for i in range(imgSize):
        for j in range(imgSize): 
          GrayscaleImage[i,j] = YY[i*imgSize+j]*255
      '''

      # Method 2: one by one 
      '''
      for i in range(imgSize):
        for j in range(imgSize):
          pts = np.array([i, j])/imgSize* border 
          y = GPR.sample_y(pts.reshape(-1,2), n_samples = 1)
          GrayscaleImage[i,j] = y * 255
      '''

      print("Total time cost:", time.time()-StartTime)
      self.GaussianHeatmap = cv2.applyColorMap(GrayscaleImage, cv2.COLORMAP_JET)


  def show(self):
      cv2.imshow('image', self.GaussianHeatmap)
      cv2.waitKey(0)


# scaledGaussian = lambda x : exp(-(1/2)*(x**2))

# imgSize = 128
# isotropicGrayscaleImage = np.zeros((imgSize,imgSize),np.uint8)

# for i in range(imgSize):
#   for j in range(imgSize):

#     distanceFromCenter = np.linalg.norm(np.array([i-imgSize/2,j-imgSize/2]))
#     distanceFromCenter = 2.5*distanceFromCenter/(imgSize/2)
#     scaledGaussianProb = scaledGaussian(distanceFromCenter)
#     isotropicGrayscaleImage[i,j] = np.clip(scaledGaussianProb*255,0,255)

# GaussianHeatmap = cv2.applyColorMap(isotropicGrayscaleImage, 
#                                                   cv2.COLORMAP_JET)
# cv2.imshow('image', GaussianHeatmap)
# cv2.waitKey(0)