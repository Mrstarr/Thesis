import cv2
import numpy as np 
from math import exp
import time


class GaussianHeatmap:

  def __init__(self, GPR, border, imgSize=512) -> None:

       
      MeanGrayscaleImage = np.zeros((imgSize,imgSize),np.float)
      VarGrayscaleImage = np.zeros((imgSize,imgSize),np.float)
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
      for i in range(imgSize):
        for j in range(imgSize):
          pts = np.array([i, j])/imgSize* border 
          #y = GPR.sample_y(pts.reshape(-1,2), n_samples = 1)
          y, std = GPR.predict(pts.reshape(-1,2), return_std = True)
          MeanGrayscaleImage[i,j] = y
          VarGrayscaleImage[i,j] = std
            
      print("Total time cost:", time.time()-StartTime)
      MeanGrayscaleImage = cv2.normalize(MeanGrayscaleImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      self.GaussianHeatmap = cv2.applyColorMap(MeanGrayscaleImage, cv2.COLORMAP_JET)
      VarGrayscaleImage = cv2.normalize(VarGrayscaleImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      self.VarianceHeatmap = cv2.applyColorMap(VarGrayscaleImage, cv2.COLORMAP_JET)


  def showMean(self):
      cv2.imshow('MeanMap', self.GaussianHeatmap)
      cv2.waitKey(0)

  def showVar(self):
      cv2.imshow('VarianceMap', self.VarianceHeatmap)
      cv2.waitKey(0)

