import cv2
import numpy as np 
from math import exp


class Gaussian_heatmap:

  def __init__(self, GPR, border, imgSize=512) -> None:
      self.GPR = GPR
      self.border = border
      self.imgSize= imgSize
      self.GrayscaleImage = np.zeros((imgSize,imgSize),np.uint8)


  def render(self):
      for i in range(self.imgSize):
        for j in range(self.imgSize):
          pts = [i - self.imgSize/2, j - self.imgSize/2] / self.imgSize * self.border * 2 

  def show(self):
      pass


scaledGaussian = lambda x : exp(-(1/2)*(x**2))

imgSize = 128
isotropicGrayscaleImage = np.zeros((imgSize,imgSize),np.uint8)

for i in range(imgSize):
  for j in range(imgSize):

    distanceFromCenter = np.linalg.norm(np.array([i-imgSize/2,j-imgSize/2]))
    distanceFromCenter = 2.5*distanceFromCenter/(imgSize/2)
    scaledGaussianProb = scaledGaussian(distanceFromCenter)
    isotropicGrayscaleImage[i,j] = np.clip(scaledGaussianProb*255,0,255)

GaussianHeatmap = cv2.applyColorMap(isotropicGrayscaleImage, 
                                                  cv2.COLORMAP_JET)
cv2.imshow('image', GaussianHeatmap)
cv2.waitKey(0)