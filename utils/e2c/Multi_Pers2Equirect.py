import os
import sys
sys.path.append('.')
sys.path.append('..')

import cv2
import numpy as np
import e2c.Pers2Equirect as P2E

from matplotlib import pyplot as plt

class Perspective:
    def __init__(self, img_array , F_P_T_array, channel=3):
        
        assert len(img_array)==len(F_P_T_array)
        
        self.img_array = img_array
        self.F_P_T_array = F_P_T_array
        self.channel = channel
    

    def GetEquirec(self, height:int=512, width:int=1024):
        #
        # PHI is left/right angle, THETA is up/down angle, both in degree
        #
        merge_image = np.zeros((height, width, self.channel))
        merge_mask = np.zeros((height, width, self.channel))

        for img, [F,P,T] in zip (self.img_array, self.F_P_T_array):
            per = P2E.Perspective(img, F, P, T, channel=self.channel)        # Load equirectangular image
            img , mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            merge_image += img
            merge_mask +=mask
        merge_mask = np.where(merge_mask==0,1,merge_mask)
        merge_image = (np.divide(merge_image,merge_mask))
        
        return merge_image
        