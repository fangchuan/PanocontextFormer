import os
import sys
import cv2
import numpy as np

class Perspective:
    def __init__(self, image:np.array , FOV:float, PHI:float, THETA:float, channel=3):
        # self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        self._img = image
        [self._height, self._width, c] = self._img.shape
        # print(f'img shape: {self._img.shape}')
        self.wFOV = FOV
        self.PHI = PHI
        self.THETA = THETA
        # self.hFOV = float(self._height) / self._width * FOV
        self.hFOV = np.rad2deg(2 * np.arctan(self._height * np.tan(np.radians(FOV/2)) / self._width))

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))
        self.channel = channel
        assert self.channel == c

    

    def GetEquirec(self, height, width):
        #
        # PHI is left/right angle, THETA is up/down angle, both in degree
        #

        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.PHI))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.THETA))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0, 1, 0)

        wahts = np.repeat(xyz[:,:,0][:, :, np.newaxis], self.channel, axis=2)        
        # inverse_mask = np.where(xyz[:,:,1]>0, 1, 0)
        # wahts = np.repeat(xyz[:,:,1][:, :, np.newaxis], 3, axis=2)
        xyz[:,:] = xyz[:,:]/wahts
        
        
        lon_map = np.where((-self.w_len < xyz[:,:,1]) & (xyz[:,:,1] < self.w_len) & 
                           (-self.h_len < xyz[:,:,2]) & (xyz[:,:,2] < self.h_len),
                           (xyz[:,:,1]+self.w_len)/2/self.w_len*self._width, 0)
        lat_map = np.where((-self.w_len<xyz[:,:,1]) & (xyz[:,:,1] < self.w_len) &
                           (-self.h_len<xyz[:,:,2]) & (xyz[:,:,2] < self.h_len),
                           (-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height, 0)
        mask = np.where((-self.w_len < xyz[:,:,1]) & (xyz[:,:,1] < self.w_len) &
                        (-self.h_len < xyz[:,:,2]) & (xyz[:,:,2] < self.h_len), 1, 0)
        
        # lon_map = np.where((-self.w_len < xyz[:,:,0]) & (xyz[:,:,0] < self.w_len) & 
        #                    (-self.h_len < xyz[:,:,2]) & (xyz[:,:,2] < self.h_len),
        #                    (xyz[:,:,0]+self.w_len)/2/self.w_len*self._width, 0)

        # lat_map = np.where((-self.w_len<xyz[:,:,0]) & (xyz[:,:,0] < self.w_len) &
        #                    (-self.h_len<xyz[:,:,2]) & (xyz[:,:,2] < self.h_len),
        #                    (-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height, 0)

        # mask = np.where((-self.w_len < xyz[:,:,0]) & (xyz[:,:,0] < self.w_len) &
        #                 (-self.h_len < xyz[:,:,2]) & (xyz[:,:,2] < self.h_len), 1, 0)

        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], self.channel, axis=2)
        while len(persp.shape) != len(mask.shape):
            persp = persp[..., np.newaxis]
        persp = persp * mask
        
        
        return persp , mask
        