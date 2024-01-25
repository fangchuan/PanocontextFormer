import os
import cv2 
import e2c.Equirect2Pers as E2P
import glob
import argparse
from PIL import Image
import numpy as np

import e2c.Multi_Pers2Equirect as m_P2E

def panorama2cube(input_dir:str, output_dir:str):

    cube_size = 640

    os.makedirs(output_dir, exist_ok=True)

    all_room_lst = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f)) and f.isdigit()]
    all_room_lst.sort(key=lambda x: int(x))

    for index in range(len(all_room_lst)):
        image_path = os.path.join(input_dir, all_room_lst[index], 'rgb.png')
        rgb_img = np.array(Image.open(image_path))
        equ = E2P.Equirectangular(rgb_img)    # Load equirectangular image
        #
        # FOV unit is degree
        # theta is z-axis angle(right direction is positive, left direction is negative)
        # phi is y-axis angle(up direction positive, down direction negative)
        # height and width is output image dimension
        #

        out_dir = os.path.join(output_dir, all_room_lst[index], 'cubemap')
        os.makedirs(out_dir, exist_ok=True)

        img = equ.GetPerspective(90, 0, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output1 = out_dir +  '/front.png'
        Image.fromarray(img.astype(np.uint8)).save(output1)

        img = equ.GetPerspective(90, 90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output2 = out_dir + '/right.png' 
        Image.fromarray(img.astype(np.uint8)).save(output2)


        img = equ.GetPerspective(90, 180, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output3 = out_dir + '/back.png' 
        Image.fromarray(img.astype(np.uint8)).save(output3)


        img = equ.GetPerspective(90, 270, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output4 = out_dir + '/left.png' 
        Image.fromarray(img.astype(np.uint8)).save(output4)

        # img = equ.GetPerspective(90, 0, 90, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        # output5 = out_dir + '/top.png' 
        # cv2.imwrite(output5, img)

        # img = equ.GetPerspective(90, 0, -90, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        # output6 = out_dir + '/bottom.png' 
        # cv2.imwrite(output6, img)

        # get different fov
        for fov in range(30, 150, 30):
            img = equ.GetPerspective(fov, 0, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
            output_image_path = os.path.join(out_dir, f'front_fov_{fov}.png')
            Image.fromarray(img.astype(np.uint8)).save(output_image_path)
            per = m_P2E.Perspective([img], [[fov, 0, 0]], channel=img.shape[-1])
            pano_img = per.GetEquirec(height=512, width=1024)
            output_pano_image_path = os.path.join(out_dir, f'front_fov_{fov}_pano.png')
            Image.fromarray(pano_img.astype(np.uint8)).save(output_pano_image_path)
            
            img = equ.GetPerspective(fov, 90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
            output_image_path = os.path.join(out_dir, f'right_fov_{fov}.png')
            Image.fromarray(img.astype(np.uint8)).save(output_image_path)
            per = m_P2E.Perspective([img], [[fov, 90, 0]], channel=img.shape[-1])
            pano_img = per.GetEquirec(height=512, width=1024)
            output_pano_image_path = os.path.join(out_dir, f'right_fov_{fov}_pano.png')
            Image.fromarray(pano_img.astype(np.uint8)).save(output_pano_image_path)
            
            img = equ.GetPerspective(fov, 180, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
            output_image_path = os.path.join(out_dir, f'back_fov_{fov}.png')
            Image.fromarray(img.astype(np.uint8)).save(output_image_path)
            per = m_P2E.Perspective([img], [[fov, 180, 0]], channel=img.shape[-1])
            pano_img = per.GetEquirec(height=512, width=1024)
            output_pano_image_path = os.path.join(out_dir, f'back_fov_{fov}_pano.png')
            Image.fromarray(pano_img.astype(np.uint8)).save(output_pano_image_path)
            
            img = equ.GetPerspective(fov, 270, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
            output_image_path = os.path.join(out_dir, f'left_fov_{fov}.png')
            Image.fromarray(img.astype(np.uint8)).save(output_image_path)
            per = m_P2E.Perspective([img], [[fov, 270, 0]], channel=img.shape[-1])
            pano_img = per.GetEquirec(height=512, width=1024)
            output_pano_image_path = os.path.join(out_dir, f'left_fov_{fov}_pano.png')
            Image.fromarray(pano_img.astype(np.uint8)).save(output_pano_image_path)
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='/media/hkust/PRODATA1/debug_20230105/office_0_000/', help='room directory contains all viewpoints images')
    parser.add_argument('--output_dir', type=str, default='/media/hkust/PRODATA1/debug_20230105/office_0_000/', help='output directory for perspective images')

    config = parser.parse_args()

    panorama2cube(config.input_dir, config.output_dir)