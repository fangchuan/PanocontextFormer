import os
import os.path as osp
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PIL import Image
from typing import List
from e2c_lib import multi_Perspec2Equirec as m_P2E
from pano_utils import vis_color_pointcloud

def read_koolai_cubemap_image(image_path:str, image_type:str='albedo')->List[np.ndarray]:
    """ read koolai cubemap image
    0:'r',
    1:'l',
    2:'u'
    3:'d',
    4:'f',
    5:'b',
    Args:
        image_path (str): cubemap image file apth

    Returns:
        numpy.ndarray: _description_
    """
    img = Image.open(image_path)
    w, h = img.size
    face_w = w // 6
    # split into 6 images
    img_list = []
    # front face
    img_list.append(img.crop((4 * face_w, 0, 5 * face_w, h)))
    # right face
    img_list.append(img.crop((0 * face_w, 0, 1 * face_w, h)))
    # back face
    img_list.append(img.crop((5 * face_w, 0, 6 * face_w, h)))
    # left face
    img_list.append(img.crop((1 * face_w, 0, 2 * face_w, h)))
    # up face
    img_list.append(img.crop((2 * face_w, 0, 3 * face_w, h)))
    # down face
    img_list.append(img.crop((3 * face_w, 0, 4 * face_w, h)))
    
    # for img_pil in img_list:
    #     if len(img_pil.split()) >= 3:
    #         img_pil = np.array(img_pil)[:,:,:3]
    #     elif len(img_pil.split()) == 1:
    #         img_pil = np.array(img_pil)
    #         print(f'img_pil: {img_pil.shape}')
    #         img_pil = img_pil[:,:,np.newaxis]
    if len(img.split()) >= 3:
        img_list = np.array([np.array(img)[:,:,:3] for img in img_list])
    elif len(img.split()) == 1:
        # img_list = np.array([np.repeat(np.array(img)[:,:,np.newaxis], 3, axis=2) for img in img_list])
        if image_type == 'depth':
            img_list = np.array([np.array(img)[:,:,np.newaxis].astype(np.uint16) for img in img_list])
            # img_list = np.array([(img/1000.0).astype(np.float) for img in img_list])
        else:
            img_list = np.array([np.array(img)[:,:,np.newaxis].astype(np.uint8) for img in img_list])
    else:
        raise ValueError(f'unsupport image shape: {img.shape}')
        
    # print(f'img_list: {img_list[0].shape}')
    # plt.imshow(img_list[0])
    # plt.show()
    return img_list

def cube2panorama(input_dir:str, output_dir:str, pano_width:int=1024, pano_height:int=512, convert_keys:List[str]=['albedo', 'depth', 'normal', 'instance', 'semantic']):
    """ convert cubemap images to panorama image

    Args:
        input_dir (str): input folder of koolai synthetic dataset
        output_dir (str): processed output folder
        pano_width (int, optional): _description_. Defaults to 1024.
        pano_height (int, optional): _description_. Defaults to 512.
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    convert_images_path_lst = []
    for key in convert_keys:
        convert_images_path_lst.append(osp.join(input_dir, f'rasterize/0/{key}.png'))

    # cubemap fovs, phi and theta angles
    F_P_T_lst = [[90, 0, 0],  # front
                 [90, 90, 0], # right
                 [90, 180, 0], # back
                 [90, 270, 0], # left
                 [90, 0, 90], # up
                 [90, 0, -90]] # down
    for img_type, img_path in zip(convert_keys, convert_images_path_lst):
        faces_img_lst = read_koolai_cubemap_image(img_path, img_type)
        img_channel = faces_img_lst[0].shape[-1]
        per = m_P2E.Perspective(faces_img_lst, F_P_T_lst, channel=img_channel)
        img = per.GetEquirec(pano_height, pano_width)
        if img_type == 'depth':
            img = (img*4).astype(np.uint16)
            img = img.reshape((pano_height, pano_width))
        elif img_type == 'albedo' or img_type == 'normal':
            img = img.astype(np.uint8)
        elif img_type == 'instance' or img_type == 'semantic':
            img = img.astype(np.uint8)
            img = img.reshape((pano_height, pano_width))
        print(f'{img_type}, img shape: {img.shape}')
        plt.imshow(img)
        plt.show()
        img = Image.fromarray(img)
        img.save(osp.join(output_dir, osp.basename(img_path)))
    
    # copy rgb image
    raw_rgb_img_path = osp.join(input_dir, 'render/0/cubic.jpg')
    Image.open(raw_rgb_img_path).save(osp.join(output_dir, 'rgb.png'))
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    # check depth
    rgb_img_filepath = osp.join(output_dir, 'albedo.png')
    depth_img_filepath = osp.join(output_dir, 'depth.png')
    saved_color_pcl_filepath = osp.join(output_dir, 'points3d.ply')
    vis_color_pointcloud(rgb_img_filepath, depth_img_filepath, saved_color_pcl_filepath, depth_scale=4000.0, normaliz=False)
    