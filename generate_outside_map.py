import os
import json
import trimesh
import torch
import cv2

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from shapely.geometry import Polygon

from igibson.dataset_utils import np_coor2xy, np_coory2v

def read_pkl_list(list_file):
    pkl_list = []
    with open(list_file) as f:
        lines = json.load(f)
        for line in lines:
            pkl_name = line.split("/")
            pkl_list.append(pkl_name[0] + "_" + pkl_name[1])
    return pkl_list

def get_gt_mesh_from_cor(cor):
    W, H = 1024, 512
    N = len(cor) // 2
    floor_z = -1.6
    floor_xy = np_coor2xy(cor[1::2], floor_z, W, H, floorW=1, floorH=1)
    c = np.sqrt((floor_xy ** 2).sum(1))
    v = np_coory2v(cor[0::2, 1], H)
    ceil_z = (c * np.tan(v)).mean()

    polygon = Polygon(floor_xy.tolist())
    transformation = np.eye(4)
    transformation[0, 0] = -1
    transformation[1, 1] = -1
    transformation[2, 3] = -1.6

    mesh = trimesh.creation.extrude_polygon(polygon, height=ceil_z - floor_z, transform=transformation)

    gt_verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    gt_faces = torch.tensor(mesh.faces)

    return gt_verts, gt_faces, mesh

def get_unit_map():
    h = 512
    w = 1024
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    Phi = -np.repeat(Phi, h, axis=0)

    X = np.expand_dims(np.sin(Theta) * np.sin(Phi),2)
    Y =  np.expand_dims(np.cos(Theta),2)
    Z = np.expand_dims(np.sin(Theta) * np.cos(Phi),2)
    unit_map = np.concatenate([X,Z,Y],axis=2)

    return unit_map

if __name__ == '__main__':
    ROOT_DIR = "/mnt/workspace/code/PanoHolisticUnderstanding/igibson_vote_data_242"
    depth_path = os.path.join(ROOT_DIR, 'igibson_depth')
    point_label_path = os.path.join(ROOT_DIR,"igbson_3d_labelmap_seg")
    scan_names = read_pkl_list(os.path.join(ROOT_DIR, 'train.json'))
    cor_occ_dir = os.path.join(ROOT_DIR, 'label_cor_occ')
    image_path = os.path.join(ROOT_DIR,"igbison_image")
    output_dir = "outside_label_map"
    os.makedirs(output_dir,exist_ok=True)

    unit_map = get_unit_map()

    for idx,scan_name in enumerate(scan_names):
        print("processing {}, {}".format(idx, scan_name))
        depth = np.load(os.path.join(depth_path, scan_name) + '_depth_gt.npy')  # Nx6
        label_map = np.load(os.path.join(point_label_path, scan_name) + '_label_map.npz')['label_map']
        with open(os.path.join(cor_occ_dir, scan_name + ".txt")) as f:
            cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

        instance_map = label_map[:,:,1]
        max_instance = np.max(instance_map[:])
        point_cloud_map = np.repeat(np.expand_dims(depth, axis=2), 3, axis=2) * unit_map
        outside_map = np.zeros(depth.shape,dtype=np.float32)
        rgb = cv2.imread(os.path.join(image_path, scan_name) + '.png')
        for instance_idx in range(int(max_instance)):
            instance_mask = (instance_map==instance_idx)
            instance_pc = point_cloud_map.copy()[instance_mask]
            instance_center = np.mean(instance_pc,0)
            dinst2ori = np.linalg.norm(instance_center)

            # generate gt_mesh
            gt_verts, gt_faces, gt_mesh = get_gt_mesh_from_cor(cor)

            ray_origins = np.zeros((1, 3))
            ray_directions = instance_center.reshape(1,3)

            # Get the intersections
            locations, index_ray, index_tri = gt_mesh.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True)
            distance2interct = np.zeros((locations.shape[0]))
            for intersect_idx in range(locations.shape[0]):
                distance2interct[intersect_idx] = np.linalg.norm(locations[intersect_idx,:])
            distance2interct = np.max(distance2interct)

            if(distance2interct>(dinst2ori-0.2)): continue
            outside_map[instance_mask] = 1.0
            rgb[:,:,2][instance_mask] = 255
        np.save(os.path.join(output_dir,scan_name+"_outside_map.npy"),outside_map)
        # cv2.imwrite(os.path.join(output_dir,scan_name+"_outside_map.jpg"),rgb)





