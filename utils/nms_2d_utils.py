import numpy as np
import cv2

from igibson.model_util_igbson import IGbsonDatasetConfig
DATASET_CONFIG = IGbsonDatasetConfig()

def heading2rotmat(heading_angle):
    heading_angle_copy = (heading_angle.clone().cpu()).detach().numpy()
    rotmat = np.zeros((3, 3))
    rotmat[2, 2] = 1
    cosval = np.cos(heading_angle_copy)
    sinval = np.sin(heading_angle_copy)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat

def get_bboxes_corners(heading_class_label,heading_residual_label,center_label,
                       size_class_label,size_residual_label):

    x_corners = [-1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2, 1 / 2, 1 / 2, -1 / 2]
    y_corners = [1 / 2, 1 / 2, -1 / 2, -1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2]
    z_corners = [1 / 2, 1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2, -1 / 2, -1 / 2]
    shape_out = np.vstack([x_corners, y_corners, z_corners]).transpose()

    all_obj_vertices = []

    rotation_matrix = heading2rotmat(
        -DATASET_CONFIG.class2angle(int(heading_class_label), heading_residual_label))
    translation = np.array(center_label.clone().cpu().detach().numpy())
    size = DATASET_CONFIG.class2size(int(size_class_label), np.array(size_residual_label.clone().cpu().detach().numpy()))
    corners_3d = (rotation_matrix @ (shape_out * size).T).T + translation

    return corners_3d

def check_plane_cross_pano(corners, plane_idx):

    for i in range(4):
        p1 = np.round(transform_xyz2pix(np.expand_dims(corners[plane_idx[i][0], :],0))).astype(np.int64).squeeze(0)
        p2 = np.round(transform_xyz2pix(np.expand_dims(corners[plane_idx[i][1], :],0))).astype(np.int64).squeeze(0)
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        _p1 = np.array(p1)
        _p2 = np.array(p2)
        dist1 = np.linalg.norm(_p1 - _p2)
        p1b = np.array([p1[0] + 1024, p1[1]])
        p2b = np.array([p2[0] - 1024, p2[1]])
        dist2 = np.linalg.norm(_p1 - p2b)

        if dist1 > dist2:
            return True

    return False

def get_plane_mask(corners):
    planes1 = np.array([[0,1],[1,2],[2,3],[3,0]])
    planes2 = np.array([[4,5],[5,6],[6,7],[7,6]])
    planes3 = np.array([[0,1],[1,5],[5,4],[4,0]])
    planes4 = np.array([[2,3],[3,7],[7,6],[6,2]])
    planes5 = np.array([[1,5],[5,6],[6,2],[2,1]])
    planes6 = np.array([[0,4],[4,7],[7,3],[3,0]])
    quality = 50
    mask = np.zeros((512,1024))

    planes = [planes1,planes2,planes3,planes4,planes5,planes6]

    for plane in planes:
        cross_flag = check_plane_cross_pano(corners, plane)
        if(cross_flag):
            pix_all_left = []
            pix_all_right = []
            for i in range(4):
                p1 = corners[plane[i][0], :]
                p2 = corners[plane[i][1], :]
                points = interpolate_line(p1, p2, quality)
                pix = np.round(transform_xyz2pix(points)).astype(np.int64)
                for pix_sub in pix:
                    if(pix_sub[0]>512):
                        pix_all_right.append(np.expand_dims(pix_sub,0))
                    else:
                        pix_all_left.append(np.expand_dims(pix_sub,0))

            pix_all_right = np.concatenate(pix_all_right, axis=0)
            cv2.fillPoly(mask, [pix_all_right], 1)
            pix_all_left = np.concatenate(pix_all_left, axis=0)
            cv2.fillPoly(mask, [pix_all_left], 1)

        else:
            pix_all = []
            for i in range(4):
                p1 = corners[plane[i][0], :]
                p2 = corners[plane[i][1], :]
                points = interpolate_line(p1, p2, quality)
                pix = np.round(transform_xyz2pix(points)).astype(np.int64)
                pix_all.append(pix)
            pix_all = np.concatenate(pix_all, axis=0)
            cv2.fillPoly(mask, [pix_all], 1)

    return mask

def binaryMaskIOU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection + 1e-5)
    return iou

def interpolate_line(p1, p2, num=30):
    t = np.expand_dims(np.linspace(0, 1, num=num, dtype=np.float32), 1)
    points = p1 * (1 - t) + t * p2
    return points

def transform_xyz2pix(point):
    point_norm = np.expand_dims(np.linalg.norm(point, axis=1), axis=1)
    bbox_points_normlized = point/point_norm
    uv = unitxyz2uv(bbox_points_normlized, equ_w=1024, equ_h=512, normlized=False)
    out = np.concatenate([1023-uv[0],uv[1]],axis=1)
    return out

def unitxyz2uv(xyz, equ_w, equ_h, normlized=False):
    x, z, y = np.split(xyz, 3, axis=-1)
    lon = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    lat = np.arctan2(y, c)

    # longitude and latitude to equirectangular coordinate
    if normlized:
        u = (lon / (2 * np.pi) + 0.5)
        v = (-lat / np.pi + 0.5)
    else:
        u = (lon / (2 * np.pi) + 0.5) * equ_w - 0.5
        v = (-lat / np.pi + 0.5) * equ_h - 0.5
    return [u, v]