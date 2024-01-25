import os
import math
import trimesh
import torch

from glob import glob
import numpy as np
from pytorch3d.structures import Meshes
from shapely.geometry import Polygon
from pytorch3d.renderer import TexturesVertex

from igibson.model_util_igbson import IGbsonDatasetConfig
from utils.softargmax import softargmax1d
from pytorch3d.transforms import RotateAxisAngle,Translate
from pytorch3d.io import save_obj,load_obj
import matplotlib.pyplot as plt

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
)

from utils.custom_sharder import CustomShader

Num_heading_bin = 12

def get_standard_bbox(device = 'cpu'):

    transformation = np.eye(4)
    transformation[2, 3] = -0.5
    corners = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]])
    polygon = Polygon(corners.tolist())
    mesh = trimesh.creation.extrude_polygon(polygon, height=1, transform=transformation)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces_verts_idx = torch.tensor(mesh.faces)
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_rgb)
    textured_mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_verts_idx.to(device)],
        textures=textures.to(device)
    )
    return textured_mesh

if __name__ == '__main__':
    end_points = {}
    all_filename = glob(os.path.join("./endpoint/"+"*.npy"))
    for i in range(len(all_filename)):
        key = all_filename[i].split("/")[-1][:-4]
        try:
            end_points[key] = torch.from_numpy(np.load(all_filename[i],allow_pickle=True))
        except:
            print("can not load ",key)
            pass
    image_size = 200

    DATASET_CONFIG = IGbsonDatasetConfig()

    prefix = 'last_'
    bs = end_points[f'{prefix}center'].shape[0]

    pred_center = torch.tensor(end_points[f'{prefix}center'],requires_grad=True)
    pred_size = torch.tensor(end_points[f'{prefix}pred_size'],requires_grad=True)
    pred_heading_residual =  torch.tensor(end_points[f'{prefix}heading_residuals'],requires_grad=True)
    heading_scores = torch.tensor(end_points[f'{prefix}heading_scores'],requires_grad=True)

    pred_heading_class_argmax = torch.argmax(heading_scores, -1) # B,num_proposal
    pred_heading_residual = torch.gather(pred_heading_residual, 2, pred_heading_class_argmax.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = softargmax1d(heading_scores.view(-1,Num_heading_bin)).view(bs,-1).unsqueeze(-1)

    pred_heading_gap = 2 * math.pi / float(Num_heading_bin)
    pred_angle = (pred_heading_gap * pred_heading_class + pred_heading_residual).squeeze(2)

    bbox_num = bs * 256
    batch_bbox_mesh = get_standard_bbox()
    batch_bbox_mesh = batch_bbox_mesh.extend(bbox_num)

    # apply_size
    all_size = pred_size.view(-1,3)
    batch_bbox_mesh = batch_bbox_mesh.scale_verts(all_size)

    # apply_rotation
    mesh_rotation = RotateAxisAngle(pred_angle.view(-1), degrees=False, axis="Z")
    # apply_translation
    mesh_translation = Translate(pred_center.view(-1,3))
    transform = mesh_rotation.compose(mesh_translation)
    updated_verts = transform.transform_points(batch_bbox_mesh.verts_padded())
    batch_bbox_mesh = batch_bbox_mesh.update_padded(updated_verts)

    batch_bbox_mesh_verts = batch_bbox_mesh.verts_padded()
    batch_bbox_mesh_faces = batch_bbox_mesh.faces_padded()

    # for bbox_idx in range(bbox_num):
    #     save_obj("dump/"+str(bbox_idx)+".obj",verts=batch_bbox_mesh_verts[bbox_idx],faces=batch_bbox_mesh_faces[bbox_idx])


    # render bbox
    R, T = look_at_view_transform(10, 0, 0)
    cameras = FoVOrthographicCameras(scale_xyz = ((0.12,0.12,0.12),), device='cpu', R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    shader = CustomShader(device='cpu', cameras=cameras)
    renderer = MeshRenderer(rasterizer, shader)
    renderer_bbox_image = renderer(batch_bbox_mesh)
    renderer_bbox_image = renderer_bbox_image.view(bs,-1,image_size,image_size,1)

    # print(image.shape)
    # for i in range(512):
    #     print(i)
    #     plt.imshow(image[i,:,:].detach().numpy())
    #     plt.show()

    # render layout_mesh
    mesh_filepath = "/Users/yuandong/Downloads/layout_mesh_2stage_551/Beechwood_1_int_00016_refine.obj"
    verts, faces, aux = load_obj(mesh_filepath)
    faces_verts_idx = faces.verts_idx
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_rgb.to('cpu'))

    textured_mesh = Meshes(
        verts=[torch.tensor(verts,requires_grad=True)],
        faces=[faces_verts_idx],
        textures=textures
    )
    textured_mesh = textured_mesh.extend(renderer_bbox_image.shape[0])

    renderer_layout_image = renderer(textured_mesh).unsqueeze(1)
    

    outside_mask = renderer_layout_image<0.5
    outside_bbox_image = outside_mask * renderer_bbox_image
    outside_bbox_image_1d = outside_bbox_image.view(bs,256,-1)
    inters_sum = torch.sum(outside_bbox_image_1d,2)

    obj_ass_mask = end_points[f'{prefix}objectness_label'].float()

    mask_and = torch.sum(torch.logical_and(renderer_layout_image>0.5, renderer_bbox_image>0.5).view(bs,256,-1),2)
    renderer_bbox_image_mask = torch.sum((renderer_bbox_image>0.5).view(bs,256,-1),2)
    inside_mask = (mask_and==renderer_bbox_image_mask)

    inv_mask_and = torch.sum(torch.logical_and(renderer_layout_image<0.5, renderer_bbox_image>0.5).view(bs,256,-1),2)
    outside_mask = (inv_mask_and==renderer_bbox_image_mask)

    inside_outside_mask = ~torch.logical_or(inside_mask,outside_mask)

    final_mask = inside_outside_mask * obj_ass_mask

    physical_loss = torch.sum(inters_sum * final_mask)/(torch.sum(final_mask) + 1e-6)/100






