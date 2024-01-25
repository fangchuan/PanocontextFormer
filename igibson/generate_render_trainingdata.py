import matplotlib.pyplot as plt
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
)
import numpy as np
import os
from utils.custom_sharder import CustomShader

def getTopViewRenderParms(device='cpu'):
    image_size = 160
    R, T = look_at_view_transform(10, 0, 0)
    cameras = FoVOrthographicCameras(scale_xyz=((0.12, 0.12, 0.12),), device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    shader = CustomShader(device=device, cameras=cameras)
    renderer = MeshRenderer(rasterizer, shader)
    return renderer

def generateMeshRenderImage(input_meshes, renderer, scan_name = ''):


    verts_rgb = torch.ones_like(input_meshes.verts_padded())
    textures = TexturesVertex(verts_rgb)
    textured_mesh = Meshes(
        verts=input_meshes.verts_padded(),
        faces=input_meshes.faces_padded(),
        textures=textures
    )

    renderer_layout_image = renderer(textured_mesh).squeeze()
    # if(scan_name == ''):
    return renderer_layout_image
    # else:
    #     output_dir = "igibson_gt_mesh_render_image"
    #     os.makedirs(output_dir,exist_ok=True)
    #     output_filepath = os.path.join(output_dir,scan_name+"_render_image.npy")
    #     np.save(output_filepath,np.array(renderer_layout_image,dtype=np.float32))
