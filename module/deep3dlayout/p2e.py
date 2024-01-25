import numpy as np
import torch
import trimesh

def get_unit_map(height = 512):
    h = height
    w = height * 2
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    Phi = -np.repeat(Phi, h, axis=0)

    X = np.expand_dims(np.sin(Theta) * np.sin(Phi), 2)
    Y = np.expand_dims(np.cos(Theta), 2)
    Z = np.expand_dims(np.sin(Theta) * np.cos(Phi), 2)
    unit_map = np.concatenate([X, Z, Y], axis=2)

    return unit_map

def get_upper_plane(image_size,fov):

    half_image_size = image_size/2
    sita = (180 - fov)/2
    z = np.tan(sita / 180 * np.pi) * half_image_size
    point1 = [-half_image_size, -half_image_size, z]
    point2 = [half_image_size, -half_image_size, z]
    point3 = [half_image_size, half_image_size, z]
    point4 = [-half_image_size, half_image_size, z]
    verts = np.array([point1,point2,point3,point4])
    faces = np.array([[1,2,3],[1,3,4]])-1
    mesh = trimesh.Trimesh(vertices=verts.tolist(), faces=faces.tolist())

    return mesh


def get_lower_plane(image_size, fov):
    half_image_size = image_size / 2
    sita = (180 - fov)/2
    z = - np.tan( sita / 180 * np.pi) * half_image_size
    point1 = [-half_image_size, -half_image_size, z]
    point2 = [half_image_size, -half_image_size, z]
    point3 = [half_image_size, half_image_size, z]
    point4 = [-half_image_size, half_image_size, z]
    verts = np.array([point1, point2, point3, point4])
    faces = np.array([[1, 2, 3], [1, 3, 4]]) - 1
    mesh = trimesh.Trimesh(vertices=verts.tolist(), faces=faces.tolist())

    return mesh

def get_p2e_grid_sample(image_size, fov, height, npy_file = None):
    if npy_file!=None:
        return torch.from_numpy(np.load(npy_file))

    unit_map = get_unit_map(height=height)
    ray_directions = unit_map.reshape(-1,3)
    ray_origins = np.zeros((ray_directions.shape[0], 3))

    plane_trimesh_up = get_upper_plane(image_size,fov)
    plane_trimesh_down = get_lower_plane(image_size,fov)

    # Get the intersections
    locations, index_ray, index_tri = plane_trimesh_up.ray.intersects_location(
    ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)

    sample_grid =  np.zeros((height,height*2,2),dtype=np.float32)
    sample_grid = sample_grid.reshape(-1,2)

    for i in range(locations.shape[0]):
        new_location = np.array([-locations[i,0]/float(image_size)-0.5,-locations[i,1]/float(image_size/2)])
        sample_grid[index_ray[i],:] = new_location

    # Get the intersections
    locations, index_ray, index_tri = plane_trimesh_down.ray.intersects_location(
    ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)

    for i in range(locations.shape[0]):
        new_location = np.array([-locations[i,0]/float(image_size)+0.5,-locations[i,1]/float(image_size/2)])
        sample_grid[index_ray[i],:] = new_location

    sample_grid = sample_grid.reshape(height,height*2,2)

    sample_grid_tensor = torch.from_numpy(sample_grid)

    return sample_grid_tensor