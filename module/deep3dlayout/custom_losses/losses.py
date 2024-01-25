##################################################################################
#Implementation built upon Mesh R-CNN and PyTorch3D ##
##################################################################################

#BSD License

#For meshrcnn software

#Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#Redistribution and use in source and binary forms, with or without modification,
#are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name Facebook nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
#ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
##import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes, SubdivideMeshes
from pytorch3d.structures import Meshes

from pytorch3d.utils import ico_sphere

import numpy as np

from module.deep3dlayout.custom_losses.bidirectional_chamfer import custom_chamfer_distance
from pytorch3d.ops import cot_laplacian
import math

from itertools import islice
from pytorch3d.ops.knn import knn_gather, knn_points

#####SMOOTHNESS LOSS
def L_smooth_loss(meshes, method: str = "cot"):    
    ###
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)

    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)

    weights = 1.0 / weights.float()
        
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    
    loss = loss.norm(dim=1)
    
    C_mask = math.e ** -loss #### return curvature mask - assign max loss to vertex lying in planes and less to vertex lying on corners/edges
    
    loss = loss * weights * C_mask
         
    return loss.sum() / N

#####SHARPNESS LOSS
def L_sharp_loss(meshes_pred, meshes_gt, w_x = 1.0, w_y = 0.0):
    ###
    if isinstance(meshes_gt, dict):
        edge_points_gt = meshes_gt['edge_points_gt']
    else:
        edge_points_gt = sample_points_from_edges(meshes_gt, no_sampling = False)

    sharp_loss_x,sharp_loss_y,_ = custom_chamfer_distance(meshes_pred.verts_padded(), edge_points_gt)

    sharp_loss = w_x*sharp_loss_x+w_y*sharp_loss_y

    return sharp_loss

def sample_points_from_edges(meshes, num_samples: int = 5000, th=0.5, no_sampling = False):    
   ##########################    
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()

    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")
     
    num_meshes = len(meshes)
  
    # Intialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
       
   
    edge_samples = []

    for i in range(num_meshes):
        edge_vertices, sharp_points = get_sharpen_edges(meshes.__getitem__(i), th=th, num_samples = num_samples)
        
        if(no_sampling):
            edge_samples.append(edge_vertices)
        else:
            edge_samples.append(sharp_points)
    
    if(len(edge_samples)>0):
        samples = torch.cat(edge_samples,0)
                   
    return samples

def get_sharpen_edges(meshes, th = 0.5, num_samples = 5000):
    
    N = len(meshes) ####default: 1

    edges = []
    edge_points = torch.zeros((N, num_samples, 3), device=meshes.device)

    if meshes.isempty():
        return edges, edge_points
        
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
    verts_packed_to_mesh_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    
    face_to_edge = meshes.faces_packed_to_edges_packed()  # (sum(F_n), 3)
    E = edges_packed.shape[0]  # sum(E_n)
    F = faces_packed.shape[0]  # sum(F_n)
        
    with torch.no_grad():
        edge_idx = face_to_edge.reshape(F * 3)  # (3 * F,) indexes into edges
        vert_idx = (
            faces_packed.view(1, F, 3).expand(3, F, 3).transpose(0, 1).reshape(3 * F, 3)
        )
        edge_idx, edge_sort_idx = edge_idx.sort()
        vert_idx = vert_idx[edge_sort_idx]
               
        edge_num = edge_idx.bincount(minlength=E)
       
        vert_edge_pair_idx = split_list(
            list(range(edge_idx.shape[0])), edge_num.tolist()
        )
        
        vert_edge_pair_idx = [
            [e[i], e[j]]
            for e in vert_edge_pair_idx
            for i in range(len(e) - 1)
            for j in range(1, len(e))
            if i != j
        ]
        vert_edge_pair_idx = torch.tensor(
            vert_edge_pair_idx, device=meshes.device, dtype=torch.int64
        )

    v0_idx = edges_packed[edge_idx, 0]
    v0 = verts_packed[v0_idx]
    v1_idx = edges_packed[edge_idx, 1]
    v1 = verts_packed[v1_idx]
       
    n_temp0 = (v1 - v0).cross(verts_packed[vert_idx[:, 0]] - v0, dim=1)
    n_temp1 = (v1 - v0).cross(verts_packed[vert_idx[:, 1]] - v0, dim=1)
    n_temp2 = (v1 - v0).cross(verts_packed[vert_idx[:, 2]] - v0, dim=1)
    n = n_temp0 + n_temp1 + n_temp2

    n0 = n[vert_edge_pair_idx[:, 0]]
    n1 = -n[vert_edge_pair_idx[:, 1]]
    
    e_sharp = 1 - torch.cosine_similarity(n0, n1, dim=1)
                   
    for i in range(len(e_sharp)):
        if(e_sharp[i] > th):
            edges.append(edges_packed[i])
    
    NE = len(edges)
    TS = num_samples  
    
    edge_vertices = []
    
    if(NE>0):
        ES = TS // NE

        ER = TS % NE
       
       
        t_points = []

        for i in range(NE):                                    
            v0_idx = edges[i][0]
            v0 = verts_packed[v0_idx]
            v1_idx = edges[i][1]
            v1 = verts_packed[v1_idx]

            edge_vertices.append(v0)
            edge_vertices.append(v1)
                        
            e_points = []
            
            e_points_x = torch.linspace(v0[0], v1[0], steps=ES)
            e_points.append(e_points_x)
            e_points_y = torch.linspace(v0[1], v1[1], steps=ES)
            e_points.append(e_points_y)
            e_points_z = torch.linspace(v0[2], v1[2], steps=ES)
            e_points.append(e_points_z)

            e_points = torch.stack(e_points, dim = 1)
                        
            e_points = e_points.unsqueeze(0)

            t_points.append(e_points)
                      
        
        ###add points from edge 0 to pad tensor 
        if(ER>0):
            v0_idx = edges[0][0]
            v0 = verts_packed[v0_idx]
            v1_idx = edges[0][1]
            v1 = verts_packed[v1_idx]
            ###interpolate points         
            #
                        
            e_points = []
            
            e_points_x = torch.linspace(v0[0], v1[0], steps=ER)
            e_points.append(e_points_x)
            e_points_y = torch.linspace(v0[1], v1[1], steps=ER)
            e_points.append(e_points_y)
            e_points_z = torch.linspace(v0[2], v1[2], steps=ER)
            e_points.append(e_points_z)

            e_points = torch.stack(e_points, dim = 1)
                        
            e_points = e_points.unsqueeze(0)

            t_points.append(e_points)

           
        edge_points = torch.cat(t_points, 1)
              
        edge_points = edge_points.to(meshes.device)

        ####output edge vertices
        edge_vertices = torch.stack(edge_vertices, dim = 0)
        edge_vertices = edge_vertices.unsqueeze(0)
       
        edge_vertices = edge_vertices.to(meshes.device)

    else:
        print('WARNING: no edges found for th:',th)
        edge_vertices = torch.zeros((N, len(e_sharp), 3), device=meshes.device)
        
    return edge_vertices, edge_points

def split_list(input, length_to_split):
    inputt = iter(input)
    return [list(islice(inputt, elem)) for elem in length_to_split]

    




