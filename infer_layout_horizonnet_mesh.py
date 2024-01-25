import torch
import numpy as np
from module.horizonnet.layout_estimation import HorizonNet,HorizonNetUpSample
import os
from tqdm import tqdm
from pytorch3d.structures import Meshes
import trimesh
from igibson.dataset_utils import np_coor2xy, np_coory2v
from shapely.geometry import Polygon
from utils.metrics import compare_meshes
import time
from utils.logger import setup_logger
from pytorch3d.io import save_obj

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def infer(end_points = None, prefixe = ''):
    from module.layout.misc.post_proc import np_refine_by_fix_z, gen_ww, infer_coory
    from scipy.ndimage.filters import maximum_filter
    from shapely.geometry import Polygon

    pred_bon = end_points[prefixe+'bon'].clone()
    pred_cor = end_points[prefixe+'cor'].clone()
    H, W = 512, 1024

    y_bon_ = (pred_bon[0].cpu().numpy() / np.pi + 0.5) * H - 0.5
    y_cor_ = pred_cor[0, 0].sigmoid().cpu().numpy()
    # Init floor/ceil plane
    z0 = 50
    _, z1 = np_refine_by_fix_z(*y_bon_, z0)

    # Detech wall-wall peaks
    def find_N_peaks(signal, r, min_v, N):
        max_v = maximum_filter(signal, size=r, mode='wrap')
        pk_loc = np.where(max_v == signal)[0]
        pk_loc = pk_loc[signal[pk_loc] > min_v]
        if N is not None:
            order = np.argsort(-signal[pk_loc])
            pk_loc = pk_loc[order[:N]]
            pk_loc = pk_loc[np.argsort(pk_loc)]
        return pk_loc, signal[pk_loc]

    post_force_cuboid = False
    min_v = 0 if post_force_cuboid else 0.05
    r = int(round(W * 0.05 / 2))
    N = 4 if post_force_cuboid else None
    xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    cor, xy_cor = gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6),
                                   force_cuboid=post_force_cuboid)
    if not post_force_cuboid:
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
        if not Polygon(xy2d).is_valid:
            import sys
            print(
                'Fail to generate valid general layout!! '
                'Generate cuboid as fallback.',
                file=sys.stderr)
            xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
            cor, xy_cor = gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor) * 2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j * 2] = cor[j, 0], cor[j, 1]
        cor_id[j * 2 + 1] = cor[j, 0], cor[j, 2]
    return {'cor_id': cor_id, 'y_bon_': y_bon_, 'y_cor_': y_cor_}

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

    return gt_verts, gt_faces

if __name__ == '__main__':
    # dataloder
    if (torch.cuda.is_available()):
        igibson_root_dir = '/mnt/workspace/code/PanoHolisticUnderstanding/igibson_vote_data_242'
    else:
        igibson_root_dir = '/Users/yuandong/Documents/Git_project_DAMO/gp3d_private/igibson/example_data'
    layout_pretrain_ckpt = '/mnt/workspace/code/DeepPanoContext/out/layout_estimation/21022217101943/model_best.pth'
    # layout_pretrain_ckpt = 'log/layout_estimation/igibson_1667840572/57109150/ckpt_epoch_100.pth'

    if(not os.path.exists(layout_pretrain_ckpt)):
        print('can not find ',layout_pretrain_ckpt)
        exit()
    else:
        print('loading: ', layout_pretrain_ckpt)

    output_dir = 'layout_test_output'
    os.system('rm -rf layout_test_output')
    os.makedirs(output_dir,exist_ok=True)

    from module.deep3dlayout.dataset_layoutmesh import PanoLayoutMeshDataset
    if(torch.cuda.is_available()):
        TEST_DATASET = PanoLayoutMeshDataset(root_dir='/mnt/workspace/code/PanoHolisticUnderstanding/igibson_vote_data_242', split = 'test')
    else:
        TEST_DATASET = PanoLayoutMeshDataset(root_dir='/Users/yuandong/Documents/Git_project_DAMO/gp3d_private/igibson/example_data', split = 'test')


    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=8 if torch.cuda.is_available() else 0,
                                               worker_init_fn=my_worker_init_fn,
                                               pin_memory=True,
                                               drop_last=False)

    print('TEST_DATASET length: ',len(test_loader))

    Network = HorizonNet(cfg=None, pretrain_ckpt=layout_pretrain_ckpt)
    if (torch.cuda.is_available()):
        Network = Network.cuda()
    Network.eval()

    chamfer = []
    F1_score = []
    F3_score = []
    F5_score = []

    log_dir = os.path.split(layout_pretrain_ckpt)[0]
    method_name = "horizonenet_"

    LOG_DIR = os.path.join(log_dir, method_name+"_dump", f'{time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(log_dir, method_name+"_dump", f'{time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())}')
    log_dir = LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = setup_logger(output=LOG_DIR, name=method_name)

    for batch_idx, batch_data_label in tqdm(enumerate(test_loader)):

        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == "scan_name"): continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        with torch.no_grad():
            end_points = Network(batch_data_label['image'])

        cor_id = infer(end_points=end_points, prefixe='initial_')['cor_id']
        pred_verts, pred_faces = get_gt_mesh_from_cor(cor_id)

        if(torch.cuda.is_available()):
            pred_verts = pred_verts.cuda()
            pred_faces = pred_faces.cuda()

        pred_meshes = Meshes(verts=[pred_verts], faces=[pred_faces])
        gt_meshes = Meshes(verts=batch_data_label['gt_mesh_vertics'], faces=batch_data_label['gt_mesh_faces'])

        cur_metrics = compare_meshes(pred_meshes, gt_meshes, scale=1.0, reduce=False)
        logger.info("Chamfer-L2: {}".format(cur_metrics["Chamfer-L2"][0].item()))
        chamfer.append(cur_metrics["Chamfer-L2"][0].item())
        F1_score.append(cur_metrics["F1@0.100000"][0].item())
        F3_score.append(cur_metrics["F1@0.300000"][0].item())
        F5_score.append(cur_metrics["F1@0.500000"][0].item())

        output_filepath = os.path.join(LOG_DIR, str(batch_idx))
        save_obj(output_filepath+"_gt.obj", batch_data_label['gt_mesh_vertics'].squeeze(), batch_data_label['gt_mesh_faces'].squeeze())
        save_obj(output_filepath+"_pred.obj", pred_meshes.cpu().verts_packed(), pred_meshes.cpu().faces_packed())


    logger.info("************* Average CD-Value: {}".format(np.mean(np.array(chamfer))))
    logger.info("************* Average F1-Score: {}".format(np.mean(np.array(F1_score))))
    logger.info("************* Average F3-Score: {}".format(np.mean(np.array(F3_score))))
    logger.info("************* Average F5-Score: {}".format(np.mean(np.array(F5_score))))



