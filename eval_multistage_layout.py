import torch
import numpy as np
from module.horizonnet.layout_estimation import HorizonNet,HorizonNetUpSample
import os
from tqdm import tqdm
from get_options import parse_option

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

def get_model(config, DATASET_CONFIG):
    from models.detector_layout import GroupFreeDetectorHorizonNet
    num_input_channel = int(config.use_color) * 3

    model = GroupFreeDetectorHorizonNet(num_class=DATASET_CONFIG.num_class,
                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              width=config.width,
                              bn_momentum=config.bn_momentum,
                              sync_bn=True if config.syncbn else False,
                              num_proposal=config.num_target,
                              sampling=config.sampling,
                              dropout=config.transformer_dropout,
                              activation=config.transformer_activation,
                              nhead=config.nhead,
                              num_decoder_layers=config.num_decoder_layers,
                              dim_feedforward=config.dim_feedforward,
                              self_position_embedding=config.self_position_embedding,
                              cross_position_embedding=config.cross_position_embedding,
                              size_cls_agnostic=True if config.size_cls_agnostic else False,
                              emb_codes_dim=config.emb_dim,
                              image_feature_fusion = config.image_feature_fusion,
                              layout_flag=config.layout_flag)

    return model

def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Create Dataset and Dataloader
    if args.dataset == 'igibson':
        from igibson.igbson_detection_dataloader import IGbsonDetectionDataset
        from igibson.model_util_igbson import IGbsonDatasetConfig

        DATASET_CONFIG = IGbsonDatasetConfig()

        TEST_DATASET = IGbsonDetectionDataset('test', num_points=args.num_point,
                                                    augment=False,
                                                    use_color=True if args.use_color else False,
                                                    use_height=True if args.use_height else False,
                                                    use_v1=(not args.use_sunrgbd_v2),
                                                    ROOT_DIR=args.igibson_root_dir,
                                                    latent_code_dim=args.emb_dim)
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=8,
                                              worker_init_fn=my_worker_init_fn,
                                              pin_memory=True,
                                              drop_last=False)
    print(f" test_loader_len: {len(test_loader)}")

    return test_loader, DATASET_CONFIG

def load_checkpoint(args, model):
    # Load checkpoint if there is any
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model']
        len_key = len("module.")
        import collections
        new_dict = collections.OrderedDict()
        for k in list(state_dict.keys()):
            if(k[:len_key] == "module."):
                new_dict[k[len_key:]] = state_dict[k]
            else:
                new_dict[k] = state_dict[k]
        model.load_state_dict(new_dict, strict=True)
        print(f"{args.checkpoint_path} loaded successfully!!!")
    else:
        raise FileNotFoundError
    return model

if __name__ == '__main__':

    args = parse_option()
    args.checkpoint_path = ''

    test_loader, DATASET_CONFIG = get_loader(args)

    layout_initial_dir = 'layout_initial'
    os.system('rm -rf layout_initial')
    os.makedirs(layout_initial_dir,exist_ok=True)

    layout_refine_dir = 'layout_refine'
    os.system('rm -rf layout_refine')
    os.makedirs(layout_refine_dir,exist_ok=True)

    layout_ensemble_dir = 'layout_ensemble'
    os.system('rm -rf layout_ensemble')
    os.makedirs(layout_ensemble_dir,exist_ok=True)

    model = get_model(config = args, DATASET_CONFIG = DATASET_CONFIG)

    # load model
    model = load_checkpoint(args, model)
    if (torch.cuda.is_available()):
        model = model.cuda()

    model.eval()

    for batch_idx, batch_data_label in tqdm(enumerate(test_loader)):

        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        with torch.no_grad():
            end_points = model(batch_data_label)

        fname = batch_data_label['scan_name'][0]
        initial_cor_id = infer(end_points=end_points,prefixe='initial_')['cor_id']
        with open(os.path.join(layout_initial_dir, f'{fname}.txt'), 'w') as f:
            for u, v in initial_cor_id:
                f.write(f'{u:.1f} {v:.1f}\n')
        f.close()

        refine_cor_id = infer(end_points=end_points,prefixe='refine_')['cor_id']
        with open(os.path.join(layout_refine_dir, f'{fname}.txt'), 'w') as f:
            for u, v in refine_cor_id:
                f.write(f'{u:.1f} {v:.1f}\n')
        f.close()

        end_points['ensemble_' + 'bon'] = (end_points['initial_' + 'bon']+end_points['refine_' + 'bon'])/2
        end_points['ensemble_' + 'cor'] = (end_points['initial_' + 'cor']+end_points['refine_' + 'cor'])/2
        ensemble_cor_id = infer(end_points=end_points,prefixe='ensemble_')['cor_id']
        with open(os.path.join(layout_ensemble_dir, f'{fname}.txt'), 'w') as f:
            for u, v in ensemble_cor_id:
                f.write(f'{u:.1f} {v:.1f}\n')
        f.close()

    eval_command_initial = 'python eval_layout.py --dt_glob '+layout_initial_dir
    eval_command_refine = 'python eval_layout.py --dt_glob '+layout_refine_dir
    eval_command_ensemble = 'python eval_layout.py --dt_glob '+layout_ensemble_dir

    print('\n=====================>Eval Layout: {}<====================='.format('initial'))
    os.system(eval_command_initial)

    print('\n=====================>Eval Layout: {}<====================='.format('refine'))
    os.system(eval_command_refine)

    print('\n=====================>Eval Layout: {}<====================='.format('ensemble'))
    os.system(eval_command_ensemble)
