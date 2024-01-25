import os
import sys
import time
import numpy as np
import json
import argparse
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_scheduler, setup_logger
from get_options import parse_option
import torch.nn.functional as F
from module.depth_estimation.networks.unifuse_ori import UniFuse
from module.depth_estimation.losses import BerhuLoss
from module.depth_estimation.metrics import Evaluator
import matplotlib.pyplot as plt
def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Create Dataset and Dataloader
    if args.dataset == 'igibson':
        from igibson.igbson_detection_dataloader_nodepth import IGbsonDetectionDataset
        from igibson.model_util_igbson import IGbsonDatasetConfig

        DATASET_CONFIG = IGbsonDatasetConfig()
        TEST_DATASET = IGbsonDetectionDataset('train', num_points=args.num_point,
                                                    augment=False,
                                                    use_color=True if args.use_color else False,
                                                    use_height=True if args.use_height else False,
                                                    use_v1=(not args.use_sunrgbd_v2),
                                                    ROOT_DIR=args.igibson_root_dir,
                                                    latent_code_dim=args.emb_dim)
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    print(f"test_len: {len(TEST_DATASET)}")

    print("training data shuffle: ",True if torch.cuda.is_available() else False)

    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=args.num_workers if torch.cuda.is_available() else 0,
                                              worker_init_fn=my_worker_init_fn,
                                              pin_memory=True,
                                              drop_last=False)

    return test_loader, DATASET_CONFIG

def load_checkpoint(model, args):
    import collections
    logger.info("=> loading checkpoint '{}'".format(args.checkpoint_path))

    pretrained_dict = torch.load(args.checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    if 'model' in pretrained_dict:
        model.load_state_dict(pretrained_dict['model'], strict=True)
    else:
        model_state_dict = model.state_dict()
        update_model_state_dict = collections.OrderedDict()

        for k, v in model_state_dict.items():
            k_temp = k.replace('depth_encoder.', '')
            if k_temp in pretrained_dict:
                update_model_state_dict.update({k: pretrained_dict[k_temp]})
            else:
                update_model_state_dict.update({k: v})

        model.load_state_dict(update_model_state_dict,strict=True)

    return model

def load_pretrain_checkpoint(checkpoint_path, model):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("loading Pretrain model... {}".format(checkpoint_path))

    return model


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False, save_best = False):
    logger.info('==> Saving...')
    state = {
        'config': args,
        'save_path': '',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    if save_best:
        state['save_path'] = os.path.join(args.log_dir, f'best_valid_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'best_valid_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'best_valid_epoch_{epoch}.pth')))

    if save_cur:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    elif epoch % args.save_freq == 0:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    else:
        state['save_path'] = 'current.pth'
        torch.save(state, os.path.join(args.log_dir, 'current.pth'))
        pass


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm

def evaluate_one_epoch(test_loader, model, args):

    model.eval()  # set model to training mode
    evaluator = Evaluator(False)
    evaluator.reset_eval_metrics()

    for batch_idx, batch_data_label in enumerate(test_loader):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        with torch.no_grad():
            end_points = model(batch_data_label)

        pred_depth = end_points["pred_depth"].detach()

        gt_depth = batch_data_label["gt_depth"]
        mask = batch_data_label["depth_val_mask"]
        for i in range(gt_depth.shape[0]):
            evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])

    evaluator.print(os.path.split(args.checkpoint_path)[0])
    return


if __name__ == '__main__':
    args = parse_option()
    args.checkpoint_path = "/mnt/workspace/code/gp3d_private_HorizonNet/log/unifuse_depth/igibson_1670591406/model.pth"

    test_loader, DATASET_CONFIG = get_loader(args)

    LOG_DIR = os.path.join(args.log_dir, 'unifuse_depth', f'{args.dataset}_{int(time.time())}')
    args.log_dir = LOG_DIR
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(output=args.log_dir, name="deep3dlayout_pretrain")
    path = os.path.join(args.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    logger.info(str(vars(args)))

    if(torch.cuda.is_available()):
        model = UniFuse (18, 512, 1024, "", 10.0, fusion_type='cee', se_in_fusion=True)
        model = load_checkpoint(model, args)
        model = model.cuda()
    else:
        model = UniFuse(18, 512, 1024, "", 10.0, fusion_type='cee', se_in_fusion=True)
        model = load_checkpoint(model, args)

    evaluate_one_epoch(test_loader, model, args)

