import argparse

def parse_option():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--width', default=1, type=int, help='backbone width')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--sampling', default='kps', type=str, help='Query points sampling method (kps, fps)')
    parser.add_argument('--emb_dim', default=512, type=int, help = 'the latent code dim')
    parser.add_argument('--image_feature_fusion', default=False, type=bool, help='whether using image_feature_fusion')
    parser.add_argument('--detection_flag', default=True, type=bool, help='whether training 3d detection')
    parser.add_argument('--layout_flag', default=True, type=bool, help='whether training layout estimation')
    parser.add_argument('--depth_flag', default=False, type=bool, help='whether training layout estimation')


    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')
    parser.add_argument('--self_position_embedding', default='loc_learned', type=str,
                        help='position_embedding in self attention (none, xyz_learned, loc_learned)')
    parser.add_argument('--cross_position_embedding', default='xyz_learned', type=str,
                        help='position embedding in cross attention (none, xyz_learned)')

    # Loss
    parser.add_argument('--query_points_generator_loss_coef', default=0.2, type=float)
    parser.add_argument('--obj_loss_coef', default=0.4, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='Loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--center_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--center_delta', default=0.1111111111111, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--size_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--size_delta', default=0.0625, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--heading_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--heading_delta', default=0.04, type=float, help='delta for smoothl1 loss in heading loss')
    parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')
    parser.add_argument('--size_cls_agnostic', default = True, type = bool, help='Use class-agnostic size prediction.')
    parser.add_argument('--emb_code_delta', default = 0.00001, type = float, help='128: 0.05; 512: 0.00002')
    parser.add_argument('--phy_viol_delta', default = -0.0005, type = float, help='128: 0.05; 512: 0.00002')
    parser.add_argument('--depth_delta', default=0.1, type = float, help='128: 0.05; 512: 0.00002')
    parser.add_argument('--mesh_delta', default=1.0, type = float, help='128: 0.05; 512: 0.00002')

    # Data
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size per GPU during training [default: 8]')
    parser.add_argument('--dataset', default='igibson', help='Dataset name. sunrgbd or scannet. [default: scannet]')
    parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 50000]')
    parser.add_argument('--data_root', default='data', help='data root path')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_color', default = True, type=bool, help='Use RGB color in input.')
    parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--igibson_root_dir', default='/mnt/workspace/code/PanoHolisticUnderstanding/igibson_vote_data_242', type=str)
    # parser.add_argument('--igibson_root_dir',default='/Users/yuandong/Documents/Git_project_DAMO/gp3d_private/igibson/example_data', type = str)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to run [default: 1]')
    parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 180]')
    parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0002,
                        help='Initial learning rate for decoder [default: 0.0004]')
    parser.add_argument('--layout_learning_rate', type=float, default=0.0001,
                        help='Initial learning rate for layout estimation')
    parser.add_argument('--depth_learning_rate', type=float, default=0.0001,
                        help='Initial learning rate for layout estimation')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate sscheduler")
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # io
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=500, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=40, help='val frequency')

    # others
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.15], nargs='+',
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    # eval
    parser.add_argument('--avg_times', default=1, type=int, help='Average times')
    parser.add_argument('--dump_dir', default='dump', help='Dump dir to save sample outputs [default: None]')
    parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
    parser.add_argument('--nms_iou', type=float, default=0.15, help='NMS IoU threshold. [default: 0.25]')
    parser.add_argument('--conf_thresh', type=float, default=0.05,
                        help='Filter out predictions with obj prob less than it. [default: 0.05]')
    parser.add_argument('--faster_eval', default=True, type=bool,
                        help='Faster evaluation by skippling empty bounding box removal.')
    parser.add_argument('--save_end_point', default=False, type=bool,
                        help='Faster evaluation by skippling empty bounding box removal.')

    # deep3dlayout
    parser.add_argument('--dp3d_position_loss_coef', default=1.0, type=float)
    parser.add_argument('--dp3d_normal_loss_coef', default=1.0, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--dp3d_sharp_loss_coef', default=0.01, type=float, help='Loss weight for box loss')
    parser.add_argument('--dp3d_sharp_loss_coef_last', default=0.01, type=float, help='Loss weight for box loss')
    parser.add_argument('--dp3d_edge_loss_coef', default=0.1, type=float, help='Loss weight for box loss')
    parser.add_argument('--dp3d_smooth_loss_coef', default=0.15, type=float, help='Loss weight for box loss')
    parser.add_argument('--dp3d_vert_loss', default=0.1, type=float, help='Loss weight for box loss')
    parser.add_argument('--fuse_type', default='biproj', type=str, help='cancat biproj inv_biproj')
    
    # contexct_model
    parser.add_argument('--context_lr_ratio', default=0.01, type=float)

    args, unparsed = parser.parse_known_args()

    return args