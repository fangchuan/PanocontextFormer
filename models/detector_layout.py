import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from .backbone_module import Pointnet2Backbone
from .transformer import TransformerDecoderLayer
from .modules import PointsObjClsModule, FPSModule, GeneralSamplingModule, PositionEmbeddingLearned, PredictHead, \
    ClsAgnosticPredictHead


class GroupFreeDetectorHorizonNet(nn.Module):
    r"""
        A Group-Free detector for 3D object detection via Transformer.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        width: (default: 1)
            PointNet backbone width ratio
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        sampling: (default: kps)
            Initial object candidate sampling method
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, width=1, bn_momentum=0.1, sync_bn=False, num_proposal=128, sampling='kps',
                 dropout=0.1, activation="relu", nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='xyz_learned', cross_position_embedding='xyz_learned',
                 size_cls_agnostic=False, emb_codes_dim = 0, image_feature_fusion = False, layout_flag = False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.bn_momentum = bn_momentum
        self.sync_bn = sync_bn
        self.width = width
        self.nhead = nhead
        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        self.size_cls_agnostic = size_cls_agnostic
        self.emb_codes_dim = emb_codes_dim

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim, width=self.width, image_feature_fusion=image_feature_fusion)

        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        if self.size_cls_agnostic:
            self.proposal_head = ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288, emb_codes_dim=self.emb_codes_dim)
        else:
            self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                             mean_size_arr, num_proposal, 288, emb_codes_dim=self.emb_codes_dim)
        if self.num_decoder_layers <= 0:
            # stop building if has no decoder layer
            return

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, 288, kernel_size=1)

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 288))
        elif self.self_position_embedding == 'loc_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(6, 288))
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            if self.size_cls_agnostic:
                self.prediction_heads.append(ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288, emb_codes_dim=self.emb_codes_dim))
            else:
                self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                         mean_size_arr, num_proposal, 288, emb_codes_dim=self.emb_codes_dim))

        # Init
        self.init_weights()
        self.init_bn_momentum()
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

        # layout-estimation-model
        self.layout_flag = layout_flag
        if(self.layout_flag):
            from module.horizonnet.layout_estimation import HorizonNet
            from module.layout.private_loss import layout_3d_loss
            from module.layout.model.horizon_upsample.upsample1d import Upsample1DHorizonNet
            if (torch.cuda.is_available()):
                self.layout_estimation_net = HorizonNet(cfg=None, pretrain_ckpt='/mnt/workspace/code/DeepPanoContext/pretratin_weight/HorizonNet/resnet50_rnn__st3d.pth')
            else:
                self.layout_estimation_net = HorizonNet(cfg=None, pretrain_ckpt='/Users/yuandong/Documents/Git_project_DAMO/DeepPanoContext_PAI/pretratin_weight/HorizonNet/resnet50_rnn__st3d.pth')

            self.horizon_feature_upsample = Upsample1DHorizonNet(1024, 288)

            # for layout-loss
            self.loss_3d = layout_3d_loss.layout_3dloss()

            self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                     mean_size_arr, num_proposal, 288,
                                                     emb_codes_dim=0))

            self.decoder.append(TransformerDecoderLayer(
                        288, nhead, dim_feedforward, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(6, 288),
                        cross_posembed=PositionEmbeddingLearned(6, 288),
                    ))
            self.obj_featrue_proj = nn.Conv1d(288, 288, kernel_size=1)

            # for layout
            # self.layout_feature_proj = nn.Conv1d(288, 288, kernel_size=1)
            self.decoder.append(TransformerDecoderLayer(
                        288, nhead, dim_feedforward, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(6, 288),
                        cross_posembed=PositionEmbeddingLearned(6, 288),
                    ))
            bias_flag = False
            if(bias_flag):
                self.joint_pred_bon = nn.Conv1d(288, 2, 1, padding=0, bias=True)
                self.joint_pred_cor = nn.Conv1d(288, 1, 1, padding=0, bias=True)
                nn.init.constant_(self.joint_pred_bon.bias[0], -0.478)
                nn.init.constant_(self.joint_pred_bon.bias[1], 0.425)
                nn.init.constant_(self.joint_pred_cor.bias, -1.)
            else:
                self.joint_pred_bon = nn.Conv1d(288, 2, 1, padding=0, bias=False)
                self.joint_pred_cor = nn.Conv1d(288, 1, 1, padding=0, bias=False)


    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}

        end_points = self.backbone_net(inputs['point_clouds'], end_points, image = inputs['image'], sample_pts = inputs['sample_pts'])

        # Query Points Generation
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            raise NotImplementedError

        # Proposal
        # cluster_feature.shape [16, 288, 256]
        # cluster_xyz.shape [16, 256, 3]
        proposal_center, proposal_size = self.proposal_head(cluster_feature,
                                                            base_xyz=cluster_xyz,
                                                            end_points=end_points,
                                                            prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            # query.shape =  torch.Size([16, 288, 256])
            # key.shape =  torch.Size([16, 288, 1024])
            # points_features.shape =  torch.Size([16, 288, 1024])
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

            # Transformer Decoder Layer
            # query_pos.shape =  torch.Size([16, 256, 6])
            # key_pos.shape =  torch.Size([16, 1024, 3])
            # query.shape =  torch.Size([16, 288, 256])
            # key.shape =  torch.Size([16, 288, 1024])
            query = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        if self.layout_flag:
            horizon_layout = self.layout_estimation_net(inputs['image'])
            for key in horizon_layout:
                end_points[key] = horizon_layout[key]
            layout_featrue_proj = horizon_layout['layout_feature']

            self.loss_3d.setGrid(inputs['unit_xyz'][0, ...][None, None, ...])

            pred_lonlat_up = torch.cat([inputs['unit_lonlat'][:, :, 0:1], end_points['initial_bon'][:, 0, :, None]], dim=-1)
            pred_lonlat_down = torch.cat([inputs['unit_lonlat'][:, :, 0:1], end_points['initial_bon'][:, 1, :, None]], dim=-1)
            gt_lonlat_up = torch.cat([inputs['unit_lonlat'][:, :, 0:1], inputs['bon'][:, 0, :, None]], dim=-1)

            layout_up_pose, GT_up_pose = self.loss_3d.lonlat2xyz_up(pred_lonlat_up, gt_lonlat_up, inputs['ratio'])
            layout_down_pose, _ = self.loss_3d.lonlat2xyz_down(pred_lonlat_down)
            layout_up_pose = layout_up_pose.detach().clone()
            layout_down_pose = layout_down_pose.detach().clone()

            layout_up_pose_trans = torch.cat([-layout_up_pose[:,:,0].unsqueeze(2),layout_up_pose[:,:,2].unsqueeze(2),-layout_up_pose[:,:,1].unsqueeze(2)],dim=2)
            layout_down_pose_trans = torch.cat([-layout_down_pose[:, :, 0].unsqueeze(2), layout_down_pose[:, :, 2].unsqueeze(2), -layout_down_pose[:, :, 1].unsqueeze(2)], dim=2)
            layout_pos = torch.cat([layout_up_pose_trans,layout_down_pose_trans], -1)

            object_pos = torch.cat([base_xyz, base_size], -1)

            query = self.obj_featrue_proj(query)

            # layout_featrue_proj = self.layout_feature_proj(layout_featrue_proj)
            layout_featrue_proj = self.horizon_feature_upsample(layout_featrue_proj)
            
            #cross attention
            object_feature_update = self.decoder[-1](query, layout_featrue_proj, object_pos, layout_pos)

            # Prediction
            base_xyz, base_size = self.prediction_heads[-1](object_feature_update,
                                                      base_xyz=cluster_xyz,
                                                      end_points=end_points,
                                                      prefix='layout_refine_')

            layout_feature_update = self.decoder[-2](layout_featrue_proj, query , layout_pos, object_pos)

            end_points['refine_bon'] = self.joint_pred_bon(layout_feature_update)
            end_points['refine_cor'] = self.joint_pred_cor(layout_feature_update)

        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
