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
from pytorch3d.ops import GraphConv

def load_checkpoint(model, checkpoint_path):
    import collections
    print("=> loading checkpoint '{}'".format(checkpoint_path))

    pretrained_dict = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

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

class GroupFreeDetectorDeep3D(nn.Module):
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
                 size_cls_agnostic=False, emb_codes_dim = 0, image_feature_fusion = False, layout_flag = False, depth_flag = False):
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

        self.depth_flag = depth_flag
        if(self.depth_flag):
            from module.depth_estimation.networks.unifuse_ori import UniFuse
            self.depth_estimation_model = UniFuse(18, 512, 1024, "", 10.0, fusion_type='cee', se_in_fusion=True)
            self.depth_estimation_model = load_checkpoint(self.depth_estimation_model,
                                                          checkpoint_path="/mnt/workspace/code/Unifuse_self_supervised/pretrain/Matterport3D/model.pth")

        # layout-estimation-model
        self.layout_flag = layout_flag
        if(self.layout_flag):
            from module.deep3dlayout.deep3dlayout_model import Deep3DlayoutNet
            self.layout_estimation_net = Deep3DlayoutNet(backbone='resnet18', decoder_type='rcnn_p2m_mhsa_pos_dual', full_size=True)
            self.decoder_obj_featrue_proj = nn.Conv1d(288, 288, kernel_size=1)
            self.decoder_vert_featrue_proj = nn.Conv1d(288, 288, kernel_size=1)

            self.decoder.append(TransformerDecoderLayer(
                        288, nhead, dim_feedforward, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(3, 288),
                        cross_posembed=PositionEmbeddingLearned(6, 288),
                    ))

            self.decoder.append(TransformerDecoderLayer(
                        288, nhead, dim_feedforward, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(6, 288),
                        cross_posembed=PositionEmbeddingLearned(3, 288),
                    ))

            self.use_activation = False

            self.layout_estimation_gconvs = nn.ModuleList()
            hidden_dim = 288
            self.add_vert_feature = False
            if self.add_vert_feature:
                vert_feat_dim = 288
            else:
                vert_feat_dim = 0
            self.layout_estimation_vert_offset = nn.Linear(hidden_dim + 3, 3)
            for i in range(6):
                if i == 0:
                    input_dim = hidden_dim + vert_feat_dim + 3
                else:
                    input_dim = hidden_dim + 3

                gconv = GraphConv(input_dim, hidden_dim, init="normal", directed=False)
                self.layout_estimation_gconvs.append(gconv)
                
            if self.size_cls_agnostic:
                self.prediction_heads.append(ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288, emb_codes_dim=0))
            else:
                self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                     mean_size_arr, num_proposal, 288,
                                                     emb_codes_dim=0))

        # Init
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.init_weights()
        self.init_bn_momentum()

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

        if (self.depth_flag):
            end_points = self.depth_estimation_model(inputs)
            pred_depth_tensor_repeat = end_points['pred_depth'].repeat(1, 3, 1, 1)
            bs = end_points['pred_depth'].shape[0]
            pc = (pred_depth_tensor_repeat * inputs['unit_map'])[inputs['pano_mask']].view(bs, 3, -1).transpose(2, 1)
            rgb_pc = torch.concat([pc, inputs['point_cloud_rgb']], dim=2)
            end_points['point_clouds'] = rgb_pc
            end_points = self.backbone_net(rgb_pc, end_points, image = inputs['image'], sample_pts = None)
        else:
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
            query, attention_weight = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        if self.layout_flag:
            deep3dlayout = self.layout_estimation_net(inputs)
            for key in deep3dlayout:
                end_points[key] = deep3dlayout[key]

            # layout-vert
            refine_mesh = deep3dlayout['deep3d_meshes'][-1]
            vert_pos_padded = refine_mesh.verts_padded()
            vert_feature_padded = torch.transpose(deep3dlayout['refine_vert_feats'],1,2)
            vert_feature_proj = self.decoder_vert_featrue_proj(vert_feature_padded)

            # object
            object_pos = torch.cat([base_xyz, base_size], -1)
            query = self.decoder_obj_featrue_proj(query)
            
            # object cross attention
            object_feature_update, end_points['attention_weight_object'] = self.decoder[-1](query, vert_feature_proj, object_pos, vert_pos_padded)
            base_xyz, base_size = self.prediction_heads[-1](object_feature_update,
                                                      base_xyz=cluster_xyz,
                                                      end_points=end_points,
                                                      prefix='layout_refine_')

            # layout-vert cross attention
            vert_feature_update, end_points['attention_weight_layout'] = self.decoder[-2](vert_feature_proj, query, vert_pos_padded, object_pos)

            verts_padded_to_packed_idx = refine_mesh.verts_padded_to_packed_idx()
            vert_pos_packed = self._padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
            vert_feature_update = torch.transpose(vert_feature_update,1,2).contiguous()
            vert_feature_update_packed = self._padded_to_packed(vert_feature_update, verts_padded_to_packed_idx)

            # updatemesh
            first_layer_feats = [vert_feature_update_packed, vert_pos_packed]

            if(self.add_vert_feature):
                ori_vert_feature = self._padded_to_packed(deep3dlayout['refine_vert_feats'], verts_padded_to_packed_idx)
                first_layer_feats.append(ori_vert_feature)

            vert_feats = torch.cat(first_layer_feats, dim=1)

            ep = refine_mesh.edges_packed()

            # Run graph conv layers
            for gconv in self.layout_estimation_gconvs:
                vert_feats_nopos = F.relu(gconv(vert_feats, ep))
                vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed], dim=1)

            # Predict a new mesh by offsetting verts
            if (self.use_activation):  ####if normalized mesh
                vert_offsets = torch.tanh(self.layout_estimation_vert_offset(vert_feats))
            else:
                vert_offsets = self.layout_estimation_vert_offset(vert_feats)

            refine_mesh = refine_mesh.offset_verts(vert_offsets)
            end_points['deep3d_meshes'].append(refine_mesh)

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

    def _padded_to_packed(self, x, idx):
        D = x.shape[-1]
        idx = idx.view(-1, 1).expand(-1, D)

        x_packed = x.view(-1, D).gather(0, idx.to(x.device))

        return x_packed
