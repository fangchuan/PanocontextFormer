import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from models.multi_head_attention import MultiheadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 obj_posembed=None, layout_posembed=None, pc_posembed=None):
        super().__init__()
        self.self_attn_obj = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_layout = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn_obj_pc = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_obj_layout = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_layout_pc = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_layout_obj = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)
        self.norm8 = nn.LayerNorm(d_model)
        self.norm9 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)
        self.dropout7 = nn.Dropout(dropout)
        self.dropout8 = nn.Dropout(dropout)
        self.dropout9 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.obj_posembed = obj_posembed
        self.layout_posembed = layout_posembed
        self.pc_posembed = pc_posembed

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, obj_query, pc_key, layout_query, obj_query_pos, pc_pos, layout_pose):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]

        :return:
        """
        # NxCxP to PxNxC
        if self.obj_posembed is not None:
            obj_pos_embed = self.obj_posembed(obj_query_pos).permute(2, 0, 1)
        else:
            obj_pos_embed = None
        if self.layout_posembed is not None:
            layout_pos_embed = self.layout_posembed(layout_pose).permute(2, 0, 1)
        else:
            layout_pos_embed = None
        if self.pc_posembed is not None:
            pc_pos_embed = self.pc_posembed(pc_pos).permute(2, 0, 1)
        else:
            pc_pos_embed = None

        obj_query = obj_query.permute(2, 0, 1)
        pc_key = pc_key.permute(2, 0, 1)
        layout_query = layout_query.permute(2, 0, 1)

        # obj self-attention
        q = k = v = self.with_pos_embed(obj_query, obj_pos_embed)
        query2 = self.self_attn_obj(q, k, value=v)[0]
        obj_query = obj_query + self.dropout1(query2)
        obj_query = self.norm1(obj_query)

        # layout self-attention
        q = k = v = self.with_pos_embed(layout_query, layout_pos_embed)
        query2 = self.self_attn_layout(q, k, value=v)[0]
        layout_query = layout_query + self.dropout2(query2)
        layout_query = self.norm2(layout_query)

        # obj-pc cross-attention
        query2, attention_weight = self.multihead_attn_obj_pc(query=self.with_pos_embed(obj_query, obj_pos_embed),
                                     key=self.with_pos_embed(pc_key, pc_pos_embed),
                                     value=self.with_pos_embed(pc_key, pc_pos_embed))
        obj_query = obj_query + self.dropout3(query2)
        obj_query = self.norm3(obj_query)

        # layout-pc cross-attention
        query2, attention_weight = self.multihead_attn_layout_pc(query=self.with_pos_embed(layout_query, layout_pos_embed),
                                     key=self.with_pos_embed(pc_key, pc_pos_embed),
                                     value=self.with_pos_embed(pc_key, pc_pos_embed))
        layout_query = layout_query + self.dropout4(query2)
        layout_query = self.norm4(layout_query)

        # obj-layout cross-attention
        query2, attention_weight = self.multihead_attn_obj_layout(query=self.with_pos_embed(obj_query, obj_pos_embed),
                                     key=self.with_pos_embed(layout_query, layout_pos_embed),
                                     value=self.with_pos_embed(layout_query, layout_pos_embed))
        obj_query = obj_query + self.dropout8(query2)
        obj_query = self.norm8(obj_query)

        # layout-obj cross-attention
        query2, attention_weight = self.multihead_attn_layout_obj(query=self.with_pos_embed(layout_query, layout_pos_embed),
                                     key=self.with_pos_embed(obj_query, obj_pos_embed),
                                     value=self.with_pos_embed(obj_query, obj_pos_embed))
        layout_query = layout_query + self.dropout9(query2)
        layout_query = self.norm9(layout_query)

        # FFN obj
        query2 = self.linear2(self.dropout(self.activation(self.linear1(obj_query))))
        obj_query = obj_query + self.dropout5(query2)
        obj_query = self.norm5(obj_query)
        # NxCxP to PxNxC
        obj_query = obj_query.permute(1, 2, 0)

        # FFN layout
        query2 = self.linear4(self.dropout6(self.activation(self.linear3(layout_query))))
        layout_query = layout_query + self.dropout7(query2)
        layout_query = self.norm6(layout_query)
        # NxCxP to PxNxC
        layout_query = layout_query.permute(1, 2, 0)

        return obj_query, layout_query


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
