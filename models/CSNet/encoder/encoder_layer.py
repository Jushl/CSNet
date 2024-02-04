from torch import nn, Tensor
from models.CSNet.utils import gen_encoder_output_proposals, MLP,_get_activation_fn, gen_sineembed_for_position
from models.CSNet.ops.modules import MSDeformAttn, MSDeformCrossAttn
import torch


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False, use_deformable_box_attn=False, box_attn_type='roi_align',
                 layer_id=None, levels=None, CDA=True):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

        if int(levels) == 5 and CDA:
            if layer_id == 0:
                self.self_attn_3 = MSDeformAttn(d_model, 3, n_heads, n_points)
            elif layer_id == 1:
                self.self_attn_3 = MSDeformAttn(d_model, 3, n_heads, n_points)
            elif layer_id == 2:
                self.self_attn_1 = MSDeformAttn(d_model, 1, n_heads, n_points)
                self.cross_attn_3 = MSDeformCrossAttn(d_model, 3, n_heads, n_points)
            elif layer_id == 3:
                self.self_attn_4 = MSDeformAttn(d_model, 4, n_heads, n_points)
            elif layer_id == 4:
                self.self_attn_4 = MSDeformAttn(d_model, 4, n_heads, n_points)
            elif layer_id == 5:
                self.self_attn_1 = MSDeformAttn(d_model, 1, n_heads, n_points)
                self.cross_attn_4 = MSDeformCrossAttn(d_model, 4, n_heads, n_points)

        elif int(levels) == 4 and CDA:
            if layer_id == 0:
                self.self_attn_2 = MSDeformAttn(d_model, 2, n_heads, n_points)
            elif layer_id == 1:
                self.self_attn_2 = MSDeformAttn(d_model, 2, n_heads, n_points)
            elif layer_id == 2:
                self.self_attn_1 = MSDeformAttn(d_model, 1, n_heads, n_points)
                self.cross_attn_2 = MSDeformCrossAttn(d_model, 2, n_heads, n_points)
            elif layer_id == 3:
                self.self_attn_3 = MSDeformAttn(d_model, 3, n_heads, n_points)
            elif layer_id == 4:
                self.self_attn_3 = MSDeformAttn(d_model, 3, n_heads, n_points)
            elif layer_id == 5:
                self.self_attn_1 = MSDeformAttn(d_model, 1, n_heads, n_points)
                self.cross_attn_3 = MSDeformCrossAttn(d_model, 3, n_heads, n_points)

        elif int(levels) == 3 and CDA:
            if layer_id == 0:
                self.self_attn_1 = MSDeformAttn(d_model, 1, n_heads, n_points)
            elif layer_id == 1:
                self.self_attn_1 = MSDeformAttn(d_model, 1, n_heads, n_points)
            elif layer_id == 2:
                self.self_attn_1 = MSDeformAttn(d_model, 1, n_heads, n_points)
                self.cross_attn_1 = MSDeformCrossAttn(d_model, 1, n_heads, n_points)
            elif layer_id == 3:
                self.self_attn_2 = MSDeformAttn(d_model, 2, n_heads, n_points)
            elif layer_id == 4:
                self.self_attn_2 = MSDeformAttn(d_model, 2, n_heads, n_points)
            elif layer_id == 5:
                self.self_attn_1 = MSDeformAttn(d_model, 1, n_heads, n_points)
                self.cross_attn_2 = MSDeformCrossAttn(d_model, 2, n_heads, n_points)

        elif int(levels) == 5 and CDA is False:
            self.self_attn_5 = MSDeformAttn(d_model, 5, n_heads, n_points)

        elif int(levels) == 4 and CDA is False:
            self.self_attn_4 = MSDeformAttn(d_model, 4, n_heads, n_points)

        elif int(levels) == 3 and CDA is False:
            self.self_attn_3 = MSDeformAttn(d_model, 3, n_heads, n_points)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None, scale=None,
                src_cross=None, pos_cross=None, reference_points_cross=None, spatial_shapes_cross=None,
                level_start_index_cross=None, key_padding_mask_cross=None, f=False, scale_c=None):
        if not f:
            if scale == 1:
                src2 = self.self_attn_1(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
            elif scale == 2:
                src2 = self.self_attn_2(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
            elif scale == 3:
                src2 = self.self_attn_3(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
            elif scale == 4:
                src2 = self.self_attn_4(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
            elif scale == 5:
                src2 = self.self_attn_5(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)

            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src = self.forward_ffn(src)
            if self.add_channel_attention:
                src = self.norm_channel(src + self.activ_channel(src))
            return src

        if f:
            if scale_c == 1:
                src2 = self.self_attn_1(self.with_pos_embed(src_cross, pos_cross), reference_points_cross, src_cross, spatial_shapes_cross, level_start_index_cross, key_padding_mask_cross)
            src_cross = src2

            if scale == 1:
                src2 = self.cross_attn_1(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask, src_cross)
            elif scale == 2:
                src2 = self.cross_attn_2(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask, src_cross)
            elif scale == 3:
                src2 = self.cross_attn_3(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask, src_cross)
            elif scale == 4:
                src2 = self.cross_attn_4(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask, src_cross)

            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src = self.forward_ffn(src)

            src = torch.cat([src_cross, src], dim=1)
            return src

