import random
import copy
from typing import Optional
import torch
from torch import nn, Tensor
from models.CSNet.utils import gen_encoder_output_proposals


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer, num_layers, norm=None, d_model=256,
                 num_queries=300,
                 deformable_encoder=False,
                 enc_layer_share=False, enc_layer_dropout_prob=None,
                 two_stage_type='no',  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            # self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
            self.layers = nn.ModuleList(encoder_layer)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(_norm_layer) for i in range(num_layers - 1)])
                self.enc_proj = nn.ModuleList([copy.deepcopy(_proj_layer) for i in range(num_layers - 1)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


    @staticmethod
    def get_12l_with_1l_with_2l_with_3l(output, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask, return_interm_indices):
        if return_interm_indices == [2, 3]:
            start1, start2 = level_start_index[-1], level_start_index[-2]

            output_12 = output[:, start2:, :]
            pos_12 = pos[:, start2:, :]
            reference_points_12 = reference_points[:, start2:, -2:, :]
            spatial_shapes_12 = spatial_shapes[-2:, :]
            key_padding_mask_12 = key_padding_mask[:, start2:]
            level_start_index_12 = level_start_index.clone()
            level_start_index_12 = level_start_index_12[-2:] - level_start_index_12[-2]

            output_1 = output[:, start1:, :]
            pos_1 = pos[:, start1:, :]
            reference_points_1 = reference_points[:, start1:, -1:, :]
            spatial_shapes_1 = spatial_shapes[-1].view(1, -1)
            key_padding_mask_1 = key_padding_mask[:, start1:]
            level_start_index_1 = level_start_index.clone()
            level_start_index_1 = level_start_index_1[-1:] - level_start_index_1[-1]

            output_2 = output[:, start2:start1, :]
            pos_2 = pos[:, start2:start1, :]
            reference_points_2 = reference_points[:, start2:start1, -2:-1, :]
            spatial_shapes_2 = spatial_shapes[-2].view(1, -1)
            key_padding_mask_2 = key_padding_mask[:, start2:start1]
            level_start_index_2 = torch.tensor([0]).to(level_start_index.device)

            output_3 = output[:, :start2, :]
            pos_3 = pos[:, :start2, :]
            reference_points_3 = reference_points[:, :start2, -3:-2, :]
            spatial_shapes_3 = spatial_shapes[-3].view(1, -1)
            key_padding_mask_3 = key_padding_mask[:, :start2]
            level_start_index_3 = torch.tensor([0]).to(level_start_index.device)
            return output_12, pos_12, reference_points_12, spatial_shapes_12, key_padding_mask_12, level_start_index_12, \
                   output_1, pos_1, reference_points_1, spatial_shapes_1, key_padding_mask_1, level_start_index_1, \
                   output_2, pos_2, reference_points_2, spatial_shapes_2, key_padding_mask_2, level_start_index_2, \
                   output_3, pos_3, reference_points_3, spatial_shapes_3, key_padding_mask_3, level_start_index_3
        else:
            assert return_interm_indices == [2, 3], 'scale3'


    @staticmethod
    def get_12l_with_123l_with_3l_with_4l(output, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask, return_interm_indices):
        if return_interm_indices == [1, 2, 3]:
            start2, start3 = level_start_index[-2], level_start_index[-3]
            output_12 = output[:, start2:, :]
            pos_12 = pos[:, start2:, :]
            reference_points_12 = reference_points[:, start2:, -2:, :]
            spatial_shapes_12 = spatial_shapes[-2:, :]
            key_padding_mask_12 = key_padding_mask[:, start2:]
            level_start_index_12 = level_start_index.clone()
            level_start_index_12 = level_start_index_12[-2:] - level_start_index_12[-2]

            output_123 = output[:, start3:, :]
            pos_123 = pos[:, start3:, :]
            reference_points_123 = reference_points[:, start3:, -3:, :]
            spatial_shapes_123 = spatial_shapes[-3:, :]
            key_padding_mask_123 = key_padding_mask[:, start3:]
            level_start_index_123 = level_start_index.clone()
            level_start_index_123 = level_start_index_123[-3:] - level_start_index_123[-3]

            output_3 = output[:, start3:start2, :]
            pos_3 = pos[:, start3:start2, :]
            reference_points_3 = reference_points[:, start3:start2, -3:-2, :]
            spatial_shapes_3 = spatial_shapes[-3].view(1, -1)
            key_padding_mask_3 = key_padding_mask[:, start3:start2]
            level_start_index_3 = torch.tensor([0]).to(level_start_index.device)

            output_4 = output[:, :start3, :]
            pos_4 = pos[:, :start3, :]
            reference_points_4 = reference_points[:, :start3, -4:-3, :]
            spatial_shapes_4 = spatial_shapes[-4].view(1, -1)
            key_padding_mask_4 = key_padding_mask[:, :start3]
            level_start_index_4 = torch.tensor([0]).to(level_start_index.device)
            return output_12, pos_12, reference_points_12, spatial_shapes_12, key_padding_mask_12, level_start_index_12, \
                   output_123, pos_123, reference_points_123, spatial_shapes_123, key_padding_mask_123, level_start_index_123, \
                   output_3, pos_3, reference_points_3, spatial_shapes_3, key_padding_mask_3, level_start_index_3,\
                   output_4, pos_4, reference_points_4, spatial_shapes_4, key_padding_mask_4, level_start_index_4
        else:
            assert return_interm_indices == [1, 2, 3], 'scale4'

    @staticmethod
    def get_123l_with_1234l_with_4l_with_5l(output, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask, return_interm_indices):
        if return_interm_indices == [0, 1, 2, 3]:
            start3, start4 = level_start_index[-3], level_start_index[-4]
            output_123 = output[:, start3:, :]
            pos_123 = pos[:, start3:, :]
            reference_points_123 = reference_points[:, start3:, -3:, :]
            spatial_shapes_123 = spatial_shapes[-3:, :]
            key_padding_mask_123 = key_padding_mask[:, start3:]
            level_start_index_123 = level_start_index.clone()
            level_start_index_123 = level_start_index_123[-3:] - level_start_index_123[-3]

            output_1234 = output[:, start4:, :]
            pos_1234 = pos[:, start4:, :]
            reference_points_1234 = reference_points[:, start4:, -4:, :]
            spatial_shapes_1234 = spatial_shapes[-4:, :]
            key_padding_mask_1234 = key_padding_mask[:, start4:]
            level_start_index_1234 = level_start_index.clone()
            level_start_index_1234 = level_start_index_1234[-4:] - level_start_index_1234[-4]

            output_4 = output[:, start4:start3, :]
            pos_4 = pos[:, start4:start3, :]
            reference_points_4 = reference_points[:, start4:start3, -4:-3, :]
            spatial_shapes_4 = spatial_shapes[-4].view(1, -1)
            key_padding_mask_4 = key_padding_mask[:, start4:start3]
            level_start_index_4 = torch.tensor([0]).to(level_start_index.device)

            output_5 = output[:, :start4, :]
            pos_5 = pos[:, :start4, :]
            reference_points_5 = reference_points[:, :start4, -5:-4, :]
            spatial_shapes_5 = spatial_shapes[-5].view(1, -1)
            key_padding_mask_5 = key_padding_mask[:, :start4]
            level_start_index_5 = torch.tensor([0]).to(level_start_index.device)
            return output_123, pos_123, reference_points_123, spatial_shapes_123, key_padding_mask_123, level_start_index_123, \
                   output_1234, pos_1234, reference_points_1234, spatial_shapes_1234, key_padding_mask_1234, level_start_index_1234, \
                   output_4, pos_4, reference_points_4, spatial_shapes_4, key_padding_mask_4, level_start_index_4, \
                   output_5, pos_5, reference_points_5, spatial_shapes_5, key_padding_mask_5, level_start_index_5
        else:
            assert return_interm_indices == [0, 1, 2, 3], 'scale5'

    def forward(self,
                src: Tensor,
                pos: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor,
                ref_token_index: Optional[Tensor] = None,
                ref_token_coord: Optional[Tensor] = None,
                return_interm_indices=None,
                cda=None
                ):
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None
        output = src
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        length = output.shape[1]
        for layer_id, layer in enumerate(self.layers):
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                if self.deformable_encoder:
                    if cda:
                        if layer_id < 2:
                            now_length = output.shape[1]
                            if return_interm_indices == [1, 2, 3]:
                                if length == now_length:
                                    output, pos_12, reference_points_12, spatial_shapes_12, key_padding_mask_12, level_start_index_12, \
                                        output_123, pos_123, reference_points_123, spatial_shapes_123, key_padding_mask_123, level_start_index_123, \
                                        output_3, pos_3, reference_points_3, spatial_shapes_3, key_padding_mask_3, level_start_index_3, \
                                        output_4, pos_4, reference_points_4, spatial_shapes_4, key_padding_mask_4, level_start_index_4 = \
                                        self.get_12l_with_123l_with_3l_with_4l(output, pos, reference_points,
                                                                               spatial_shapes, level_start_index,
                                                                               key_padding_mask, return_interm_indices)
                                output = layer(src=output, pos=pos_12, reference_points=reference_points_12,
                                               spatial_shapes=spatial_shapes_12, level_start_index=level_start_index_12,
                                               key_padding_mask=key_padding_mask_12, scale=2)
                            elif return_interm_indices == [2, 3]:
                                if length == now_length:
                                    output_12, pos_12, reference_points_12, spatial_shapes_12, key_padding_mask_12, level_start_index_12, \
                                        output, pos_1, reference_points_1, spatial_shapes_1, key_padding_mask_1, level_start_index_1, \
                                        output_2, pos_2, reference_points_2, spatial_shapes_2, key_padding_mask_2, level_start_index_2, \
                                        output_3, pos_3, reference_points_3, spatial_shapes_3, key_padding_mask_3, level_start_index_3 = \
                                        self.get_12l_with_1l_with_2l_with_3l(output, pos, reference_points,
                                                                             spatial_shapes, level_start_index,
                                                                             key_padding_mask, return_interm_indices)
                                output = layer(src=output, pos=pos_1, reference_points=reference_points_1,
                                               spatial_shapes=spatial_shapes_1, level_start_index=level_start_index_1,
                                               key_padding_mask=key_padding_mask_1, scale=1)

                            elif return_interm_indices == [0, 1, 2, 3]:
                                if length == now_length:
                                    output, pos_123, reference_points_123, spatial_shapes_123, key_padding_mask_123, level_start_index_123, \
                                        output_1234, pos_1234, reference_points_1234, spatial_shapes_1234, key_padding_mask_1234, level_start_index_1234, \
                                        output_4, pos_4, reference_points_4, spatial_shapes_4, key_padding_mask_4, level_start_index_4, \
                                        output_5, pos_5, reference_points_5, spatial_shapes_5, key_padding_mask_5, level_start_index_5 = \
                                        self.get_123l_with_1234l_with_4l_with_5l(output, pos, reference_points,
                                                                             spatial_shapes, level_start_index,
                                                                             key_padding_mask, return_interm_indices)
                                output = layer(src=output, pos=pos_123, reference_points=reference_points_123,
                                               spatial_shapes=spatial_shapes_123, level_start_index=level_start_index_123,
                                               key_padding_mask=key_padding_mask_123, scale=3)

                        if layer_id >= 2 and layer_id < 3:
                            cr_f = True
                            if return_interm_indices == [1, 2, 3]:
                                output = layer(src=output, pos=pos_12, reference_points=reference_points_12,
                                               spatial_shapes=spatial_shapes_12, level_start_index=level_start_index_12,
                                               key_padding_mask=key_padding_mask_12, scale=2,
                                               src_cross=output_3, pos_cross=pos_3,
                                               reference_points_cross=reference_points_3,
                                               spatial_shapes_cross=spatial_shapes_3,
                                               level_start_index_cross=level_start_index_3,
                                               key_padding_mask_cross=key_padding_mask_3, f=cr_f, scale_c=1)
                            elif return_interm_indices == [2, 3]:
                                output = layer(src=output, pos=pos_1, reference_points=reference_points_1,
                                               spatial_shapes=spatial_shapes_1, level_start_index=level_start_index_1,
                                               key_padding_mask=key_padding_mask_1, scale=1,
                                               src_cross=output_2, pos_cross=pos_2,
                                               reference_points_cross=reference_points_2,
                                               spatial_shapes_cross=spatial_shapes_2,
                                               level_start_index_cross=level_start_index_2,
                                               key_padding_mask_cross=key_padding_mask_2, f=cr_f, scale_c=1)
                            elif return_interm_indices == [0, 1, 2, 3]:
                                output = layer(src=output, pos=pos_123, reference_points=reference_points_123,
                                               spatial_shapes=spatial_shapes_123, level_start_index=level_start_index_123,
                                               key_padding_mask=key_padding_mask_123, scale=3,
                                               src_cross=output_4, pos_cross=pos_4,
                                               reference_points_cross=reference_points_4,
                                               spatial_shapes_cross=spatial_shapes_4,
                                               level_start_index_cross=level_start_index_4,
                                               key_padding_mask_cross=key_padding_mask_4, f=cr_f, scale_c=1)

                        if layer_id >= 3 and layer_id < 5:
                            if return_interm_indices == [1, 2, 3]:
                                output = layer(src=output, pos=pos_123, reference_points=reference_points_123,
                                               spatial_shapes=spatial_shapes_123, level_start_index=level_start_index_123,
                                               key_padding_mask=key_padding_mask_123, scale=3)
                            elif return_interm_indices == [2, 3]:
                                output = layer(src=output, pos=pos_12, reference_points=reference_points_12,
                                               spatial_shapes=spatial_shapes_12, level_start_index=level_start_index_12,
                                               key_padding_mask=key_padding_mask_12, scale=2)
                            elif return_interm_indices == [0, 1, 2, 3]:
                                output = layer(src=output, pos=pos_1234, reference_points=reference_points_1234,
                                               spatial_shapes=spatial_shapes_1234, level_start_index=level_start_index_1234,
                                               key_padding_mask=key_padding_mask_1234, scale=4)

                        if layer_id >= 5 and layer_id < 6:
                            cr_f = True
                            if return_interm_indices == [1, 2, 3]:
                                output = layer(src=output, pos=pos_123, reference_points=reference_points_123,
                                               spatial_shapes=spatial_shapes_123,
                                               level_start_index=level_start_index_123,
                                               key_padding_mask=key_padding_mask_123, scale=3,
                                               src_cross=output_4, pos_cross=pos_4,
                                               reference_points_cross=reference_points_4,
                                               spatial_shapes_cross=spatial_shapes_4,
                                               level_start_index_cross=level_start_index_4,
                                               key_padding_mask_cross=key_padding_mask_4, f=cr_f, scale_c=1)
                            elif return_interm_indices == [2, 3]:
                                output = layer(src=output, pos=pos_12, reference_points=reference_points_12,
                                               spatial_shapes=spatial_shapes_12, level_start_index=level_start_index_12,
                                               key_padding_mask=key_padding_mask_12, scale=2,
                                               src_cross=output_3, pos_cross=pos_3,
                                               reference_points_cross=reference_points_3,
                                               spatial_shapes_cross=spatial_shapes_3,
                                               level_start_index_cross=level_start_index_3,
                                               key_padding_mask_cross=key_padding_mask_3, f=cr_f, scale_c=1)
                            elif return_interm_indices == [0, 1, 2, 3]:
                                output = layer(src=output, pos=pos_1234, reference_points=reference_points_1234,
                                               spatial_shapes=spatial_shapes_1234, level_start_index=level_start_index_1234,
                                               key_padding_mask=key_padding_mask_1234, scale=4,
                                               src_cross=output_5, pos_cross=pos_5,
                                               reference_points_cross=reference_points_5,
                                               spatial_shapes_cross=spatial_shapes_5,
                                               level_start_index_cross=level_start_index_5,
                                               key_padding_mask_cross=key_padding_mask_5, f=cr_f, scale_c=1)

                    elif not cda:
                        if return_interm_indices == [1, 2, 3]:
                            output = layer(src=output, pos=pos, reference_points=reference_points,
                                           spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                           key_padding_mask=key_padding_mask, scale=4)
                        elif return_interm_indices == [2, 3]:
                            output = layer(src=output, pos=pos, reference_points=reference_points,
                                           spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                           key_padding_mask=key_padding_mask, scale=3)
                        elif return_interm_indices == [0, 1, 2, 3]:
                            output = layer(src=output, pos=pos, reference_points=reference_points,
                                           spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                           key_padding_mask=key_padding_mask, scale=5)

                else:
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1), key_padding_mask=key_padding_mask).transpose(0, 1)

            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1']) or (self.two_stage_type == 'enceachlayer')) \
                    and (layer_id != self.num_layers - 1):
                output_memory, output_proposals = gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes)
                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))

                # gather boxes
                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]  # bs, nq
                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))

                output = output_memory

            # aux loss
            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(intermediate_output)  # n_enc/n_enc-1, bs, \sum{hw}, d_model
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref