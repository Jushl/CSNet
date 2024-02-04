import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
from util.misc import NestedTensor, clean_state_dict
from models.CSNet.backbone.position_encoding import build_position_encoding
from models.CSNet.backbone.Convnext.convnext import build_convnext
from models.CSNet.backbone.Swin.swin_transformer import build_swin_transformer
from models.CSNet.backbone.ResNet.resnet import resnet50
from models.CSNet.backbone.SNN.snn import resnet_snn
import matplotlib.pyplot as plt


class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict( state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels, return_interm_indices: list, name: str, dataset: str, timestep=None):
        super().__init__()

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        self.body = backbone
        self.num_channels = num_channels
        self.name = name
        self.dataset = dataset
        self.timestep = timestep

    def forward(self, tensor_list: NestedTensor):
        if self.name == 'RESNET-SNN':
            if self.dataset == 'gen1':
                B_size, N_steps, H, W = tensor_list.tensors.shape[0], tensor_list.tensors.shape[1], tensor_list.tensors.shape[2], tensor_list.tensors.shape[3]
                tensor_list.tensors = tensor_list.tensors.view(B_size, -1, N_steps, H, W).permute(0, 1, 3, 4, 2)
            elif self.dataset == 'coco':
                B_size, Channels, H, W = tensor_list.tensors.shape[0], tensor_list.tensors.shape[1], tensor_list.tensors.shape[2], tensor_list.tensors.shape[3]
                tensor_list.tensors = tensor_list.tensors.view(B_size, -1, Channels, H, W).permute(0, 2, 3, 4, 1).repeat(1, 1, 1, 1, self.timestep)

        xs, fg_pos = self.body(tensor_list.tensors)

        processed = []
        for name, feature_map in xs.items():
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        fig = plt.figure(figsize=(72, 91))
        for i in range(len(processed)):
            a = fig.add_subplot(1, 5, i + 1)
            img_plot = plt.imshow(processed[i])
            a.axis("off")
        plt.savefig('resnet18_feature_maps.jpg', bbox_inches='tight')

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out, fg_pos


class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_indices: list, steps: int, Layers: list, dataset: str):
        if name == 'RESNET':
            backbone = resnet50(return_interm_indices=return_interm_indices, pretrained=train_backbone)
        elif name == 'RESNET-SNN':
            backbone = resnet_snn(return_interm_indices=return_interm_indices, layers=Layers, steps=steps, dataset=dataset)
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))

        assert return_interm_indices in [[2, 3], [1, 2, 3], [0, 1, 2, 3]]

        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices):]
        super().__init__(backbone=backbone, train_backbone=train_backbone, num_channels=num_channels,
                         return_interm_indices=return_interm_indices, name=name, dataset=dataset, timestep=steps)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):

        xs, fg_pos = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos, fg_pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[2, 3], [1, 2, 3], [0, 1, 2, 3]]

    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, 'use_checkpoint', False)

    if args.backbone in ['RESNET', 'RESNET-SNN']:
        backbone = Backbone(name=args.backbone, train_backbone=train_backbone, return_interm_indices=return_interm_indices, steps=args.STEPS, Layers=args.LAYERS, dataset=args.dataset_file)
        bb_num_channels = backbone.num_channels
    elif args.backbone in ['swin_T_224_1k', 'swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']:
        pretrain_img_size = int(args.backbone.split('_')[-2])
        backbone = build_swin_transformer(args.backbone, pretrain_img_size=pretrain_img_size, \
                    out_indices=tuple(return_interm_indices), dilation=args.dilation, use_checkpoint=use_checkpoint)

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in backbone.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break
        if "backbone_dir" in args:
            pretrained_dir = args.backbone_dir
            PTDICT = {
                'swin_T_224_1k': 'swin_tiny_patch4_window7_224.pth',
                'swin_B_384_22k': 'swin_base_patch4_window12_384.pth',
                'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
            }
            pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
            checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
            from collections import OrderedDict
            def key_select_function(keyname):
                if 'head' in keyname:
                    return False
                if args.dilation and 'layers.3' in keyname:
                    return False
                return True
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices):]
    elif args.backbone in ['convnext_xlarge_22k']:
        backbone = build_convnext(modelname=args.backbone, pretrained=True, out_indices=tuple(return_interm_indices),backbone_dir=args.backbone_dir)
        bb_num_channels = backbone.dims[4 - len(return_interm_indices):]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels 
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    return model
