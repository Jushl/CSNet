import torch.utils.data
from .gen1 import build as build_gen1
from .gen4 import build as build_gen4
from .COCO017 import build as build_coco

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'gen1':
        return build_gen1(image_set, args)
    elif args.dataset_file == 'gen4':
        return build_gen4(image_set, args)
    elif args.dataset_file == 'coco':
        return build_coco(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
