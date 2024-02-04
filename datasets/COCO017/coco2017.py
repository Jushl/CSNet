from pathlib import Path
from datasets.coco2017 import Coco2017Detection
import datasets.COCO017.coco2017_transforms as T


def make_coco2017_transforms(event_setting, backbone, timestep):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    max_size = 788
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if event_setting == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])
    if event_setting == 'val' or 'test':
        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])
    raise ValueError(f'unknown {event_setting}')


def build(event_setting, args):
    root = Path(args.gen1_path)
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        "train": (root / f'{event_setting}2017', root / f'annotations' / f'instances_{event_setting}2017.json'),
        "val": (root / f'{event_setting}2017', root / f'annotations' / f'instances_{event_setting}2017.json'),
        "test": (root / f'{event_setting}2017', root / f'annotations' / f'instances_{event_setting}2017.json'),
    }
    img_folder, ann_file = PATHS[event_setting]
    dataset = Coco2017Detection(img_folder, ann_file,
                            transforms=make_coco2017_transforms(event_setting, args.backbone, args.STEPS),
                            return_masks=args.masks,)
    return dataset