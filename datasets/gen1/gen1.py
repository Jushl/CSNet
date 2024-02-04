from pathlib import Path
from datasets.coco import CocoDetection
import datasets.gen1.gen1_transforms as T


def make_gen1_transforms(event_setting, backbone, timestep):
    scales = [224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416]
    max_size = 733
    scales2_resize = [300, 350, 400]
    scales2_crop = [288, 400]

    if backbone == 'RESNET':
        ts = 1
    else:
        ts = timestep
    if event_setting == 'train':
        return T.Compose([
            T.Drop(),
            T.ToTensor(time_step=ts),
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([T.RandomResize(scales2_resize),
                           T.RandomSizeCrop(*scales2_crop),
                           T.RandomResize(scales, max_size=max_size),
                           ])),
            T.Normalize([0.5055], [0.1215])
        ])
    if event_setting == 'val' or 'test':
        return T.Compose([
            T.ToTensor(time_step=ts),
            T.RandomResize([max(scales)], max_size=max_size),
            T.Normalize([0.5055], [0.1215])
        ])
    raise ValueError(f'unknown {event_setting}')


def build(event_setting, args):
    root = Path(args.gen1_path)
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        "train": (root / f'{event_setting}' / 'events', root / f'{event_setting}' / 'annotations' /'npy_annotations.json'),
        "val": (root / f'{event_setting}' / 'events', root / f'{event_setting}' / 'annotations' / 'npy_annotations.json'),
        "test": (root / f'{event_setting}' / 'events', root / f'{event_setting}' / 'annotations' / 'npy_annotations.json'),
    }
    img_folder, ann_file = PATHS[event_setting]
    dataset = CocoDetection(img_folder, ann_file,
                            transforms=make_gen1_transforms(event_setting, args.backbone, args.STEPS),
                            return_masks=args.masks,
                            dataset_file=args.dataset_file)
    return dataset
