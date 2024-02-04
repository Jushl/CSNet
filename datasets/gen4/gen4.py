from pathlib import Path
from datasets.coco import CocoDetection
import datasets.gen1.gen1_transforms as T


def make_gen1_transforms(event_setting, backbone, timestep):
    scales = [352, 360, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
    max_size = 633
    scales2_resize = [400, 450, 500]
    scales2_crop = [384, 500]

    if backbone == 'resnet50':
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
        "train": (root / 'train_events_path', root / 'train_annotations_path'),
        "val": (root / 'val_events_path', root / 'val_annotations_path'),
        "test": (root / 'test_events_path', root / 'test_annotations_path'),
    }
    img_folder, ann_file = PATHS[event_setting]
    dataset = CocoDetection(img_folder, ann_file,
                            transforms=make_gen1_transforms(event_setting, args.backbone, args.STEPS),
                            return_masks=args.masks,
                            dataset_file=args.dataset_file)
    return dataset
