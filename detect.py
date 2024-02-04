import PIL.Image
import cv2
import warnings
import argparse
import random
import os
import sys
import numpy as np
from util.logger import setup_logger
import util.misc as utils
import torch, json
from util.slconfig import SLConfig
from util import box_ops
from config import get_args_parser
from models.CSNet.dino import build_dino
warnings.filterwarnings("ignore", category=UserWarning)

CLASSES = ['car', 'pedestrian']


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 2
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def detect(args, npy, size=None, thershold=None):
    utils.init_distributed_mode(args)
    cfg = SLConfig.fromfile(args.config_file)  # CSNet_B.py
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:  # 存入所有设置
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + ' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)
    model, criterion, postprocessors = build_dino(args)
    model.to(device)
    ema_m = None

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    logger.info("params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m

    model_without_ddp.to(device)
    model_without_ddp.eval()
    criterion.eval()

    if args.dataset_file == 'gen4':
        from datasets.gen4.gen4_transforms import ToTensor, Normalize, RandomResize
        if args.backbone == 'resnet50':
            totensor = ToTensor(1)
        else:
            totensor = ToTensor(2)

    elif args.dataset_file == 'gen1':
        from datasets.gen1.gen1_transforms import ToTensor, Normalize, RandomResize
        if args.backbone == 'resnet50':
            totensor = ToTensor(1)
        else:
            totensor = ToTensor(2)

    elif args.dataset_file == 'gen3':
        from datasets.gen3.gen3_transforms import ToTensor, Normalize, RandomResize
        if args.backbone == 'resnet50':
            totensor = ToTensor(1)
        else:
            totensor = ToTensor(2)

    elif args.dataset_file == 'DSEC':
        from datasets.DSEC.DSEC_transforms import ToTensor, Normalize, RandomResize
        if args.backbone == 'resnet50':
            totensor = ToTensor(1)
        else:
            totensor = ToTensor(2)

    elif args.dataset_file == 'coco':
        from datasets.COCO017.coco2017_transforms import RandomResize
        import datasets.COCO017.coco2017_transforms as T
        randomresize = RandomResize([max([768])], max_size=768)
        normalize = T.Compose([T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

    # GEN1
    totensor_img = ToTensor(1)
    resize = RandomResize([size], max_size=size)
    norm = Normalize([0.5055], [0.1215])
    raw_event, _ = totensor_img(npy, None)
    image = raw_event.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0).numpy().astype(np.uint8)
    img = image.copy()
    event, _ = totensor(npy, None)
    event, _ = resize(event, None)
    event, _ = norm(event, None)
    eh, ew = event.shape[-2], event.shape[-1]
    event = event.view(1, -1, eh, ew)

    # COCO
    # event = PIL.Image.open(npy)
    # image = np.array(event).copy()
    # img = image[:, :, ::-1].copy()
    # event, _ = randomresize(event, None)
    # event, _ = normalize(event, None)
    # eh, ew = event.shape[1], event.shape[2]
    # event = event.view(1, -1, eh, ew)

    output = model_without_ddp(event.to(device))
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    thershold = thershold
    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold

    tgt = {
        'boxes': boxes[select_mask],
        'size': torch.Tensor([event.shape[-2], event.shape[-1]]),
        'orig_size': [image.shape[0], image.shape[1]],
        'box_label': labels[select_mask],
        'conf': scores[select_mask]
    }

    H, W = tgt['size'].tolist()
    Ho, Wo = tgt['orig_size']
    for i in range(len(tgt['box_label'])):
        unnormbbox = tgt['boxes'][i].cpu() * torch.Tensor([Wo, Ho, Wo, Ho])
        unnormbbox[:2] -= unnormbbox[2:] / 2
        unnormbbox[2:] += unnormbbox[:2]
        xyxy = unnormbbox.numpy()

        label_num = tgt['box_label'][i]
        conf = tgt['conf'][i]
        if args.dataset_file == 'gen1':
            if int(label_num) == 0:
                label, color = 'car', (255, 0, 0)
            elif int(label_num) == 1:
                label, color = 'pedestrian', (0, 0, 255)

        lab_and_conf = f'{label:} {conf:.2f}'
        plot_one_box(xyxy, img, label=lab_and_conf, color=color, line_thickness=1)

    # cv2.imshow('detect', img)
    cv2.imwrite(os.path.join(f'weights/detection/', npy.split('/' or '\\')[-1].replace('.npy', '.png')), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    options = {'dn_scalar': 100, 'embed_init_tgt': True, 'dn_label_coef': 1.0, 'dn_bbox_coef': 1.0, 'use_ema': False, 'dn_box_noise_scale': 1.0}
    args.options = options

    args.dataset_file = 'gen1'  # [gen1, gen3, gen4, DSEC]
    npy = './weights/events/18-03-29_12-30-10_500000_60500000_49499999.npy'
    detect(args, npy, size=788, thershold=0.35)


