import random
import torch
from torch import Tensor
import numpy as np
from typing import Any
from util.box_ops import box_xyxy_to_cxcywh
import torchvision.transforms.functional as F
from util.misc import interpolate
import torchvision.transforms as T


@torch.jit.unused
def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


@torch.jit.unused
def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
    target["size"] = torch.tensor([h, w])
    fields = ["labels", "area", "iscrowd"]
    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)
        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def resize(image, target, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        h, w = image_size
        if h == size:
            return (h, w)
        oh = size
        ow = int(size * w / h)
        return [oh, ow]

    def get_size(image_size, size, max_size=None):
        return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size((image.shape[1], image.shape[2]), size, max_size)
    rescaled_image = F.resize(image, size)
    rescaled_image_size = (rescaled_image.shape[1], rescaled_image.shape[2])

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image_size, (image.shape[1], image.shape[2])))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def time_entropy_encoder(time_data):
    diff = np.diff(time_data)
    diff = np.insert(diff, 0, 0)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    diff = (diff - diff_mean) ** 2 / diff_std
    time_entropy = np.exp(-diff)
    return time_entropy


def time_weight_encoder(time_data):
    unique, counts = np.unique(time_data, return_counts=True)
    result = np.zeros_like(time_data)
    for u, c in zip(unique, counts):
        result[time_data == u] = c
    max_count = np.max(result)
    time_weight = result / max_count
    return time_weight


def time_sequence_encoder(time_data, Kr=1.05):
    time_data[time_data == 0] = 1
    time_sequence = np.power(Kr, -np.log(np.abs(time_data)))
    time_sequence = time_sequence[np.argsort(time_sequence)]
    return time_sequence


def process_data(data, n_segments):
    per_num_seg = len(data) // n_segments
    outputs = torch.zeros(n_segments, 240, 304)
    max_limit = (data[:, 0].max(), data[:, 1].max())
    assert max_limit <= (240, 304), 'Exceeds the limit!!!'
    for i in range(n_segments):
        start = i * per_num_seg
        end = (i + 1) * per_num_seg if i < n_segments - 1 else len(data)
        seg_data = data[start:end]
        for j, row in enumerate(seg_data):
            coords = [int(row[0]), int(row[1])]
            if coords != [0, 0]:
                outputs[i, coords[0], coords[1]] += row[3].astype(np.float32)
    return outputs


def stream_to_tensor(data=None, num_channel=None):
    if isinstance(data, str): data = np.load(data)
    data[:, 2] -= data[0, 2]
    data[:, 3] -= 0.5
    time_data, polarity = data[:, 2], data[:, 3]
    time_entropy = time_entropy_encoder(time_data)
    time_sequence = time_sequence_encoder(time_data, Kr=1.05)
    time_weight = time_weight_encoder(time_data)
    time_data = time_entropy + time_sequence + time_weight
    polarity *= time_data
    data[:, 3] = polarity
    data[:, [1, 0]] = data[:, [0, 1]]
    event_stream_tensor = process_data(data, num_channel)
    return event_stream_tensor


def to_tensor(pic, num_channel=1, mapping=True) -> Tensor:
    default_float_dtype = torch.get_default_dtype()
    event = stream_to_tensor(pic, num_channel).contiguous()
    event = event.view(1, -1)
    pos_norm_event = event[event > 0]
    pos_mean = pos_norm_event.mean()
    neg_norm_event = event[event < 0]
    neg_mean = neg_norm_event.mean()

    event = torch.clamp(event, min=neg_mean * 1.5, max=pos_mean * 1.5)

    pos_norm_event = event[event > 0]
    pos_mean = pos_norm_event.mean()
    pos_var = pos_norm_event.var()
    neg_norm_event = event[event < 0]
    neg_mean = neg_norm_event.mean()
    neg_var = neg_norm_event.var()

    event = torch.clamp(event, min=neg_mean - 1 * neg_var, max=pos_mean + 1 * pos_var)
    max, min = torch.max(event, dim=1)[0], torch.min(event, dim=1)[0]
    event[event > 0] /= max
    event[event < 0] /= abs(min)
    normalized_event = event.view(num_channel, 240, 304)

    if mapping:
        mapped_event = torch.zeros_like(normalized_event)
        mapped_event[normalized_event < 0] = normalized_event[normalized_event < 0] * 128 + 128
        mapped_event[normalized_event >= 0] = normalized_event[normalized_event >= 0] * 127 + 128
        normalized_event = mapped_event

    if isinstance(normalized_event, torch.ByteTensor):
        return normalized_event.to(dtype=default_float_dtype)
    else:
        return normalized_event


class ToTensor(object):
    def __init__(self, time_step):
        self.time_step = time_step

    def __call__(self, img, target):
        return to_tensor(img, self.time_step), target


def hflip(image, target):
    flipped_image = F.hflip(image)
    h, w = flipped_image.shape[1], flipped_image.shape[2]
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])  # xyxy的垂直镜像反转
        target["boxes"] = boxes
    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)
    return flipped_image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Tensor, target: dict):
        img_height, img_width = img.shape[1], img.shape[2]
        w = random.randint(self.min_size, min(img_width, self.max_size))
        h = random.randint(self.min_size, min(img_height, self.max_size))
        region = T.RandomCrop.get_params(img, (h, w))
        return crop(img, target, region)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image /= 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[1], image.shape[2]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Drop(object):
    def __init__(self, d=0.1, p=0.5):
        self.d = d
        self.p = p

    def __call__(self, image, target=None):
        data = np.load(image)
        if random.random() < self.p:
            data_num = data.shape[0]
            del_num = int(data_num * self.d)
            indices = np.random.choice(data_num, size=del_num, replace=False)
            data = np.delete(data, indices, axis=0)
        image = data
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


if __name__ == '__main__':
    import time
    data = np.load(r'D:\publicData\GEN1\Gen1_30000\train\events\17-03-30_12-53-58_1037500000_1097500000_10099999.npy')
    totensor = ToTensor(time_step=1)

    t1 = time.time()
    img = totensor(data, None)
    t2 = time.time()
    print(t2-t1)

