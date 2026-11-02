"""Partial-observation MessyTable multiview dataset."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)


def load_image(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.
    return (tensor - MEAN) / STD


class MessyTableMultiViewDataset(Dataset):
    def __init__(self, root='data_messytable_large', split='train',
                 min_tiny=2., max_tiny=32.):
        self.root = Path(root)
        self.split = split
        self.min_tiny = min_tiny
        self.max_tiny = max_tiny
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f'Missing dataset split: {split_dir}')
        self.scenes = sorted(p for p in split_dir.iterdir()
                             if p.is_dir() and (p / 'annotations.json').is_file())

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        folder = self.scenes[index]
        with (folder / 'annotations.json').open(encoding='utf-8') as f:
            ann = json.load(f)
        center_id = str(ann['center_camera'])
        side_ids = [str(x) for x in ann['side_cameras']]
        cams = ann['cameras']
        center = load_image(folder / cams[center_id]['filename'])
        if side_ids:
            views = torch.stack([load_image(folder / cams[cid]['filename'])
                                 for cid in side_ids])
        else:
            views = center.new_empty((0, *center.shape))
        if views.shape[1:] != center.shape:
            raise ValueError(f'Image shape mismatch in scene {folder.name}')
        declared_size = ann.get('image_size')
        if declared_size is not None and tuple(declared_size) != (center.shape[2], center.shape[1]):
            raise ValueError(f'Incorrect image_size in scene {folder.name}')

        center_instances = cams[center_id]['instances']
        raw_ids = ann.get('instance_ids', ann.get('common_instance_ids', list(center_instances)))
        ids = [str(i) for i in raw_ids if str(i) in center_instances]
        boxes = torch.tensor([center_instances[i]['bbox_xyxy'] for i in ids],
                             dtype=torch.float32).reshape(-1, 4)
        if ids:
            wh = boxes[:, 2:] - boxes[:, :2]
            keep = (torch.isfinite(boxes).all(-1) &
                    (wh.min(-1).values >= self.min_tiny) &
                    (wh.max(-1).values <= self.max_tiny))
            ids = [i for i, yes in zip(ids, keep.tolist()) if yes]
            boxes = boxes[keep]
        n, v = len(ids), len(side_ids)
        view_boxes = torch.zeros((v, n, 4), dtype=torch.float32)
        pair_mask = torch.zeros((v, n), dtype=torch.bool)
        for vi, cid in enumerate(side_ids):
            instances = cams[cid].get('instances', {})
            for ni, iid in enumerate(ids):
                obj = instances.get(iid)
                if obj is None:
                    continue
                box = torch.as_tensor(obj['bbox_xyxy'], dtype=torch.float32)
                if torch.isfinite(box).all() and (box[2:] > box[:2]).all():
                    view_boxes[vi, ni] = box
                    pair_mask[vi, ni] = True
        return {'center': center, 'views': views, 'center_boxes': boxes,
                'view_boxes': view_boxes, 'pair_mask': pair_mask,
                'instance_ids': ids, 'view_ids': side_ids,
                'scene_id': ann['scene_id'], 'split': self.split}


def multiview_collate_fn(samples):
    if not samples:
        raise ValueError('Empty batch')
    b = len(samples)
    max_v = max(s['views'].shape[0] for s in samples)
    max_n = max(s['center_boxes'].shape[0] for s in samples)
    c, h, w = samples[0]['center'].shape
    if any(s['center'].shape != (c, h, w) for s in samples):
        raise ValueError('Images in batch must have the same resolution')
    result = {
        'center': torch.stack([s['center'] for s in samples]),
        'views': torch.zeros((b, max_v, c, h, w), dtype=samples[0]['center'].dtype),
        'center_boxes': torch.zeros((b, max_n, 4), dtype=torch.float32),
        'view_boxes': torch.zeros((b, max_v, max_n, 4), dtype=torch.float32),
        'view_mask': torch.zeros((b, max_v), dtype=torch.bool),
        'instance_mask': torch.zeros((b, max_n), dtype=torch.bool),
        'pair_mask': torch.zeros((b, max_v, max_n), dtype=torch.bool),
        'instance_ids': [s['instance_ids'] for s in samples],
        'view_ids': [s['view_ids'] for s in samples],
        'scene_ids': [s['scene_id'] for s in samples],
    }
    for i, s in enumerate(samples):
        v, n = s['views'].shape[0], s['center_boxes'].shape[0]
        result['views'][i, :v] = s['views']
        result['center_boxes'][i, :n] = s['center_boxes']
        result['view_boxes'][i, :v, :n] = s['view_boxes']
        result['view_mask'][i, :v] = True
        result['instance_mask'][i, :n] = True
        result['pair_mask'][i, :v, :n] = s['pair_mask']
    return result
