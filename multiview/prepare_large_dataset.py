"""Convert original MessyTable JSON to partial-visibility multiview scenes."""
import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm


def resize_meta(size, target):
    w, h = size
    scale = min(target / w, target / h)
    nw, nh = round(w * scale), round(h * scale)
    return {'scale': scale, 'offset_x': (target - nw) // 2,
            'offset_y': (target - nh) // 2, 'new_size': (nw, nh)}


def letterbox(image, meta, target):
    canvas = Image.new('RGB', (target, target), (0, 0, 0))
    canvas.paste(image.convert('RGB').resize(tuple(meta['new_size']), Image.Resampling.BILINEAR),
                 (meta['offset_x'], meta['offset_y']))
    return canvas


def convert_box(box, meta, target):
    s, dx, dy = meta['scale'], meta['offset_x'], meta['offset_y']
    x1, y1, x2, y2 = box
    return [max(0., min(float(target), x1 * s + dx)),
            max(0., min(float(target), y1 * s + dy)),
            max(0., min(float(target), x2 * s + dx)),
            max(0., min(float(target), y2 * s + dy))]


def is_tiny(box, minimum, maximum):
    w, h = box[2] - box[0], box[3] - box[1]
    return min(w, h) >= minimum and max(w, h) <= maximum


def convert_split(raw, output, split, size, minimum, maximum, max_views, min_pairs):
    with (raw / 'labels' / f'{split}.json').open() as f:
        source = json.load(f)
    scenes, intrinsics = source['scenes'], source.get('intrinsics', {})
    out_split = output / split
    out_split.mkdir(parents=True, exist_ok=True)
    stats = {'total': len(scenes), 'kept': 0, 'no_tiny': 0, 'no_pairs': 0,
             'missing_images': 0, 'pairs': 0, 'center_tiny': 0}

    for scene_id, scene in tqdm(scenes.items(), desc=split):
        cams = scene.get('cameras', {})
        if '1' not in cams:
            stats['missing_images'] += 1
            continue
        center_path = raw / 'images' / cams['1']['pathname']
        if not center_path.is_file():
            stats['missing_images'] += 1
            continue
        with Image.open(center_path) as im:
            center_size = im.size
        center_meta = resize_meta(center_size, size)
        tiny_ids = []
        for iid, obj in cams['1'].get('instances', {}).items():
            box = convert_box(obj['pos'], center_meta, size)
            if is_tiny(box, minimum, maximum):
                tiny_ids.append(str(iid))
        if not tiny_ids:
            stats['no_tiny'] += 1
            continue

        # Select views with the most annotated tiny-object matches.
        # No requirement that every object be present in every view.
        candidates = []
        for cid, cam in cams.items():
            if cid == '1':
                continue
            path = raw / 'images' / cam['pathname']
            if not path.is_file():
                continue
            matches = [iid for iid in tiny_ids if iid in cam.get('instances', {})]
            if matches:
                candidates.append((cid, len(matches)))
        candidates.sort(key=lambda x: (-x[1], int(x[0])))
        selected = [cid for cid, _ in candidates[:max_views]]
        pair_count = sum(sum(iid in cams[cid]['instances'] for iid in tiny_ids) for cid in selected)
        if pair_count < min_pairs:
            stats['no_pairs'] += 1
            continue

        # Store only center objects with at least one annotated side match.
        used_ids = [iid for iid in tiny_ids if any(iid in cams[cid]['instances'] for cid in selected)]
        scene_dir = out_split / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        annotation = {'scene_id': scene_id, 'split': split, 'center_camera': '1',
                      'side_cameras': selected, 'image_size': [size, size],
                      'bbox_format': 'xyxy', 'common_instance_ids': used_ids,
                      'instance_ids': used_ids, 'cameras': {}}
        for cid, filename in [('1', 'center.jpg')] + [(cid, f'view_{cid}.jpg') for cid in selected]:
            cam = cams[cid]
            with Image.open(raw / 'images' / cam['pathname']) as im:
                original_size = list(im.size)
                meta = resize_meta(im.size, size)
                letterbox(im, meta, size).save(scene_dir / filename, quality=95)
            instances = {}
            for iid in used_ids:
                if iid not in cam.get('instances', {}):
                    continue
                obj = cam['instances'][iid]
                box = convert_box(obj['pos'], meta, size)
                # Skip invalid/clipped side boxes; they are not valid supervision.
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                instances[iid] = {'bbox_xyxy': box, 'cls': obj.get('cls'),
                                  'subcls': obj.get('subcls')}
            annotation['cameras'][cid] = {
                'filename': filename, 'source_filename': cam['pathname'],
                'original_size': original_size,
                'intrinsics': intrinsics.get(cid), 'extrinsics': cam.get('extrinsics'),
                'resize': {k: v for k, v in meta.items() if k != 'new_size'},
                'instances': instances}
        # Recompute pairs after checking side boxes.
        actual_pairs = sum(iid in annotation['cameras'][cid]['instances']
                           for cid in selected for iid in used_ids)
        if actual_pairs < min_pairs:
            import shutil
            shutil.rmtree(scene_dir)
            stats['no_pairs'] += 1
            continue
        with (scene_dir / 'annotations.json').open('w') as f:
            json.dump(annotation, f, ensure_ascii=False)
        stats['kept'] += 1
        stats['pairs'] += actual_pairs
        stats['center_tiny'] += len(used_ids)
    print(split, stats)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--min-tiny', type=float, default=2)
    p.add_argument('--max-tiny', type=float, default=32)
    p.add_argument('--max-views', type=int, default=8)
    p.add_argument('--min-pairs', type=int, default=1)
    args = p.parse_args()
    raw, out = Path(args.raw_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ('train', 'val', 'test'):
        stats[split] = convert_split(raw, out, split, args.size, args.min_tiny,
                                     args.max_tiny, args.max_views, args.min_pairs)
    with (out / 'manifest.json').open('w') as f:
        json.dump({'settings': vars(args), 'splits': stats}, f, indent=2)


if __name__ == '__main__':
    main()
