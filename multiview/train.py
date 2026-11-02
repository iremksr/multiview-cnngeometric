"""Train CNNGeometric on partially observed MessyTable scenes.
Run from repository root: python -m multiview.train ...
"""
import argparse
import csv
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from multiview.dataset import MessyTableMultiViewDataset, multiview_collate_fn
from multiview.loss import TinyAlignmentLoss
from multiview.model import MultiViewCNNGeometric


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eligible_scenes(dataset):
    indices = []
    for i, folder in enumerate(tqdm(dataset.scenes, desc=f'Indexing {dataset.split}')):
        with (folder / 'annotations.json').open(encoding='utf-8') as f:
            ann = json.load(f)
        ids = [str(x) for x in ann.get('instance_ids', ann.get('common_instance_ids', []))]
        if any(iid in ann['cameras'][str(cid)].get('instances', {})
               for iid in ids for cid in ann['side_cameras']):
            indices.append(i)
    return Subset(dataset, indices)


def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    # Center/box are per-pair means, regularization is a per-supervised-view mean.
    sums = {'center_loss': 0., 'box_loss': 0., 'reg_loss': 0.,
            'before_iou': 0., 'after_iou': 0.}
    pairs_total = views_total = hits_total = 0
    for batch in tqdm(loader, desc='train' if training else 'val', leave=False):
        keys = ('views', 'center', 'center_boxes', 'view_boxes',
                'view_mask', 'instance_mask', 'pair_mask')
        x = {k: batch[k].to(device, non_blocking=True) for k in keys}
        valid = (x['view_mask'][:, :, None] & x['instance_mask'][:, None, :]
                 & x['pair_mask'])
        pairs = int(valid.sum().item())
        if pairs == 0:
            continue
        supervised_views = int((x['view_mask'] & valid.any(-1)).sum().item())
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            result = model(x['views'], x['center'], return_warped=False,
                           view_mask=x['view_mask'])
            losses = criterion(theta=result['theta'], center_boxes=x['center_boxes'],
                               view_boxes=x['view_boxes'], view_mask=x['view_mask'],
                               instance_mask=x['instance_mask'], pair_mask=x['pair_mask'])
            if not torch.isfinite(losses['loss']).item():
                raise RuntimeError('Non-finite loss: check annotations and theta')
            if training:
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=1., error_if_nonfinite=True)
                optimizer.step()
        before, after, hits, count = criterion.alignment_metrics(
            result['theta'].detach(), x['center_boxes'], x['view_boxes'],
            x['view_mask'], x['instance_mask'], x['pair_mask'])
        sums['center_loss'] += losses['center_loss'].item() * pairs
        sums['box_loss'] += losses['box_loss'].item() * pairs
        sums['reg_loss'] += losses['reg_loss'].item() * supervised_views
        sums['before_iou'] += before
        sums['after_iou'] += after
        pairs_total += count
        views_total += supervised_views
        hits_total += hits
    if not pairs_total:
        raise RuntimeError('No valid center-side-instance pairs in epoch')
    metrics = {'center_loss': sums['center_loss'] / pairs_total,
               'box_loss': sums['box_loss'] / pairs_total,
               'reg_loss': sums['reg_loss'] / max(views_total, 1),
               'before_iou': sums['before_iou'] / pairs_total,
               'after_iou': sums['after_iou'] / pairs_total,
               'iou50': hits_total / pairs_total,
               'pairs': pairs_total}
    metrics['loss'] = (criterion.center_weight * metrics['center_loss']
                       + criterion.box_weight * metrics['box_loss']
                       + criterion.reg_weight * metrics['reg_loss'])
    return metrics


def atomic_save(path, state):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def load_finetuned(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    weights = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(weights, dict):
        raise ValueError(f'No model weights found in {path}')
    model.load_state_dict(weights, strict=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--pretrained', required=True, help='Original CNNGeometric StreetView .pth.tar')
    p.add_argument('--init-weights', default=None, help='Previous fine-tuned checkpoint')
    p.add_argument('--resume', default=None, help='Resume experiment including optimizer')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-train-scenes', type=int, default=0,
                   help='Limit scenes for a smoke test; 0 means all')
    p.add_argument('--max-val-scenes', type=int, default=0,
                   help='Limit scenes for a smoke test; 0 means all')
    args = p.parse_args()
    if args.resume and args.init_weights:
        p.error('--resume and --init-weights cannot be combined')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train = eligible_scenes(MessyTableMultiViewDataset(args.data_root, 'train'))
    val = eligible_scenes(MessyTableMultiViewDataset(args.data_root, 'val'))
    if args.max_train_scenes > 0:
        train = Subset(train, list(range(min(args.max_train_scenes, len(train)))))
    if args.max_val_scenes > 0:
        val = Subset(val, list(range(min(args.max_val_scenes, len(val)))))
    if not len(train) or not len(val):
        raise RuntimeError('No eligible train or validation scenes')
    print(f'Device: {device}; train scenes: {len(train)}; val scenes: {len(val)}', flush=True)
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0),
                          num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                          collate_fn=multiview_collate_fn)
               for i, ds in enumerate((train, val))]
    model = MultiViewCNNGeometric(checkpoint_path=args.pretrained,
                                  freeze_backbone=True).to(device)
    if args.init_weights:
        load_finetuned(model, args.init_weights)
        print('Initialized from:', args.init_weights, flush=True)
    criterion = TinyAlignmentLoss(image_size=512).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    start, best, bad_epochs = 1, float('inf'), 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        start, best = saved['epoch'] + 1, saved['best_val']
        bad_epochs = saved.get('patience_counter', 0)
        print('Resuming at epoch', start, flush=True)
    history = out / 'history.csv'
    columns = ['epoch', 'train_loss', 'val_loss', 'train_center', 'val_center',
               'train_box', 'val_box', 'train_reg', 'val_reg',
               'train_before_iou', 'val_before_iou', 'train_after_iou',
               'val_after_iou', 'train_iou50', 'val_iou50', 'train_pairs', 'val_pairs']
    if not args.resume or not history.exists():
        with history.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(columns)
    else:
        with history.open(newline='', encoding='utf-8') as f:
            header = next(csv.reader(f), [])
        if header != columns:
            raise RuntimeError('Existing history.csv columns differ; use a new output directory')
    for epoch in range(start, args.epochs + 1):
        train_m = epoch_pass(model, loaders[0], criterion, device, optimizer)
        val_m = epoch_pass(model, loaders[1], criterion, device)
        improved = val_m['loss'] < best
        if improved:
            best, bad_epochs = val_m['loss'], 0
        else:
            bad_epochs += 1
        with history.open('a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, train_m['loss'], val_m['loss'],
                train_m['center_loss'], val_m['center_loss'],
                train_m['box_loss'], val_m['box_loss'],
                train_m['reg_loss'], val_m['reg_loss'],
                train_m['before_iou'], val_m['before_iou'],
                train_m['after_iou'], val_m['after_iou'],
                train_m['iou50'], val_m['iou50'],
                train_m['pairs'], val_m['pairs']])
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'best_val': best,
                 'patience_counter': bad_epochs}
        atomic_save(out / 'last.pth', state)
        if improved:
            atomic_save(out / 'best.pth', state)
        gpu_mem = (f' | max GPU alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB'
                   if device.type == 'cuda' else '')
        print(f'Epoch {epoch}/{args.epochs} | train={train_m["loss"]:.6f} '
              f'val={val_m["loss"]:.6f} best={best:.6f} '
              f'val IoU {val_m["before_iou"]:.3f}->{val_m["after_iou"]:.3f} '
              f'IoU@.5={val_m["iou50"]:.3f} bad={bad_epochs}{gpu_mem}', flush=True)
        if bad_epochs >= args.patience:
            print('Early stopping', flush=True)
            break
    print('Done:', out, flush=True)


if __name__ == '__main__':
    main()
