"""Train CNNGeometric with partial instance supervision.
Run from repository root: python -m multiview.train ...
"""
import argparse
import csv
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
    # Read annotations only; avoid decoding tens of thousands of images.
    import json
    indices = []
    for i, folder in enumerate(tqdm(dataset.scenes, desc=f'Indexing {dataset.split}')):
        with (folder / 'annotations.json').open() as f:
            a = json.load(f)
        ids = a.get('instance_ids', a.get('common_instance_ids', []))
        if any(iid in a['cameras'][cid]['instances']
               for iid in ids for cid in a['side_cameras']):
            indices.append(i)
    return Subset(dataset, indices)


def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {k: 0. for k in ('loss', 'center_loss', 'box_loss', 'reg_loss')}
    total_pairs = 0
    for batch in tqdm(loader, desc='train' if training else 'val', leave=False):
        keys = ('views', 'center', 'center_boxes', 'view_boxes', 'view_mask',
                'instance_mask', 'pair_mask')
        x = {k: batch[k].to(device, non_blocking=True) for k in keys}
        pairs = int(x['pair_mask'].sum().item())
        if pairs == 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            result = model(x['views'], x['center'], return_warped=False)
            losses = criterion(theta=result['theta'], center_boxes=x['center_boxes'],
                               view_boxes=x['view_boxes'], view_mask=x['view_mask'],
                               instance_mask=x['instance_mask'], pair_mask=x['pair_mask'])
            if not torch.isfinite(losses['loss']).item():
                raise RuntimeError('Non-finite loss; inspect input annotations and theta')
            if training:
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=1., error_if_nonfinite=True)
                optimizer.step()
        for k in sums:
            sums[k] += losses[k].item() * pairs
        total_pairs += pairs
    if total_pairs == 0:
        raise RuntimeError('No valid center-side-instance pairs in epoch')
    return {k: v / total_pairs for k, v in sums.items()}


def atomic_save(path, state):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--pretrained', required=True, help='Original StreetView .pth.tar')
    p.add_argument('--init-weights', default=None, help='Previous fine-tuned best.pth (model weights only)')
    p.add_argument('--resume', default=None, help='Resume this experiment with optimizer state')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.resume and args.init_weights:
        p.error('--resume and --init-weights cannot be combined')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train = eligible_scenes(MessyTableMultiViewDataset(args.data_root, 'train'))
    val = eligible_scenes(MessyTableMultiViewDataset(args.data_root, 'val'))
    if not train or not val:
        raise RuntimeError('No train or val scenes with valid annotated pairs')
    print(f'Device: {device}; train scenes: {len(train)}; val scenes: {len(val)}', flush=True)
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0),
                          num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                          collate_fn=multiview_collate_fn)
               for i, ds in enumerate((train, val))]
    model = MultiViewCNNGeometric(checkpoint_path=args.pretrained,
                                  freeze_backbone=True).to(device)
    if args.init_weights:
        previous = torch.load(args.init_weights, map_location='cpu', weights_only=False)
        model.load_state_dict(previous['model_state_dict'])
        print('Initialized from fine-tuned weights:', args.init_weights, flush=True)
    criterion = TinyAlignmentLoss(image_size=512).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    start, best, bad_epochs = 1, float('inf'), 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        start, best = saved['epoch'] + 1, saved['best_val']
        bad_epochs = saved.get('patience_counter', 0)
        print('Resuming at epoch', start, flush=True)
    history = out / 'history.csv'
    if not args.resume or not history.exists():
        with history.open('w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'train_center',
                                    'val_center', 'train_box', 'val_box', 'train_reg', 'val_reg'])
    for epoch in range(start, args.epochs + 1):
        train_m = epoch_pass(model, loaders[0], criterion, device, optimizer)
        val_m = epoch_pass(model, loaders[1], criterion, device)
        improved = val_m['loss'] < best
        if improved:
            best, bad_epochs = val_m['loss'], 0
        else:
            bad_epochs += 1
        with history.open('a', newline='') as f:
            csv.writer(f).writerow([epoch, train_m['loss'], val_m['loss'],
                                    train_m['center_loss'], val_m['center_loss'],
                                    train_m['box_loss'], val_m['box_loss'],
                                    train_m['reg_loss'], val_m['reg_loss']])
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'best_val': best,
                 'patience_counter': bad_epochs}
        atomic_save(out / 'last.pth', state)
        if improved:
            atomic_save(out / 'best.pth', state)
        print(f'Epoch {epoch}/{args.epochs} | train={train_m["loss"]:.6f} '
              f'val={val_m["loss"]:.6f} best={best:.6f} bad={bad_epochs}', flush=True)
        if bad_epochs >= args.patience:
            print('Early stopping')
            break
    print('Done:', out)


if __name__ == '__main__':
    main()
