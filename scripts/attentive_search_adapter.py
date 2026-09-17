"""Expose head depth, normalization and batch size without editing NeuroVFM."""
import argparse
import importlib.util
from pathlib import Path
import sys


def parse_mlp_dims(value):
    try:
        dims = [int(x) for x in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated widths, e.g. 128,64,32') from exc
    if len(dims) not in (1, 2, 3) or any(d not in (32, 64, 128, 256) for d in dims):
        raise argparse.ArgumentTypeError('Use 1–3 layers with widths 32,64,128,256')
    if len(set(dims)) != 1 and not all(a > b for a,b in zip(dims,dims[1:])):
        raise argparse.ArgumentTypeError('Widths must be equal or strictly decreasing')
    return dims


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--trainer-source', type=Path, required=True)
    parser.add_argument('--head-depth', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--head-norm', type=int, choices=(0, 1), default=0)
    parser.add_argument('--batch-size', type=int, choices=(4, 8, 16), default=4)
    parser.add_argument('--mlp-dims', type=parse_mlp_dims, default=None)
    opts, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location('neurovfm_attentive_source', opts.trainer_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_head = module.ClassifyThenAggregate
    original_loader = module.DataLoader

    def head(*args, **kwargs):
        kwargs['mlp_hidden_dims'] = opts.mlp_dims or [kwargs['hidden_dim']] * opts.head_depth
        kwargs['use_norm'] = bool(opts.head_norm)
        return original_head(*args, **kwargs)

    def loader(*args, **kwargs):
        # Change training batch only; validation/test ordering and batches stay fixed.
        if kwargs.get('shuffle') is True:
            kwargs['batch_size'] = opts.batch_size
        return original_loader(*args, **kwargs)

    module.ClassifyThenAggregate = head
    module.DataLoader = loader
    sys.argv = [str(opts.trainer_source)] + remaining
    module.main()


if __name__ == '__main__':
    main()
