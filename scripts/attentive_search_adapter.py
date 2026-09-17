"""Expose head depth, normalization and batch size without editing NeuroVFM."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--trainer-source', type=Path, required=True)
    parser.add_argument('--head-depth', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--head-norm', type=int, choices=(0, 1), default=0)
    parser.add_argument('--batch-size', type=int, choices=(4, 8, 16), default=4)
    opts, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location('neurovfm_attentive_source', opts.trainer_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_head = module.ClassifyThenAggregate
    original_loader = module.DataLoader

    def head(*args, **kwargs):
        kwargs['mlp_hidden_dims'] = [kwargs['hidden_dim']] * opts.head_depth
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
