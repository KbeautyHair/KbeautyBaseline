"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader, get_val_loader
from core.solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)
    if args.mode == 'train':
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers,
                                             dataset_dir=args.dataset_dir
                                             ),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers,
                                             dataset_dir=args.dataset_dir
                                             ),
                        val=get_val_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            dataset_dir=args.dataset_dir
                                            )
                        )
        solver.train(loaders)
    elif args.mode == 'sample':
        solver.sample()
    elif args.mode == 'eval':
        fid_values, fid_mean = solver.evaluate()
        for key, value in fid_values.items():
            print(key, value)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=2,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=2,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=0,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/mqset',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/mqset',
                        help='Directory containing validation images')
    parser.add_argument('--test_img_dir', type=str, default='data/mqset',
                        help='Directory containing test images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples/k-hairstyle',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints/k-hairstyle',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='imagelists',
                        help='Directory of train, valid image lists (npy files)')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval/k-hairstyle',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results/k-hairstyle',
                        help='Directory for saving generated images')
    parser.add_argument('--src_dir', type=str, default='sample_images/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='sample_images/ref',
                        help='Directory containing input reference images')

    parser.add_argument('--src_domain', type=int, default=0,
                        help='Source domain (e.g., 0, 1)')
    parser.add_argument('--trg_domain', type=int, default=1,
                        help='Target domain (e.g., 0, 1)')
    parser.add_argument('--num_sample', type=int, default=300,
                        help='Number of samples to generate')

    # face alignment (not used in k-hairstyle baseline)
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=30000)

    args = parser.parse_args()
    main(args)
