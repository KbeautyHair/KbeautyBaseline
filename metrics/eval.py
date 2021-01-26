"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

from core.data_loader import InputFetcher
from metrics.fid import calculate_fid_given_paths
from core.data_loader import get_sample_loader
from core import utils


@torch.no_grad()
def calculate_metrics(nets, args, step, mode):
    print('Calculating evaluation metrics...')
    #assert mode in ['latent', 'reference']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domains = [n for n in range(args.num_domains)]
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)


    for trg_idx, trg_domain in enumerate(domains):
        task = '%s' % trg_domain
        path_fake = os.path.join(args.eval_dir, task)
        shutil.rmtree(path_fake, ignore_errors=True)
        os.makedirs(path_fake)

        loader = get_sample_loader(root=args.val_img_dir, img_size=args.img_size,
                                       batch_size=args.val_batch_size,
                                       shuffle=False, num_workers=args.num_workers,
                                       drop_last=False, trg_domain=trg_domain, mode = mode,
                                       dataset_dir=args.dataset_dir, threshold = args.num_sample)

        fetcher = InputFetcher(loader, None, args.latent_dim, 'test')

        print('Generating images for %s...' % task)

        for i in tqdm(range(len(loader))):
            # fetch images and labels
            inputs = next(fetcher)
            x_src, x_ref, y = inputs.src, inputs.trg, inputs.y
            N = x_src.size(0)
            x_src = x_src.to(device)
            x_ref = x_ref.to(device)
            y_trg = torch.tensor([trg_idx] * N).to(device)

            masks = None

            s_trg = nets.style_encoder(x_ref, y_trg)

            x_fake = nets.generator(x_src, s_trg, masks=masks)

             # save generated images to calculate FID later
            for k in range(N):
                filename = os.path.join(
                    path_fake,
                    '%.4i.png' % (i*args.val_batch_size+(k+1)))
                utils.save_image(x_fake[k], ncol=1, filename=filename)

    # calculate and report fid values
    fid_values, fid_mean = calculate_fid_for_all_tasks(args, domains, step=step, mode=mode, dataset_dir = args.dataset_dir)
    return fid_values, fid_mean

def calculate_fid_for_all_tasks(args, domains, step, mode,dataset_dir = ''):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    for trg_domain in domains:
        task = '%s' % trg_domain
        path_real = args.val_img_dir
        print('Calculating FID for %s...' % task)
        fid_value = calculate_fid_given_paths(
            paths=[path_real, args.eval_dir],
            img_size=args.img_size,
            batch_size=args.val_batch_size,
            trg_domain = trg_domain,
            dataset_dir = dataset_dir)
        fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    utils.save_json(fid_values, filename)
    return fid_values, fid_mean