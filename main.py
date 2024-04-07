#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""
包一层外包脚本
"""
import os

import click

from wsi_inference import wsi
from summarize_tme_features import summarize


os.environ["YOLOv5_VERBOSE"] = 'False'
# logger.remove()
# logger.add(sys.stderr, level="INFO")

__DIR__ = os.path.dirname(os.path.abspath(__file__))


SEED = 42
SCALE_FACTOR = 32
DEFAULT_MPP = 0.25


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--data_path', required=True, type=str, help="input data filename or directory.")
@click.option('--meta_info', default=os.path.join(__DIR__, "meta_info.yaml"), type=str,
              help="meta info yaml file: contains label texts and colors.")
@click.option('--model', default='lung', type=str, help="Model path, torch jit model.")
@click.option('--output_dir', default='slide_results', type=str, help="Output folder.")
@click.option('--device', default='cpu', type=str, help='Run on cpu or gpu.')
@click.option('--roi', default='tissue', type=str, help='ROI region.')
@click.option('--batch_size', default=64, type=int, help='Number of batch size.')
@click.option('--num_workers', default=64, type=int, help='Number of workers for data loader.')
@click.option('--max_memory', default=None, type=int,
              help='Maximum MB to store masks in memory. Default is 80%% of free memory.')
@click.option('--box_only', is_flag=True, help="Only save box and ignore mask.")
@click.option('--save_img', is_flag=True,
              help="Plot nuclei box/mask into png, don't enable this option for large image.")
@click.option('--save_csv', is_flag=True,
              help="Save nuclei information into csv, not necessary.")
@click.option('--export_text', is_flag=True,
              help="If save_csv is enabled, whether to convert numeric labels into text.")
@click.option('--export_mask', is_flag=True,
              help="If save_csv is enabled, whether to export mask polygons into csv.")
@click.option('--save_nuclei', is_flag=True, help='Store nuclei morphological features into a csv file.')
@click.option('--slides_mapping_file', default=None, type=str,
              help="A csv file explains slide_id -> patient_id.")
@click.option('--scale_factor', default=SCALE_FACTOR, type=float,
              help='Apply density analysis by shrinking whole slide to 1/scale_factor.')
@click.option('--default_mpp', default=DEFAULT_MPP, type=float,
              help='Normalize slides results under different mpp into default_mpp.')
@click.option('--n_classes', default=None, type=int, help='Number of nuclei types in analysis.')
@click.option('--n_patches', default=10, type=int, help='Number of maximum patches to analysis.')
@click.option('--patch_size', default=2048, type=int, help='Patch size for analysis.')
@click.option('--max_dist', default=100., type=float,
              help='Maximum distance between nuclei when considering connecting.')
@click.option('--nms_thresh', default=0.015, type=float, help='Maximum overlapping between patches.')
@click.option('--score_thresh', default=160, type=float, help='Minimum coverage of tumor region.')
@click.option('--seed', default=SEED, type=int, help='Random seed to use.')
def main(
        model, device, meta_info, data_path, output_dir, box_only, num_workers,
        max_memory, save_img, save_csv, export_text, export_mask, batch_size, roi, seed,
        scale_factor, default_mpp, n_classes, n_patches, patch_size, max_dist, nms_thresh, score_thresh,
        save_nuclei, slides_mapping_file
):

    u"""
    Welcome
    :return:
    """

    wsi(
        model, device, meta_info, data_path, output_dir, box_only, num_workers,
        max_memory, save_img, save_csv, export_text, export_mask, batch_size, roi
    )

    summarize(
        os.path.join(output_dir), os.path.join(output_dir, "summarized"), data_path,
        slides_mapping_file, scale_factor, default_mpp,
        n_classes, n_patches, patch_size, max_dist, nms_thresh, score_thresh,
        device, num_workers, box_only, save_nuclei, save_img, seed,
    )



if __name__ == "__main__":
    main()

