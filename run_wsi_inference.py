#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
from collections import defaultdict

import click
import psutil
import torch
from loguru import logger
from openslide import open_slide

import configs as CONFIGS
from utils.utils_image import Slide
from utils.utils_wsi import ObjectIterator, WholeSlideDataset, folder_iterator, export_detections_to_image, \
    export_detections_to_table, wsi_imwrite, get_slide_and_ann_file, generate_roi_masks, load_cfg, load_hdyolo_model, \
    yolo_inference_iterator, save_predicted_models, load_predicted_models

__DIR__ = os.path.dirname(os.path.abspath(__file__))


# TODO: using multiprocessing.Queue for producer/consumer without IO blocking.
def analyze_one_slide(model, dataset, batch_size=64, n_workers=64,
                      compute_masks=True, nms_params=None,
                      device=torch.device("cpu"), export_masks=None, max_mem=None):
    if nms_params is None:
        nms_params = {}
    _byte2mb = lambda x: x / 1e6
    max_mem = max_mem or _byte2mb(psutil.virtual_memory().free * 0.8)
    input_size = dataset.model_input_size
    N_patches = len(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=0, shuffle=False, pin_memory=True,
    )

    model.eval()
    t0 = time.time()

    generator = yolo_inference_iterator(
        model, data_loader,
        input_size=input_size,
        compute_masks=compute_masks,
        device=device,
        **nms_params,
    )

    results = defaultdict(list)
    masks, mask_mem, file_index = [], 0, 0
    for o in generator:
        # logger.info(o)
        for k, v in o.items():
            if k != 'masks':
                results[k].append(v.cpu())
            else:
                mask_tensor = v.cpu()
                masks.append(mask_tensor)
                mask_mem += _byte2mb(sys.getsizeof(mask_tensor.storage()))
                avail_mem = min(max_mem, _byte2mb(psutil.virtual_memory().free * 0.8))
                # logger.info(f"Track memory usage: {mask_mem}, {mask_mem/max_mem}")
                if export_masks and mask_mem >= avail_mem:
                    file_index += 1
                    filename = f"{export_masks}_{file_index:0{len(str(N_patches))}}"
                    torch.save(torch.cat(masks), filename)
                    while masks:
                        masks.pop()
                    mask_mem = 0

    if export_masks and masks:
        if file_index == 0:
            filename = export_masks
        else:
            file_index += 1
            filename = f"{export_masks}_{file_index:0{len(str(N_patches))}}"
        torch.save(torch.cat(masks), filename)
        while masks:
            masks.pop()

    res = {k: torch.cat(v) for k, v in results.items()}
    if masks:
        res['masks'] = torch.cat(masks)
    t1 = time.time()
    logger.info(f"{t1 - t0} s")

    return {'cell_stats': res, 'inference_time': t1 - t0}


@click.command()
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
def wsi(model, device, meta_info, data_path, output_dir, box_only, num_workers,
        max_memory, save_img, save_csv, export_text, export_mask, batch_size, roi):
    u""" WSI inference with HD-Yolo. """
    if model in CONFIGS.MODEL_PATHS:
        model = CONFIGS.MODEL_PATHS[model]

    model = load_hdyolo_model(model, nms_params=CONFIGS.NMS_PARAMS)
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning(f"Cuda is not available, use cpu instead.")
        device = 'cpu'
    device = torch.device(device)

    if device.type == 'cpu':  # half precision only supported on CUDA
        model = model.float()
    model.eval()
    model = model.to(device)
    logger.info(f"Load model to {device} (nms: {model.headers.det.nms_params}")

    meta_info = load_cfg(meta_info)
    dataset_configs = {'mpp': CONFIGS.DEFAULT_MPP, **CONFIGS.DATASETS, **meta_info}
    logger.info(f"Dataset configs: {dataset_configs}")

    if os.path.isdir(data_path):
        keep_fn = lambda x: os.path.splitext(x)[1] in ['.svs', '.tiff']
        slide_files = list(folder_iterator(data_path, keep_fn))
    else:
        rel_path = os.path.basename(data_path)
        slide_files = [(0, rel_path, data_path)]
    logger.info(f"Inputs: {data_path} ({len(slide_files)} files observed). ")
    logger.info(f"Outputs: {output_dir}")

    for file_idx, rel_path, slide_file in slide_files:
        output_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        slide_id = os.path.splitext(os.path.basename(slide_file))[0]
        res_file = os.path.join(output_dir, f"{slide_id}.pt")
        res_file_masks = os.path.join(output_dir, f"{slide_id}.masks.pt")

        if not os.path.exists(res_file):
            t0 = time.time()
            try:
                osr = Slide(*get_slide_and_ann_file(slide_file))
                roi_masks = generate_roi_masks(osr, roi)
                dataset = WholeSlideDataset(osr, masks=roi_masks, processor=None, **dataset_configs)
                osr.attach_reader(open_slide(osr.img_file))
                logger.info(dataset.info())
            except Exception as e:
                logger.error(f"Failed to create slide dataset for: {slide_file}.")
                logger.error(e)
                continue
            logger.info(f"Loading slide: {time.time() - t0} s")
            outputs = analyze_one_slide(model, dataset,
                                        compute_masks=not box_only,
                                        batch_size=batch_size,
                                        n_workers=num_workers,
                                        nms_params={},
                                        device=device,
                                        export_masks=res_file_masks,
                                        max_mem=max_memory,
                                        )
            outputs['meta_info'] = meta_info
            outputs['slide_info'] = dataset.slide.info()
            outputs['slide_size'] = dataset.slide_size
            outputs['model'] = model
            outputs['rois'] = dataset.masks

            # we save nuclei masks in a separate file to speed up features extraction without mask.
            if 'masks' in outputs['cell_stats']:
                output_masks = outputs['cell_stats']['masks']
                del outputs['cell_stats']['masks']
                torch.save(output_masks, res_file_masks)
                save_predicted_models(res_file, outputs)
                outputs['cell_stats']['masks'] = output_masks
            else:
                save_predicted_models(res_file, outputs)
            osr.detach_reader(close=True)
            logger.info(f"Total time: {time.time() - t0} s")
        else:
            outputs = {}

        if save_img or save_csv:
            if not outputs:
                outputs = load_predicted_models(res_file)
            if box_only:
                param_masks = None
            else:
                if 'masks' in outputs['cell_stats']:
                    param_masks = outputs['cell_stats']['masks']
                elif os.path.exists(res_file_masks):
                    param_masks = torch.load(res_file_masks)
                else:
                    param_masks = res_file_masks

        if save_img:
            img_file = os.path.join(output_dir, f"{slide_id}.tiff")
            if not os.path.exists(img_file):
                logger.info(f"Exporting result to image: ", end="")
                t0 = time.time()

                object_iterator = ObjectIterator(
                    boxes=outputs['cell_stats']['boxes'],
                    labels=outputs['cell_stats']['labels'],
                    scores=outputs['cell_stats']['scores'],
                    masks=param_masks,
                )
                mask = export_detections_to_image(
                    object_iterator, outputs['slide_size'],
                    labels_color=outputs['meta_info']['labels_color'],
                    save_masks=not box_only, border=3,
                    alpha=1.0 if box_only else CONFIGS.MASK_ALPHA,
                )
                # Image.fromarray(mask).save(img_file)
                wsi_imwrite(mask, img_file, outputs['slide_info'], CONFIGS.TIFF_PARAMS,
                            model=outputs['model'],
                            )
                logger.info(f"{time.time() - t0} s")

        if save_csv:
            csv_file = os.path.join(output_dir, f"{slide_id}.csv")
            if not os.path.exists(csv_file):
                logger.info(f"Exporting result to csv: ", end="")
                t0 = time.time()

                if export_text and 'labels_text' in outputs['meta_info']:
                    labels_text = outputs['meta_info']['labels_text']
                else:
                    labels_text = None
                object_iterator = ObjectIterator(
                    boxes=outputs['cell_stats']['boxes'],
                    labels=outputs['cell_stats']['labels'],
                    scores=outputs['cell_stats']['scores'],
                    masks=param_masks,
                )
                df = export_detections_to_table(
                    object_iterator,
                    labels_text=labels_text,
                    save_masks=(not box_only) and export_mask,
                )
                df.to_csv(csv_file, index=False)
                logger.info(f"{time.time() - t0} s")


if __name__ == '__main__':
    wsi()
    pass
