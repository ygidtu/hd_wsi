#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import os
import time

import click
from kfb import TSlide
import skimage.io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from loguru import logger

import configs as CONFIGS
from utils.utils_image import get_pad_width, rgba2rgb
from utils.utils_wsi import ObjectIterator, folder_iterator
from utils.utils_wsi import export_detections_to_image, export_detections_to_table
from utils.utils_wsi import load_cfg, load_hdyolo_model, is_image_file


def analyze_one_patch(img, model, dataset_configs, mpp=None, compute_masks=True, ):
    model_par0 = next(model.parameters())
    img = img.to(model_par0.device, model_par0.dtype, non_blocking=True)

    # rescale
    if mpp is not None and mpp != dataset_configs['mpp']:
        scale_factor = dataset_configs['mpp'] / mpp
        img_rescale = F.interpolate(img[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
    else:
        scale_factor = 1.0
        img_rescale = img
    h_rescale, w_rescale = img_rescale.shape[1:]

    # pad to 64
    if h_rescale % 64 != 0 or w_rescale % 64 != 0:
        input_h, input_w = math.ceil(h_rescale / 64) * 64, math.ceil(w_rescale / 64) * 64
        pad_width = get_pad_width((h_rescale, w_rescale), (input_h, input_w), pos='center', stride=1)
        inputs = F.pad(img_rescale[None], [pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1]],
                       mode='constant', value=0.)
    else:
        pad_width = [(0, 0), (0, 0)]
        inputs = img_rescale[None]

    t0 = time.time()

    print(compute_masks)
    with torch.no_grad():
        outputs = model(inputs, compute_masks=compute_masks)[1]
        res = outputs[0]['det']

    # unpad and scale back to original coords
    res['boxes'] -= res['boxes'].new([pad_width[1][0], pad_width[0][0], pad_width[1][0], pad_width[0][0]])
    res['boxes'] /= scale_factor
    res['labels'] = res['labels'].to(torch.int32)
    res['boxes'] = res['boxes'].to(torch.float32)
    res = {k: v.cpu().detach() for k, v in res.items()}
    t1 = time.time()

    return {'cell_stats': res, 'inference_time': t1 - t0}


def overlay_masks_on_image(image, mask):
    msk = Image.fromarray(mask)
    blended = Image.fromarray(image)
    blended.paste(msk, mask=msk.split()[-1])

    return blended


@click.command()
@click.option('--data_path', required=True, type=str, help="Input data filename or directory.")
@click.option('--meta_info', default='meta_info.yaml', type=str,
              help="A yaml file contains: label texts and colors.")
@click.option('--model', default='lung', type=str, help="Model path, torch jit model.")
@click.option('--output_dir', default='patch_results', type=str, help="Output folder.")
@click.option('--device', default='cuda', type=str, help='Run on cpu or gpu.')
@click.option('--mpp', default=None, type=float, help='Input data mpp.')
@click.option('--box_only', is_flag=True, help="Only save box and ignore mask.")
@click.option('--export_text', is_flag=True,
              help="If save_csv is enabled, whether to convert numeric labels into text.")
def patch(data_path, meta_info, model, output_dir, device, mpp, box_only, export_text):
    u""" Patch inference with HD-Yolo. """
    if model in CONFIGS.MODEL_PATHS:
        model = CONFIGS.MODEL_PATHS[model]

    model = load_hdyolo_model(model, nms_params=CONFIGS.NMS_PARAMS)
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning(f"Cuda is not available, use cpu instead.")
        device = 'cpu'
    device = torch.device(device)

    if device.type == 'cpu':  # half precision only supported on CUDA
        model.float()
    model.eval()
    model.to(device)
    logger.info(f"Load model to {device} (nms: {model.headers.det.nms_params}")

    meta_info = load_cfg(meta_info)
    dataset_configs = {'mpp': CONFIGS.DEFAULT_MPP, **CONFIGS.DATASETS, **meta_info}
    logger.info(f"Dataset configs: {dataset_configs}")

    if os.path.isdir(data_path):
        keep_fn = lambda x: is_image_file(x)
        patch_files = list(folder_iterator(data_path, keep_fn))
    else:
        rel_path = os.path.basename(data_path)
        patch_files = [(0, rel_path, data_path)]
    logger.info(f"Inputs: {data_path} ({len(patch_files)} files observed). ")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"Outputs: {output_dir}")

    for file_idx, rel_path, patch_path in patch_files:
        logger.info(patch_path)
        output_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_id, ext = os.path.splitext(os.path.basename(patch_path))
        # run inference
        # img = read_image(patch_path).type(torch.float32) / 255

        if ext.endswith(".kfb"):
            raw_img = TSlide(patch_path).read_whole_image(2)
            if raw_img.ndim > 2:
                if raw_img.shape[-1] not in (3, 4) and raw_img.shape[-3] in (3, 4):
                    raw_img = np.swapaxes(raw_img, -1, -3)
                    raw_img = np.swapaxes(raw_img, -2, -3)
            raw_img = rgba2rgb(raw_img)
        else:
            print("?")
            raw_img = rgba2rgb(skimage.io.imread(patch_path))

        img = ToTensor()(raw_img)
        # img = img.type(torch.IntTensor)
        # print(img)
        outputs = analyze_one_patch(
            img, model, dataset_configs, mpp=mpp,
            compute_masks=not box_only,
        )
        logger.info(f"Inference time: {outputs['inference_time']} s")
        res_file = os.path.join(output_dir, f"{image_id}_pred.pt")
        torch.save(outputs, res_file)

        if 'masks' in outputs['cell_stats']:
            param_masks = outputs['cell_stats']['masks']
        else:
            param_masks = None

        # save image
        object_iterator = ObjectIterator(
            boxes=outputs['cell_stats']['boxes'],
            labels=outputs['cell_stats']['labels'],
            scores=outputs['cell_stats']['scores'],
            masks=param_masks,
        )
        mask_img = export_detections_to_image(
            object_iterator, (img.shape[1], img.shape[2]),
            labels_color=dataset_configs['labels_color'],
            save_masks=not box_only, border=3,
            alpha=1.0 if box_only else CONFIGS.MASK_ALPHA,
        )
        export_img = overlay_masks_on_image(raw_img, mask_img)
        img_file = os.path.join(output_dir, f"{image_id}_pred{ext}")
        export_img.save(img_file)

        # save to csv
        if export_text and 'labels_text' in dataset_configs:
            labels_text = dataset_configs['labels_text']
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
            save_masks=not box_only,
        )
        csv_file = os.path.join(output_dir, f"{image_id}_pred.csv")
        df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    patch()
    pass
