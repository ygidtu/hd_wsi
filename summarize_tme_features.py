#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import time
from glob import glob
import pandas as pd
from loguru import logger
from tqdm import tqdm

from utils.utils_features import *
from utils.utils_image import Slide
from utils.utils_wsi import folder_iterator, load_predicted_models, ObjectIterator


def summarize(
        model_res_path, output_dir, data_path, slides_mapping_file, scale_factor, default_mpp,
        n_classes, n_patches, patch_size, max_dist, nms_thresh, score_thresh,
        device, num_workers, box_only, save_nuclei, save_img, seed,
):
    u""" WSI feature extraction. """
    assert os.path.exists(model_res_path), f"{model_res_path} does not exists."

    pts = glob(os.path.join(model_res_path, "*.pt"))
    if os.path.isdir(model_res_path) and not model_res_path.endswith(".pt") and not pts:
        res_folder = model_res_path
        keep_fn = lambda x: not x.startswith('.') and x.endswith('.pt') and not x.endswith('.masks.pt')
        res_files = list(folder_iterator(model_res_path, keep_fn=keep_fn))
    elif os.path.isdir(model_res_path) and pts:
        res_folder = model_res_path
        res_files = [(i, os.path.basename(j), j) for i, j in enumerate(pts)]
    else:
        res_folder, res_file = os.path.split(model_res_path)
        res_files = [(0, res_file, model_res_path)]

    assert len(res_files), f"Missing result files (.pt) in {model_res_path}."

    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning(f"Cuda is not available, use cpu instead.")
        device = 'cpu'
    device = torch.device(device)

    outputs = {}
    for file_idx, rel_path, res_path in tqdm(res_files):
        output_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        slide_id = os.path.splitext(rel_path)[0]
        logger.info(slide_id)

        pkl_filename = os.path.join(output_dir, f"{slide_id}.features.pkl")
        if not os.path.exists(pkl_filename):
            res_file = os.path.join(res_folder, f"{slide_id}.pt")
            res_file_masks = os.path.join(res_folder, f"{slide_id}.masks.pt")
            if os.path.exists(res_file):  # Load wsi results
                res = load_predicted_models(res_file)
                if not len(res['cell_stats']):
                    logger.info(f"Warning: {res_file} is empty!")
                    res = None
            else:
                logger.info(f"Warning: {res_file} doesn't exists!")
                res = None

            if res is None:
                output = {}
            else:
                t0 = time.time()
                # [res['cell_stats'], res['rois'], res['slide_size'], res['slide_info'], slide['meta_info']]
                res_nuclei, slide_size = res['cell_stats'], res['slide_size']
                slide_info, meta_info = res['slide_info'], res['meta_info']
                logger.info(
                    f"Find results for: slide_id={slide_id}, magnitude={slide_info['magnitude']}, "
                    f"mpp={slide_info['mpp']}, slide_size={slide_info['level_dims'][0]}"
                )

                mpp_scale = slide_info['mpp'] / default_mpp
                slide_size = int(math.ceil(slide_size[0] * mpp_scale)), int(math.ceil(slide_size[1] * mpp_scale))

                # res_nuclei["boxes"] = filter_out_inf(res_nuclei["boxes"])
                # remove boxes with inf
                kept_idx = [not any(x == torch.inf) for x in tqdm(res_nuclei["boxes"])]
                for k, v in res_nuclei.items():
                    res_nuclei[k] = v[kept_idx]

                res_nuclei['boxes'] *= mpp_scale

                # Extract nuclei features
                object_iterator = ObjectIterator(
                    boxes=res_nuclei['boxes'],
                    labels=res_nuclei['labels'],
                    scores=res_nuclei['scores'],
                    masks=None if box_only else res_file_masks,
                )
                # keep_fn = lambda x: (x['label'] <= n_classes) and (x['label'] >= 0),
                nuclei_features = extract_nuclei_features(
                    object_iterator, num_workers=num_workers,
                )
                if save_nuclei:
                    output_df = {
                        'x0': res_nuclei['boxes'][:, 0].numpy(),
                        'y0': res_nuclei['boxes'][:, 1].numpy(),
                        'x1': res_nuclei['boxes'][:, 2].numpy(),
                        'y1': res_nuclei['boxes'][:, 3].numpy(),
                        'x_c': (res_nuclei['boxes'][:, 0] + res_nuclei['boxes'][:, 2]).numpy() / 2,
                        'y_c': (res_nuclei['boxes'][:, 1] + res_nuclei['boxes'][:, 3]).numpy() / 2,
                        'labels': res_nuclei['labels'].numpy(),
                        'scores': res_nuclei['scores'].numpy(),
                        **nuclei_features,
                    }
                    save_path = os.path.join(output_dir, f"{slide_id}.nuclei.csv")
                    pd.DataFrame(output_df).to_csv(save_path)
                # tmp = nuclei_features.groupby('label').agg(['mean', 'std', 'count'])

                # Generate nuclei_map
                res_nuclei = filter_wsi_results(res_nuclei, n_classes=n_classes)
                nuclei_map, r_ave = generate_nuclei_map(
                    res_nuclei, slide_size=slide_size,
                    n_classes=n_classes, use_scores=False,
                )
                logger.info(f"{nuclei_map.shape} {r_ave}")

                # Extract TME features
                scatter_img = scatter_plot(nuclei_map, r_ave,
                                           labels_color=meta_info['labels_color'],
                                           scale_factor=1. / scale_factor
                                           )
                tme_features, cloud_d, roi_mask = extract_tme_features(
                    nuclei_map, r_ave,
                    scale_factor=1. / scale_factor,
                    roi_indices=[0],
                    n_patches=n_patches,
                    patch_size=patch_size,
                    nms_thresh=nms_thresh,
                    score_thresh=score_thresh,
                    max_dist=max_dist,
                    seed=seed, device=device,
                )
                density_img = density_plot(cloud_d, scale_factor=1. / scale_factor)

                # load a thumbnail image
                svs_file = os.path.join(data_path, slide_info['img_file'])
                try:
                    xml_file = slide_info.get('xml_file')
                    xml_file = os.path.join(data_path, xml_file) if xml_file is not None else None
                    slide = Slide(svs_file, xml_file, verbose=False)
                    slide_img = slide.thumbnail((1024, 1024))
                except:
                    logger.info(f"Didn't find the original slide: {svs_file}. Will skip slide thumbnail image.")
                    slide = None
                    slide_img = None
                t3 = time.time()

                output = {
                    'base_features': {**nuclei_features, **tme_features},
                    'nuclei_map': nuclei_map, 'r_ave': r_ave, 'cloud_d': cloud_d,
                    'slide_img': slide_img, 'scatter_img': scatter_img,
                    'density_img': density_img, 'roi_mask': roi_mask, 'time': t3 - t0,
                }
                logger.info({k: (v if isinstance(v, numbers.Number) else f'len={len(v)}')
                             for k, v in output['base_features'].items()})
                logger.info(f"total time: {output['time']} s.")

                if save_img:
                    logger.info(f"Save images: ", end="")
                    t0 = time.time()
                    if slide_img is not None:
                        save_path = os.path.join(output_dir, f"{slide_id}.slide_img.png")
                        skimage.io.imsave(save_path, (slide_img * 255.0).astype(np.uint8))

                    save_path = os.path.join(output_dir, f"{slide_id}.scatter_img.png")
                    skimage.io.imsave(save_path, (scatter_img * 255.0).astype(np.uint8))

                    save_path = os.path.join(output_dir, f"{slide_id}.density_img.png")
                    skimage.io.imsave(save_path, (density_img * 255.0).astype(np.uint8))

                    save_path = os.path.join(output_dir, f"{slide_id}.roi_mask.png")
                    skimage.io.imsave(save_path, (roi_mask * 255.0).astype(np.uint8))
                    logger.info(f"{time.time() - t0} s. ")

            with open(pkl_filename, 'wb') as f:  # save results to pkl
                pickle.dump(output, f)

        # register pkl_file to slide_id
        outputs[slide_id] = pkl_filename

    # Summarize normalized features
    if slides_mapping_file is not None and os.path.exists(slides_mapping_file):
        slide_pat_map = pd.read_csv(slides_mapping_file).to_dict()
    else:
        slide_pat_map = None

    # Summarize results based on patient_id
    bfs = {}
    for slide_id, pkl_filename in outputs.items():
        with open(pkl_filename, 'rb') as f:
            output = pickle.load(f)
        bfs[slide_id] = output['base_features']
    df = summarize_normalized_features(bfs, slide_pat_map=slide_pat_map)
    csv_filename = os.path.join(output_dir, "feature_summary.csv")
    df.to_csv(csv_filename)


if __name__ == '__main__':
    pass
