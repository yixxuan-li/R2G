import argparse
import json
import os
import pprint

import numpy as np
import os.path as osp
import pandas as pd

from utils import unpickle_data, pickle_data
from utils import get_allocentric_relation
from referit3d.in_out.scan_2cad import load_scan2cad_meta_data, load_has_front_meta_data
from referit3d.in_out.scan_2cad import register_scan2cad_bboxes, register_front_direction


SCAN2CAD_META_FILE = '/data1/liyixuan/R2G/referit3d/referit3d/data/scan2cad/object_oriented_bboxes/object_oriented_bboxes_aligned_scans.json'
BAD_SCAN2CAD_MAPPINGS_FILE = '/data1/liyixuan/R2G/referit3d/referit3d/data/scan2cad/bad_mappings.json'
HAS_FRONT_FILE = '/data1/liyixuan/R2G/referit3d/referit3d/data/scan2cad/shapenet_has_front.csv'


def parse_args():
    parser = argparse.ArgumentParser('Generating annotations for spatial 3D reference (Sr3D).')

    parser.add_argument('-preprocessed_scannet_file', type=str, help='.pkl (output) of prepare_scannet_data.py',
                        required=True)

    args = parser.parse_args()

    args_string = pprint.pformat(vars(args))
    print(args_string)

    return args



if __name__ == '__main__':
    #
    # Parse arguments
    #
    args = parse_args()

    #
    # Read the scans
    #
    scannet, all_scans = unpickle_data(args.preprocessed_scannet_file)

    #
    # Augment data with scan2CAD bboxes
    #
    scan2CAD = load_scan2cad_meta_data(SCAN2CAD_META_FILE)
    # patched_scan2CAD = load_scan2cad_meta_data(SCAN2CAD_PATCHED_META_FILE)
    register_scan2cad_bboxes(all_scans, scan2CAD, BAD_SCAN2CAD_MAPPINGS_FILE)

    #
    # Augment data with has-front information (for allocentric questions).
    #
    has_front = load_has_front_meta_data(HAS_FRONT_FILE)
    register_front_direction(all_scans, scan2CAD, has_front)


    for scan in all_scans:
        len_object = len(scan.three_d_objects)
        relation_matrix = np.zeros((len_object, len_object, 10))
        for i, oi in enumerate(scan.three_d_objects):
            for j, oj in enumerate(scan.three_d_objects):
                if i == j:
                    continue
                if oj.has_front_direction:
                    allo_relation = get_allocentric_relation(oj, oi) 
                    if allo_relation == 0:
                        relation_matrix[i][j] += [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                    elif allo_relation == 2:
                        relation_matrix[i][j] += [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                    elif allo_relation == 1:
                        relation_matrix[i][j] += [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    elif allo_relation == 3:
                        relation_matrix[i][j] += [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                
                if i < j :
                    # above below
                    target_extrema = oi.get_axis_align_bbox().extrema
                    target_zmin = target_extrema[2]
                    target_zmax = target_extrema[5]
                    anchor_extrema = oj.get_axis_align_bbox().extrema
                    anchor_zmin = anchor_extrema[2]
                    anchor_zmax = anchor_extrema[5]
                    target_bottom_anchor_top_dist = target_zmin - anchor_zmax
                    target_top_anchor_bottom_dist = anchor_zmin - target_zmax
                    iou_2d, i_ratios, a_ratios = oi.iou_2d(oj)
                    i_target_ratio, i_anchor_ratio = i_ratios
                    target_anchor_area_ratio, anchor_target_area_ratio = a_ratios  
                    # Above, Below 
                    if target_bottom_anchor_top_dist > 0.06 and max(i_anchor_ratio, i_target_ratio) > 0.2:
                        relation_matrix[i][j] += [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        relation_matrix[j][i] += [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif target_top_anchor_bottom_dist > 0.06 and max(i_anchor_ratio, i_target_ratio) > 0.2:
                        relation_matrix[i][j] += [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                        relation_matrix[j][i] += [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    # supported, support
                    if i_target_ratio > 0.2 and abs(target_bottom_anchor_top_dist) <= 0.15 and target_anchor_area_ratio < 1.5:
                        relation_matrix[i][j] += [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                        relation_matrix[j][i] += [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    if i_anchor_ratio > 0.2 and abs(target_top_anchor_bottom_dist) <= 0.15 and anchor_target_area_ratio < 1.5:
                        relation_matrix[i][j] += [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                        relation_matrix[j][i] += [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        print(np.sum(np.sum(np.sum(relation_matrix))))    
        scan.relation_matrix = relation_matrix




    pickle_data("/data1/liyixuan/data/keep_all_points_00_view_with_global_scan_alignment_relation_ready.pkl", scannet, all_scans)

