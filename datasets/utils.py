"""
Handle the dataset loading.
include unpickle_data(), pickle_data(), read_lines(), decode_stimulus_string(), max_io_workers(), 
        dataset_to_dataloader(), sample_scan_object(), pad_samples(), check_segmented_object_order(),
        objects_bboxes(), instance_labels_of_context(), mean_rgb_unit_norm_transform().
"""


import numpy as np
import six
import string
import random
import torch
import os
import re
from six.moves import cPickle
from six.moves import range
import multiprocessing as mp
from enum import Enum, unique
from torch.utils.data import DataLoader
from shapely.geometry import MultiPoint, Point
from collections import defaultdict


@unique
class OrientedSections(Enum):
    front = 0
    right = 1
    back = 2
    left = 3
    grey_area = 4


# 
# referit3d utterance process
#

def unpickle_data(file_name, python2_to_3=False):
    """
    Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def pickle_data(file_name, *args):
    """
    Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()



def read_lines(file_name):
    trimmed_lines = []
    with open(file_name) as fin:
        for line in fin:
            trimmed_lines.append(line.rstrip())
    return trimmed_lines


def decode_stimulus_string(s):
    """
    For Nr3D 
    Split into scene_id, instance_label, # objects, target object id,
    distractors object id.

    :param s: the stimulus string
    """
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = \
            s.split('-', maxsplit=4)

    instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractors_ids



#
# dataset prepare 
#

def max_io_workers():
    """ number of available cores -1."""
    n = max(mp.cpu_count() - 1, 1)
    print('Using {} cores for I/O.'.format(n))
    return n


def dataset_to_dataloader(dataset, split, batch_size, n_workers, pin_memory=False, seed=None, collate_fn = None):
    """
    :param dataset:
    :param split:
    :param batch_size:
    :param n_workers:
    :param pin_memory:
    :param seed:
    :return:
    """
    batch_size_multiplier = 1 if split == 'train' else 2
    b_size = int(batch_size_multiplier * batch_size)

    drop_last = False
    if split == 'train' and len(dataset) % b_size == 1:
        print('dropping last batch during training')
        drop_last = True

    shuffle = split == 'train'

    worker_init_fn = lambda x: np.random.seed(seed)
    if split == 'test':
        if type(seed) is not int:
            warnings.warn('Test split is not seeded in a deterministic manner.')

    data_loader = DataLoader(dataset,
                             batch_size=b_size,
                             num_workers=n_workers,
                             shuffle=shuffle,
                             drop_last=drop_last,
                             pin_memory=pin_memory,
                             worker_init_fn=worker_init_fn,
                             collate_fn = collate_fn
                             )
    return data_loader


def sample_scan_object(object, n_points):
    sample = object.sample(n_samples=n_points)
    return np.concatenate([sample['xyz'], sample['color']], axis=1)


def pad_samples(samples, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.ones(shape, dtype=samples.dtype) * padding_value
        temp[:samples.shape[0], :samples.shape[1]] = samples
        samples = temp

    return samples


def check_segmented_object_order(scans):
    """ check all scan objects have the three_d_objects sorted by id
    :param scans: (dict)
    """
    for scan_id, scan in scans.items():
        idx = scan.three_d_objects[0].object_id
        for o in scan.three_d_objects:
            if not (o.object_id == idx):
                print('Check failed for {}'.format(scan_id))
                return False
            idx += 1
    return True


def objects_bboxes(context):
    b_boxes = []
    for o in context:
        bbox = o.get_bbox(axis_aligned=True)

        # Get the centre
        cx, cy, cz = bbox.cx, bbox.cy, bbox.cz

        # Get the scale
        lx, ly, lz = bbox.lx, bbox.ly, bbox.lz

        b_boxes.append([cx, cy, cz, lx, ly, lz])

    return np.array(b_boxes).reshape((len(context), 6))


def instance_labels_of_context(context, max_context_size, label_to_idx=None, add_padding=True):
    """
    :param context: a list of the objects
    :return:
    """
    instance_labels = [i.instance_label for i in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        instance_labels.extend(['pad'] * n_pad)

    if label_to_idx is not None:
        instance_labels = np.array([label_to_idx[x] for x in instance_labels])

    return instance_labels


def mean_rgb_unit_norm_transform(segmented_objects, mean_rgb, unit_norm, epsilon_dist=10e-6, inplace=True):
    """
    :param segmented_objects: K x n_points x 6, K point-clouds with color.
    :param mean_rgb:
    :param unit_norm:
    :param epsilon_dist: if max-dist is less than this, we apply not scaling in unit-sphere.
    :param inplace: it False, the transformation is applied in a copy of the segmented_objects.
    :return:
    """
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    segmented_objects[:, :, 3:6] -= np.expand_dims(mean_rgb, 0)

    # center xyz
    if unit_norm:
        xyz = segmented_objects[:, :, :3]
        mean_center = xyz.mean(axis=1)
        xyz -= np.expand_dims(mean_center, 1)
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)), -1)
        max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        segmented_objects[:, :, :3] = xyz
    return segmented_objects

def get_anchor_sections(extrema, a, dl, df, d2):
    """
    @param extrema:
    @param a:
    @param dl:
    @param df:
    @param d2:
    @return:
    """
    xmin, xmax, ymin, ymax = extrema
    b = 90 - a
    a = np.deg2rad(a)
    b = np.deg2rad(b)

    section_names = [OrientedSections.front, OrientedSections.back, OrientedSections.right, OrientedSections.left]
    ret = {}
    for section in section_names:
        if section.name == 'front':
            p1 = (xmin, ymin)
            p2 = (xmin, ymax)
            p3 = (xmin - df, ymax + (np.sin(a) * df / np.sin(b)))
            p4 = (xmin - df, ymin - (np.sin(a) * df / np.sin(b)))
            p5 = (xmin - df - d2, ymin - (np.sin(a) * df / np.sin(b)))
            p6 = (xmin - df - d2, ymax + (np.sin(a) * df / np.sin(b)))
            ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
        elif section.name == 'back':
            p1 = (xmax, ymin)
            p2 = (xmax, ymax)
            p3 = (xmax + df, ymax + (np.sin(a) * df / np.sin(b)))
            p4 = (xmax + df, ymin - (np.sin(a) * df / np.sin(b)))
            p5 = (xmax + df + d2, ymin - (np.sin(a) * df / np.sin(b)))
            p6 = (xmax + df + d2, ymax + (np.sin(a) * df / np.sin(b)))
            ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
        elif section.name == 'left':
            p1 = (xmin, ymax)
            p2 = (xmax, ymax)
            p3 = (xmax + (np.sin(a) * dl / np.sin(b)), ymax + dl)
            p4 = (xmin - (np.sin(a) * dl / np.sin(b)), ymax + dl)
            p6 = (xmin - (np.sin(a) * dl / np.sin(b)), ymax + dl + d2)
            p5 = (xmax + (np.sin(a) * dl / np.sin(b)), ymax + dl + d2)
            ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
        elif section.name == 'right':
            p1 = (xmin, ymin)
            p2 = (xmax, ymin)
            p3 = (xmax + (np.sin(a) * dl / np.sin(b)), ymin - dl)
            p4 = (xmin - (np.sin(a) * dl / np.sin(b)), ymin - dl)
            p5 = (xmin - (np.sin(a) * dl / np.sin(b)), ymin - dl - d2)
            p6 = (xmax + (np.sin(a) * dl / np.sin(b)), ymin - dl - d2)
            ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
    return ret

def which_section_point_in(anchor_bbox, anchor_sections, target_point):
    # Transform the point in order to be compared with the object's
    # axes aligned bb
    point = target_point - [anchor_bbox.cx, anchor_bbox.cy, anchor_bbox.cz]
    point = np.hstack([point, [1]]).reshape(1, -1)
    rotation = anchor_bbox.inverse_rotation_matrix()
    axis_aligned_point = np.dot(rotation, point.T).T[:, 0:3]
    [px, py, _] = axis_aligned_point.reshape(-1)

    for sec_name, section in anchor_sections.items():
        if section.contains(Point(px, py)):
            return sec_name

    # No section
    return OrientedSections.grey_area


def get_allocentric_relation(anchor, target):
    """
    Calculate the allocentric relation between an anchor and a target object
    """

    #Set the hyper-parameters(From referit3d)
    max_df = 1
    max_dl = 1
    a = 10
    d2 = 4

    df = min(2 * anchor.get_bbox().lx, max_df)
    dl = min(2 * anchor.get_bbox().ly, max_dl)

    # Get anchor oriented sections
    [xmin, ymin, _, xmax, ymax, _] = anchor.get_bbox(axis_aligned=False).extrema
    anchor_bbox_extrema = [xmin, xmax, ymin, ymax]
    anchor_sections = get_anchor_sections(anchor_bbox_extrema, a, dl, df, d2)

    # Ignore references where an anchor intersects with a target object Can be relaxed
    iou_2d, i_ratios, a_ratios = target.iou_2d(anchor)
    if np.any(np.array(i_ratios) > 0.2):
        return 4

    # get the oriented sections that each target object occupy
    center_points = target.get_center_position()
    section_id = which_section_point_in(anchor.get_bbox(), anchor_sections, center_points).value

    return section_id
