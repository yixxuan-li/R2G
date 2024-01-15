import pathlib
import os.path as osp
import pandas as pd
import numpy as np
from ast import literal_eval

from .vocabulary import build_vocab, Vocabulary
from .utils import read_lines, unpickle_data
from .utils import decode_stimulus_string

def scannet_official_train_val(valid_views=None, verbose=True):
    """
    :param valid_views: None or list like ['00', '01']
    :return:
    """
    pre_fix = osp.split(pathlib.Path(__file__).parent.absolute())[0]
    train_split = osp.join(pre_fix, 'datasets/data/scannet/splits/official/v2/scannetv2_train.txt')
    train_split = read_lines(train_split)
    test_split = osp.join(pre_fix, 'datasets/data/scannet/splits/official/v2/scannetv2_val.txt')
    test_split = read_lines(test_split)

    if valid_views is not None:
        train_split = [sc for sc in train_split if sc[-2:] in valid_views]
        test_split = [sc for sc in test_split if sc[-2:] in valid_views]

    if verbose:
        print('#train/test scans:', len(train_split), '/', len(test_split))

    scans_split = dict()
    scans_split['train'] = set(train_split)
    scans_split['test'] = set(test_split)
    return scans_split


def objects_counter_percentile(scan_ids, all_scans, prc):
    all_obs_len = list()
    for scan_id in all_scans:
        if scan_id in scan_ids:
            all_obs_len.append(len(all_scans[scan_id].three_d_objects))
    return np.percentile(all_obs_len, prc)


def load_referential_data(args, referit_csv, scans_split):
    """
    :param args:
    :param referit_csv:
    :param scans_split:
    :return:
    """
    referit_data = pd.read_csv(referit_csv)

    if args.mentions_target_class_only:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data['mentions_target_class']]
        referit_data.reset_index(drop=True, inplace=True)
        print('Dropping utterances without explicit '
              'mention to the target class {}->{}'.format(n_original, len(referit_data)))
        
    # discard between samples
        
    if "sr3d" in referit_csv and not args.with_between:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data['reference_type'] != 'between']
        referit_data.reset_index(drop=True, inplace=True)
        print('Dropping utterances with between {}->{}'.format(n_original, len(referit_data)))

    # n_original = len(referit_data)
    # for key in {'scene0657_00', 'scene0056_01', 'scene0317_00', 'scene0666_00', 'scene0422_00', 'scene0403_01', 'scene0208_00', 'scene0666_01', 'scene0408_00', 'scene0403_00', 'scene0088_03', 'scene0140_00', 'scene0384_00', 'scene0621_00', 'scene0140_01', 'scene0624_00', 'scene0558_02', 'scene0336_01', 'scene0411_01', 'scene0064_01', 'scene0040_00', 'scene0245_00', 'scene0466_01', 'scene0411_02', 'scene0669_01', 'scene0030_00', 'scene0692_02', 'scene0692_01', 'scene0092_03', 'scene0030_02', 'scene0500_00', 'scene0525_00', 'scene0092_04', 'scene0141_01', 'scene0673_05', 'scene0692_00', 'scene0336_00', 'scene0141_02', 'scene0673_04', 'scene0218_00', 'scene0566_00', 'scene0054_00', 'scene0500_01', 'scene0525_01', 'scene0218_01', 'scene0141_00', 'scene0064_00', 'scene0220_01'}:
    #     referit_data = referit_data[referit_data['scan_id'] != key]
    #     referit_data.reset_index(drop=True, inplace=True)
    # print('Dropping utterances in relation {}->{}'.format(n_original, len(referit_data)))


    if  not args.use_LLM:
        if "nr3d" in referit_csv:
            referit_data = referit_data[['tokens', 'instance_type', 'scan_id',
                                        'dataset', 'target_id', 'utterance', 'stimulus_id']]
        elif "sr3d" in referit_csv:
            referit_data = referit_data[['tokens', 'instance_type', 'scan_id',
                                        'dataset', 'target_id', 'utterance', 'stimulus_id', 'anchor_ids', 'reference_type']]
    else:
        if "nr3d" in referit_csv:
            referit_data = referit_data[['tokens', 'instance_type', 'scan_id', 'instruction',
                                        'dataset', 'target_id', 'utterance', 'stimulus_id']]
        elif "sr3d" in referit_csv:
            referit_data = referit_data[['tokens', 'instance_type', 'scan_id', 'instruction',
                                        'dataset', 'target_id', 'utterance', 'stimulus_id', 'anchor_ids', 'reference_type']]

    referit_data.tokens = referit_data['tokens'].apply(literal_eval)

    # Add the is_train data to the pandas data frame (needed in creating data loaders for the train and test)
    is_train = referit_data.scan_id.apply(lambda x: x in scans_split['train'])
    referit_data['is_train'] = is_train

    # Trim data based on token length
    train_token_lens = referit_data.tokens[is_train].apply(lambda x: len(x))
    print('{}-th percentile of token length for remaining (training) data'
          ' is: {:.1f}'.format(95, np.percentile(train_token_lens, 95)))
    n_original = len(referit_data)
    referit_data = referit_data[referit_data.tokens.apply(lambda x: len(x) <= args.max_seq_len)]
    referit_data.reset_index(drop=True, inplace=True)
    print('Dropping utterances with more than {} tokens, {}->{}'.format(args.max_seq_len, n_original, len(referit_data)))

    # do this last, so that all the previous actions remain unchanged
    if args.augment_with_sr3d is not None:
        print('Adding Sr3D as augmentation.')
        sr3d = pd.read_csv(args.augment_with_sr3d)
        sr3d.tokens = sr3d['tokens'].apply(literal_eval)
        is_train = sr3d.scan_id.apply(lambda x: x in scans_split['train'])
        sr3d['is_train'] = is_train
        sr3d = sr3d[is_train]
        sr3d = sr3d[referit_data.columns]
        print('Dataset-size before augmentation:', len(referit_data))
        referit_data = pd.concat([referit_data, sr3d], axis=0)
        referit_data.reset_index(inplace=True, drop=True)
        print('Dataset-size after augmentation:', len(referit_data))

    context_size = referit_data[~referit_data.is_train].stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    print('(mean) Random guessing among target-class test objects {:.4f}'.format( (1 / context_size).mean() ))

    return referit_data


def load_scan_related_data(preprocessed_scannet_file, verbose=True, add_pad=True):
    _, all_scans = unpickle_data(preprocessed_scannet_file)
    if verbose:
        print('Loaded in RAM {} scans'.format(len(all_scans)))

    instance_labels = set()
    for scan in all_scans:
        idx = np.array([o.object_id for o in scan.three_d_objects])
        instance_labels.update([o.instance_label for o in scan.three_d_objects])
        assert np.all(idx == np.arange(len(idx)))  # assert the list of objects-ids -is- the range(n_objects).
                                                   # because we use this ordering when we sample objects from a scan.
    all_scans = {scan.scan_id: scan for scan in all_scans}  # place scans in dictionary


    class_to_idx = {}
    i = 0
    for el in sorted(instance_labels):
        class_to_idx[el] = i
        i += 1

    if verbose:
        print('{} instance classes exist in these scans'.format(len(class_to_idx)))

    # Add the pad class needed for object classification
    if add_pad:
        class_to_idx['pad'] = len(class_to_idx)

    scans_split = scannet_official_train_val()
    return all_scans, scans_split, class_to_idx


def mean_color(scan_ids, all_scans):
    mean_rgb = np.zeros((1, 3), dtype=np.float32)
    n_points = 0
    for scan_id in scan_ids:
        color = all_scans[scan_id].color
        mean_rgb += np.sum(color, axis=0)
        n_points += len(color)
    mean_rgb /= n_points
    return mean_rgb


def compute_auxiliary_data(referit_data, all_scans, args):
    """Given a train-split compute useful quantities like mean-rgb, a word-vocabulary.
    :param referit_data: pandas Dataframe, as returned from load_referential_data()
    :param all_scans:
    :param args:
    :return:
    """
    # Vocabulary
    if args.vocab_file:
        vocab = Vocabulary.load(args.vocab_file)
        print(('Using external, provided vocabulary with {} words.'.format(len(vocab))))
    else:
        train_tokens = referit_data[referit_data.is_train]['tokens']
        vocab = build_vocab([x for x in train_tokens], args.min_word_freq)
        print(('Length of vocabulary, with min_word_freq={} is {}'.format(args.min_word_freq, len(vocab))))

    if all_scans is None:
        return vocab

    # Mean RGB for the training
    training_scan_ids = set(referit_data[referit_data['is_train']]['scan_id'])
    print('{} training scans will be used.'.format(len(training_scan_ids)))
    mean_rgb = mean_color(training_scan_ids, all_scans)

    # Percentile of number of objects in the training data
    prc = 90
    obj_cnt = objects_counter_percentile(training_scan_ids, all_scans, prc)
    print('{}-th percentile of number of objects in the (training) scans'
          ' is: {:.2f}'.format(prc, obj_cnt))

    # Percentile of number of objects in the testing data
    prc = 99
    testing_scan_ids = set(referit_data[~referit_data['is_train']]['scan_id'])
    obj_cnt = objects_counter_percentile(testing_scan_ids, all_scans, prc)
    print('{}-th percentile of number of objects in the (testing) scans'
          ' is: {:.2f}'.format(prc, obj_cnt))
    return mean_rgb, vocab


def trim_scans_per_referit3d_data(referit_data, scans):
    # remove scans not in referit_data
    in_r3d = referit_data.scan_id.unique()
    to_drop = []
    for k in scans:
        if k not in in_r3d:
            to_drop.append(k)
    for k in to_drop:
        del scans[k]
    print('Dropped {} scans to reduce mem-foot-print.'.format(len(to_drop)))
    return scans

