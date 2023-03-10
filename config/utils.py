import argparse
import os
import os.path as osp

def create_dir(dir_path):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def str2bool(v):
    """
    Boolean values for argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')