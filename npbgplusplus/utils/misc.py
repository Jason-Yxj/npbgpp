import os
import numpy as np


def number_of_files(path: str):
    return len(os.listdir(path))


def select_indices(margin, select, total):
    return list(np.linspace(margin, total - margin, select, dtype=int))


def offset_indices(indices, offset):
    return [ind + offset for ind in indices]


def join_lists(a, b):
    return [*a, *b]

def select_test_id(scene):
    test_id = []
    if scene == 'scene0710_00':
        test_id = [510, 560, 949, 966, 979, 1377, 1496, 1540]
    elif scene == 'scene0758_00':
        test_id = [232, 538, 676, 949, 1014, 1339, 1438, 1783]
    elif scene == 'scene0781_00':
        test_id = [300, 981, 1157, 1272, 1700, 1874, 2114, 2194]
    return test_id