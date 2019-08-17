import argparse
import logging
from collections import OrderedDict

import numpy as np

header = {
    'f': np.r_[0],
    'sub_idx': np.r_[1],
    'bag_idx': np.r_[2],
    'cluster': np.r_[3]
}

legacy_header = {
    'f': np.r_[0],
    'sub_idx': np.r_[1],
    'bag_idx': np.r_[2],
    'spk_lbl': np.r_[3],
    'gst_lbl': np.r_[4],
    'cluster': np.r_[5]
}

fmaps = {
    'rotated_legacy': {
        'frame': np.r_[0],
        'orig': np.r_[1:41],
        'aux': np.r_[41:46],
        'traj': np.r_[46:86],
        'hog': np.r_[86:182],
        'hof': np.r_[182:290],
        'mbh':np.r_[290:482]
    },

    'non_rotated_legacy': {
        'frame': np.r_[0],
        'aux': np.r_[1:12],
        'traj': np.r_[12:52],
        'orig': np.r_[52:92],
        'hog': np.r_[92:188],
        'hof': np.r_[188:296],
        'mbh':np.r_[296:488]
    },

    'non_rotated': {
        'aux': np.r_[0:11],
        'traj': np.r_[11:51],
        'orig': np.r_[51:91],
        'hog': np.r_[91:187],
        'hof': np.r_[187:295],
        'mbh':np.r_[295:487]
    },

    'labels': {
        'walking': 0,
        'stepping': 1,
        'drinking': 2,
        'speaking': 3,
        'hand_gesture': 4,
        'head_gesture': 5,
        'laughter': 6,
        'hair_touching': 7,
        'action_occluded': 8,
        'm_walking': 9,
        'm_stepping': 10,
        'm_drinking': 11,
        'm_speaking': 12,
        'm_hand_gesture': 13,
        'm_head_gesture': 14,
        'm_laughter': 15,
        'm_hair_touching': 16,
        'm_action_occluded': 17
    }

}

def get_non_rotated_fmap(l):
    fmap_lengths = OrderedDict([('aux', 9),
                                ('traj', l*2),
                                ('hog', 96),
                                ('hof', 108),
                                ('mbh', 192)])

    fmap = dict()
    prev = 0
    for key in fmap_lengths:
        fmap[key] = np.arange(prev, prev+fmap_lengths[key])
        prev += fmap_lengths[key]

    return merge_fmap(header, fmap)

def merge_fmap(fmap1,fmap2):
    new_map = dict()

    # find the max val in fmap1
    max_v = 0
    for k,v in fmap1.items():
        max_v = max(max_v,np.max(v))
        new_map[k] = v

    for k,v in fmap2.items():
        assert k not in new_map
        new_map[k] = max_v + v + 1

    return new_map


def extract_fmap(fmap, parts):
    idxs = np.concatenate([fmap[p] for p in parts])
    
    new_fmap = dict()
    curr = 0
    for p in parts:
        new_fmap[p] = np.r_[curr:curr+len(fmap[p])]
        curr += len(fmap[p])

    return idxs, new_fmap


def main(args):
    a = get_non_rotated_fmap(15)

    print(a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the feature maps.')

    parser.add_argument('-d', '--debug', help="Print lots of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(args)