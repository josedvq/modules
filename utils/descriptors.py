from functools import reduce

import numpy as np

def merge_dicts(f, *dicts):
    return reduce(lambda d1, d2: reduce(lambda d, t:
                                        dict(list(d.items()) +
                                             [(t[0], f(d[t[0]], t[1])
                                               if t[0] in d else
                                               t[1])]),
                                        d2.items(), d1),
                  dicts, {})

class MotherDescriptorMap:
    def __init__(self, fmaps):
        self.fmaps = [DescriptorMap(fmap) for fmap in fmaps]

    def __getitem__(self, idx):
        return self.fmaps[idx]

    def __len__(self):
        return np.sum([len(fmap) for fmap in self.fmaps]).item()

    def __str__(self):
        return str(self.fmaps)

    def __repr__(self):
        return str(self.fmaps)

    def append(self, desc_map):
        self.fmaps.append(desc_map)

    def merge(self):
        dicts = [fm.fmap for fm in self.fmaps]
        f = lambda x,y: np.sort(np.concatenate([x,y]), kind='mergesort')
        dm = DescriptorMap(merge_dicts(f, *dicts))
        return dm

    def get_list_of_dicts(self):
        return [fmap.get_dict() for fmap in self.fmaps]


class DescriptorMap:
    def __init__(self, fmap):
        self.fmap = fmap
        
    def __getitem__(self, idx):
        return self.fmap[idx]

    def __len__(self):
        return len(self.join())

    def __iter__(self):
        return iter(self.fmap.items())

    def __str__(self):
        return str(self.fmap)

    def __repr__(self):
        return str(self.fmap)

    def join(self):
        return np.concatenate([v for k,v in self.fmap.items()])

    def get_dict(self):
        return self.fmap

    def keys(self):
        return self.fmap.keys()