import cv2
import logging
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import precomputed as P


def tta(args, images):
    """Augment all images in a batch and return list of augmented batches"""

    ret = []
    n1 = math.ceil(args.tta ** 0.5)
    n2 = math.ceil(args.tta / n1)
    k = 0
    for i in range(n1):
        for j in range(n2):
            if k >= args.tta:
                break

            dw = round(args.tta_size * images.size(2))
            dh = round(args.tta_size * images.size(3))
            w = i * (images.size(2) - dw) // max(n1 - 1, 1)
            h = j * (images.size(3) - dh) // max(n2 - 1, 1)

            imgs = images[:, :, w:w + dw, h:h + dh]
            if k & 1:
                imgs = imgs.flip(3)
            if k & 2:
                imgs = imgs.flip(2)
            if k & 4:
                imgs = imgs.transpose(2, 3)

            ret.append(nn.functional.interpolate(imgs, images.size()[2:], mode='nearest'))
            k += 1

    return ret

def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 10 ** 9) + worker_id)

def get_train_val_loader(args, predict=False):
    def train_transform1(image):
        if random.random() < 0.5:
            image = image[:, ::-1, :]
        if random.random() < 0.5:
            image = image[::-1, :, :]
        if random.random() < 0.5:
            image = image.transpose([1, 0, 2])
        image = np.ascontiguousarray(image)

        if args.scale_aug != 1:
            size = random.randint(round(512 * args.scale_aug), 512)
            x = random.randint(0, 512 - size)
            y = random.randint(0, 512 - size)
            image = image[x:x + size, y:y + size]
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)

        return image

    def train_transform2(image):
        a, b = np.random.normal(1, args.pw_aug[0], (6, 1, 1)), np.random.normal(0, args.pw_aug[1], (6, 1, 1))
        a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
        return image * a + b

    if not predict:
        train_dataset = CellularDataset(args.data, 'train_all_controls' if args.all_controls_train else 'train_controls',
                transform=(train_transform1, train_transform2), cv_number=args.cv_number,
                split_seed=args.data_split_seed, normalization=args.data_normalization)
        train = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True,
                num_workers=args.num_data_workers, worker_init_fn=worker_init_fn)

    for i in range(1 if not predict else 2):
        dataset = CellularDataset(args.data, 'val' if i == 0 else 'train', cv_number=args.cv_number,
                split_seed=args.data_split_seed, normalization=args.data_normalization)
        loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_data_workers,
                worker_init_fn=worker_init_fn)
        if i == 0:
            val = loader
        else:
            train = loader

    assert len(set(train.dataset.data).intersection(set(val.dataset.data))) == 0
    return train, val

def get_test_loader(args, exclude_leak=False):
    test_dataset = CellularDataset(args.data, 'test' if not exclude_leak else 'test_noleak',
            normalization=args.data_normalization)
    return DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_data_workers,
            worker_init_fn=worker_init_fn)


class CellularDataset(Dataset):
    treatment_classes = 1108

    def __init__(self, root_dir, mode, split_seed=0, cv_number=0, transform=None, normalization='global'):
        """
        :param split_seed: seed for train/val split of labeled experiments and HUVEC-18
        :param mode: possible choices:
                        train -- dataset containing only non-control images from training set
                        train_controls -- dataset containing non-control and control images from training set
                        train_all_controls -- dataset containing non-control and control images from training set and
                                              control images from validation and test set
                        val -- dataset containing only non-control images from validation set
                        test -- dataset containing only non-control images from test set
                        test_noleak -- dataset containing only non-control images from test set excluding HUVEC-18
        :param transform: tuple of 2 functions for image transformation. First is called right after loading with image
                          in numpy format. Second is called after normalization and converting to tensor
        """

        super().__init__()

        self.root = Path(root_dir)
        self.transform = transform

        assert normalization in ['global', 'experiment', 'sample']
        self.normalization = normalization

        if mode == 'train_controls':
            mode = 'train'
            move_controls = True
            all_controls = False
        elif mode == 'train_all_controls':
            mode = 'train'
            move_controls = True
            all_controls = True
        else:
            move_controls = False
            all_controls = False

        if mode == 'test_noleak':
            mode = 'test'
            exclude_leak = True
        else:
            exclude_leak = False
        assert mode in ['train', 'val', 'test']
        self.mode = mode

        csv = pd.read_csv(self.root / ('train.csv' if mode in ['train', 'val'] else 'test.csv'))
        csv_controls = pd.read_csv(self.root / ('train_controls.csv' if mode in ['train', 'val'] else 'test_controls.csv'))
        if all_controls:
            csv_controls_test = pd.read_csv(self.root / 'test_controls.csv')
        self.data = []  # (experiment, plate, well, site, cell_type, sirna or None)
        experiments = {}
        for row in chain(csv.iterrows(), csv_controls.iterrows(), *([csv_controls_test.iterrows()] if all_controls else [])):
            r = row[1]
            typ = r.experiment[:r.experiment.find('-')]
            self.data.append((r.experiment, r.plate, r.well, 1, typ, r.sirna if hasattr(r, 'sirna') else None))
            self.data.append((r.experiment, r.plate, r.well, 2, typ, r.sirna if hasattr(r, 'sirna') else None))
            if not hasattr(r, 'sirna') or r.sirna < self.treatment_classes:
                if typ not in experiments:
                    experiments[typ] = set()
                experiments[typ].add(r.experiment)
        if mode in ['train', 'val']:
            data_dict = {(e, p, w): sir for e, p, w, s, typ, sir in self.data}
            for row in pd.read_csv(self.root / 'test.csv').iterrows():
                r = row[1]
                typ = r.experiment[:r.experiment.find('-')]
                if r.experiment == 'HUVEC-18':
                    sirna = data_dict[('RPE-03', (r.plate - 2) % 4 + 1, r.well)]
                    assert sirna < self.treatment_classes
                    self.data.append((r.experiment, r.plate, r.well, 1, typ, sirna))
                    self.data.append((r.experiment, r.plate, r.well, 2, typ, sirna))
                    if typ not in experiments:
                        experiments[typ] = set()
                    experiments[typ].add(r.experiment)
            if not all_controls:
                for row in pd.read_csv(self.root / 'test_controls.csv').iterrows():
                    r = row[1]
                    typ = r.experiment[:r.experiment.find('-')]
                    if r.experiment == 'HUVEC-18':
                        sirna = data_dict[('RPE-03', (r.plate - 2) % 4 + 1, r.well)]
                        assert sirna == r.sirna or sirna == 1138 or r.sirna == 1138
                        self.data.append((r.experiment, r.plate, r.well, 1, typ, r.sirna))
                        self.data.append((r.experiment, r.plate, r.well, 2, typ, r.sirna))
        if exclude_leak:
            self.data = list(filter(lambda x: x[0] != 'HUVEC-18', self.data))

        self.cell_types = sorted(experiments.keys())
        all_data = self.data.copy()

        if mode != 'test':
            state = random.Random(split_seed)
            cells = list(map(itemgetter(1), sorted(experiments.items())))
            for i in range(len(cells)):
                cells[i] = sorted(cells[i])
                if i == 3:
                    cells[i] = cells[i] + cells[i]  # duplicate U2OS experiments for validation
                state.shuffle(cells[i])

            # cell[i] is a list of experiments for i-th cell type
            assert list(map(len, cells)) == [7, 17, 7, 6]

            # counts of experiments from given cell type for given fold
            counts = [
                [2, 2, 1, 1],
                [1, 3, 2, 1],
                [1, 3, 1, 1],
                [1, 3, 1, 1],
                [1, 3, 1, 1],
                [1, 3, 1, 1],
            ]

            splits = []
            start = [0, 0, 0, 0]
            for count in counts:
                splits.append(sorted(cells[0][start[0]:start[0] + count[0]]) +
                              sorted(cells[1][start[1]:start[1] + count[1]]) +
                              sorted(cells[2][start[2]:start[2] + count[2]]) +
                              sorted(cells[3][start[3]:start[3] + count[3]]))
                for i in range(4):
                    start[i] += count[i]
            assert start == [7, 17, 7, 6]
            logging.info('Splits: {}'.format(splits))

            if cv_number != -1:
                val = sorted(splits[cv_number])
            else:
                val = []
            all = []
            for k, v in sorted(experiments.items()):
                v = sorted(v)
                all.extend(v)
            tr = sorted(set(all) - set(val))

            if mode == 'train':
                logging.info('Train dataset: {}'.format(sorted(tr)))
                self.data = list(filter(lambda d: d[0] in tr, self.data))
            elif mode == 'val':
                logging.info('Val dataset: {}'.format(val))
                self.data = list(filter(lambda d: d[0] in val, self.data))
            else:
                assert 0

        assert len(set(self.data)) == len(self.data)
        assert len(set(all_data)) == len(all_data)

        controls = list(filter(lambda d: d[-1] is not None and d[-1] >= self.treatment_classes,
            (all_data if all_controls else self.data)))
        self.data = list(filter(lambda d: not (d[-1] is not None and d[-1] >= self.treatment_classes),
            self.data))
        if move_controls:
            self.data += controls

        self.filter()

        logging.info('{} dataset size: data: {}'.format(mode, len(self.data)))

    def filter(self, func=None):
        """
        Filter dataset by given function. If function is not specified, it will clear current filter
        :param func: func((index, (experiment, plate, well, site, cell_type, sirna or None))) -> bool
        """
        if func is None:
            self.data_indices = None
        else:
            self.data_indices = list(filter(lambda i: func(i, self.data[i]), range(len(self.data))))

    def __len__(self):
        return len(self.data_indices if self.data_indices is not None else self.data)

    def __getitem__(self, i):
        i = self.data_indices[i] if self.data_indices is not None else i
        d = self.data[i]

        images = []
        for channel in range(1, 7):
            for dir in ['train', 'test']:
                path = self.root / dir / d[0] / 'Plate{}'.format(d[1]) / '{}_s{}_w{}.png'.format(d[2], d[3], channel)
                if path.exists():
                    break
            else:
                assert 0
            images.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            assert images[-1] is not None
        image = np.stack(images, axis=-1)

        if self.transform is not None:
            image = self.transform[0](image)

        image = F.to_tensor(image)

        if self.normalization == 'experiment':
            pixel_mean = torch.tensor(P.pixel_stats[d[0]][0]) / 255
            pixel_std = torch.tensor(P.pixel_stats[d[0]][1]) / 255
        elif self.normalization == 'global':
            pixel_mean = torch.tensor(list(map(lambda x: x[0], P.pixel_stats.values()))).mean(0) / 255
            pixel_std = torch.tensor(list(map(lambda x: x[1], P.pixel_stats.values()))).mean(0) / 255
        elif self.normalization == 'sample':
            pixel_mean = image.mean([1, 2])
            pixel_std = image.std([1, 2]) + 1e-8
        else:
            assert 0

        image = (image - pixel_mean.reshape(-1, 1, 1)) / pixel_std.reshape(-1, 1, 1)

        if self.transform is not None:
            image = self.transform[1](image)

        cell_type = nn.functional.one_hot(torch.tensor(self.cell_types.index(d[-2]), dtype=torch.long),
                len(self.cell_types)).float()

        r = [image, cell_type, torch.tensor(i, dtype=torch.long)]
        if self.mode != 'test':
            r.append(torch.tensor(d[-1], dtype=torch.long))
        return tuple(r)
