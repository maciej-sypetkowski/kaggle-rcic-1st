#!/usr/bin/env python3

import argparse
import logging
import math
import numpy as np
import pandas as pd
import pickle
import sys
from collections import defaultdict
from functools import reduce
from itertools import permutations, groupby, chain
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path

from scipy.optimize import linear_sum_assignment


class Dataset:
    CLASSES = 1108

    def __init__(self, path):
        self.data = {}
        self.controls = {}

        path = Path(path)
        for is_control, file in [(0, 'train.csv'), (1, 'train_controls.csv'), (1, 'test_controls.csv')]:
            csv = pd.read_csv(path / file)
            for row in csv.iterrows():
                r = row[1]
                (self.controls if is_control else self.data)[self.split(r.id_code)] = r.sirna

        # HUVEC-18 leak
        for file in ['test.csv']:
            csv = pd.read_csv(path / file)
            for row in csv.iterrows():
                r = row[1]
                if self.split(r.id_code)[0:2] == ('HUVEC', '18'):
                    s = self.split(r.id_code)
                    s = list(s)
                    s[0] = 'RPE'
                    s[1] = '03'
                    s[2] = (s[2] - 1) % 4
                    s = tuple(s)
                    assert self.data[s] < self.CLASSES
                    self.data[self.split(r.id_code)] = self.data[s]

        self.groups, self.group_assignment = self._get_groups()

    @staticmethod
    def split(id_code):
        """Return (cell_type, experiment number of given cell type, plate number, well)"""

        a = id_code.find('-')
        b = id_code.find('_')
        c = id_code.rfind('_')
        return id_code[:a], id_code[a + 1:b], int(id_code[b + 1:c]) - 1, id_code[c + 1:]

    def _get_groups(self):
        """Calculate class groups that are on plates and assignment for the labeled set"""

        data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))
        for (serie, exper, plate, _), sirna in self.data.items():
            data[serie][exper][plate][sirna] += 1

        groups = set()
        for serie in data:
            for exper in data[serie]:
                for plate in data[serie][exper]:
                    k = tuple(sorted(list(data[serie][exper][plate].keys())))
                    if len(k) == self.CLASSES // 4:
                        groups.add(k)
        groups = sorted(groups)
        assert len(groups) == 4
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                assert len(set(groups[i]).intersection(set(groups[j]))) == 0

        assignment = {}
        for serie in data:
            for exper in data[serie]:
                gs = []
                for plate in data[serie][exper]:
                    k = tuple(sorted(list(data[serie][exper][plate].keys())))
                    sc = [len(set(g).intersection(set(k))) for g in groups]
                    assert sum(sc) == max(sc)
                    g = sc.index(max(sc))
                    gs.append(g)
                assignment[(serie, exper)] = tuple(gs)
                assert(sorted(gs) == [0, 1, 2, 3])

        return groups, assignment

    def assign_groups(self, data):
        """Find group assignments as dictionary in format {code_id: list_of_classes}"""

        ret = {}
        for exper_name, exper in groupby(sorted(data), key=lambda x: self.split(x[0])[:2]):
            exper = list(exper)
            ks, vs = [], []
            for _, v in groupby(sorted(exper), key=lambda x: self.split(x[0])[2]):
                v = list(v)
                ks.append(list(map(itemgetter(0), v)))
                vs.append(list(map(itemgetter(1), v)))
            # ks[i][j] -- code id of j-th well on i-th plate of experiment 'exper_name'
            # vs[i][j] -- logits for j-th well on i-th plate of experiment 'exper_name'

            scs = []
            for v in vs:
                v = np.array(v)
                v = v.argmax(1)
                sc = [len(list(filter(lambda x: x in g, v))) for g in map(set, self.groups)]
                scs.append(sc)
            # scs[i][j] -- number of best classes that are on i-th plate and are in j-th class group

            scs = np.array(scs)
            scs = scs / scs.sum(0, keepdims=True)

            perms = []
            for perm in permutations(range(len(vs))):
                score = 0
                for i, j in enumerate(perm):
                    score += scs[i, j]
                perms.append((score, perm))
            perms.sort(key=lambda x: -x[0])

            best_perm = perms[0][1]
            conf = perms[0][0] - (perms[1][0] if len(perms) > 1 else perms[0][0])
            score = perms[0][0]

            if exper_name in self.group_assignment:
                if self.group_assignment[exper_name] == best_perm:
                    assignment_type = 'correct_assignment'
                else:
                    assignment_type = 'incorrect_assignment'
            else:
                assignment_type = 'prediction'

            logging.info('groups: {:8} -> {} ( score: {:.5f}  conf: {:.5f} ) {}  size: {}'.format(
                '-'.join(exper_name), best_perm, score, conf, assignment_type, sum(map(len, ks))))

            for i, k in enumerate(ks):
                for n in k:
                    assert n not in ret
                    ret[n] = self.groups[best_perm[i]]
        return ret

    def accuracy(self, data):
        if isinstance(data, dict):
            data = data.items()

        correct_hits = 0
        total = 0
        correct_hits_exper = defaultdict(lambda: 0)
        total_exper = defaultdict(lambda: 0)
        for k, v in data:
            split = self.split(k)
            total += 1
            total_exper[split[:2]] += 1
            if v == self.data[split]:
                correct_hits += 1
                correct_hits_exper[split[:2]] += 1

        if total == 0:
            return 0, {}
        return correct_hits / total, dict(map(lambda x: (x[0][0], x[0][1] / x[1][1] if x[1][1] != 0 else 0),
            zip(correct_hits_exper.items(), total_exper.items())))


class PredictionGroup:
    def __init__(self, x):
        if isinstance(x, dict):
            x = x.items()
        self.data = []
        for k, v in x:
            for pred in (v if isinstance(v, list) else [v]):
                self.data.append((k, pred[:Dataset.CLASSES]))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def combine(self, f=None):
        if f is None:
            f = lambda x: x.sum(0)

        r = {}
        for code_id, iterable in groupby(sorted(self.data, key=lambda x: x[0]), key=lambda x: x[0]):
            iterable = list(iterable)
            pred = np.array(list(map(lambda x: x[1], iterable)))
            r[code_id] = f(pred)
        return PredictionGroup(r)

    def retain_plate_classes(self, assignment):
        r = []
        for code_id, pred in self:
            new_pred = pred.copy()
            new_pred[list(set(range(len(new_pred))) - set(assignment[code_id]))] = -np.inf
            r.append((code_id, new_pred))
        return PredictionGroup(r)

    def assign_argmax(self):
        for k, v in self:
            yield k, v.argmax()

    def _assign_unique_in_plate(self, plate):
        preds = np.array(list(map(itemgetter(1), plate)))
        preds = np.vectorize(lambda x: x if x != -np.inf else -1e10)(preds)
        _, indices = linear_sum_assignment(-preds)
        return [(k, v.item()) for (k, _), v in zip(plate, indices)]

    def assign_unique(self, pool=__builtins__):
        plates = (list(plate) for _, plate in groupby(sorted(self, key=itemgetter(0)),
            key=lambda x: Dataset.split(x[0])[:3]))
        return chain(*pool.map(self._assign_unique_in_plate, plates))

    def concat(*args):
        r = []
        for w in args:
            for k, v in w:
                r.append((k, [v]))
        return PredictionGroup(r)

    def normalize(self):
        return self.map(lambda x: (x - x.mean()) / max(x.std(), 1e-8))

    def map(self, f=None):
        if not self.data:
            return PredictionGroup([])

        preds = np.array(list(map(itemgetter(1), self)))
        if f is not None:
            preds = f(preds)
        return PredictionGroup(((k, preds[i])) for i, (k, _) in enumerate(self))


class Prediction:
    def __init__(self, data, y=None):
        if y is not None:
            self.val, self.test = data, y

        else:
            if isinstance(data, Path) or isinstance(data, str):
                with Path(data).open('rb') as f:
                    data = pickle.load(f)

            self.val = PredictionGroup(data['val'])
            self.test = PredictionGroup(data['test'])

    def _map(self, f):
        if isinstance(self, Prediction):
            return Prediction(f(self.val), f(self.test))
        else:
            return Prediction(
                    f(list(map(lambda x: x.val, self))),
                    f(list(map(lambda x: x.test, self))),
            )

    def combine(self, *args, **kwargs):
        return self._map(lambda x: x.combine(*args, **kwargs))

    def retain_plate_classes(self, dataset):
        return self._map(lambda x: x.retain_plate_classes(dataset.assign_groups(x)))

    def concat(*args):
        return Prediction._map(args, lambda x: PredictionGroup.concat(*x))

    def normalize(self, *args, **kwargs):
        return self._map(lambda x: x.normalize(*args, **kwargs))

    def map(self, *args, **kwargs):
        return self._map(lambda x: x.map(*args, **kwargs))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('../data/'))
    parser.add_argument('-t', '--threads', type=int, default=12)
    parser.add_argument('-w', '--weights', type=lambda x: list(map(float, x.split(','))))
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = [1] * len(args.files)

    return args

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG, format='{asctime}:{levelname}: {message}', style='{',
            handlers=[logging.StreamHandler(sys.stderr)])
    logging.info('Args: {}'.format(args))

    pool = Pool(args.threads)

    logging.info('Loading dataset')
    dataset = Dataset(args.data)

    logging.info('Loading predictions')
    preds = []
    for i, file in enumerate(args.files):
        pred = Prediction(args.files[i])
        pred = pred.combine()
        score = dataset.accuracy(pred.val.assign_argmax())
        preds.append(pred)
        logging.info('File {} -> score: {}'.format(args.files[i], score))


    preds = list(map(lambda p: p[0].map(lambda x: (x * p[1])), zip(preds, args.weights)))
    pred = Prediction.concat(*preds)

    logging.info('Evaluating...')
    logging.info('Average score:                       {}'.format(dataset.accuracy(pred.val.assign_argmax())))
    pred = pred.combine()
    logging.info('Score after ensemble:                {}'.format(dataset.accuracy(pred.val.assign_argmax())))
    pred = pred.retain_plate_classes(dataset)
    logging.info('Score after retaining plate classes: {}'.format(dataset.accuracy(pred.val.assign_argmax())))
    logging.info('Score after linear sum assignment:   {}'.format(dataset.accuracy(pred.val.assign_unique(pool=pool))))

    logging.info('Saving csv submission into {}'.format(args.output))
    with args.output.open('w') as f:
        print('id_code,sirna', file=f)
        for k, v in sorted(pred.test.assign_unique(pool=pool)):
            print(','.join([str(k), str(v)]), file=f)
