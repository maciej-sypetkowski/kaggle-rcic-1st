#!/usr/bin/env python3

import itertools
import logging
import math
import pickle
import random
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from apex import amp
from torch import nn

import dataset
import precomputed as P
from model import ModelAndLoss

def parse_args():
    def lr_type(x):
        x = x.split(',')
        return x[0], list(map(float, x[1:]))

    def bool_type(x):
        if x.lower() in ['1', 'true']:
            return True
        if x.lower() in ['0', 'false']:
            return False
        raise ValueError()

    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', choices=('train', 'val', 'predict'))
    parser.add_argument('--backbone', default='mem-densenet161',
            help='backbone for the architecture. '
                 'Supported backbones: ResNets, ResNeXts, DenseNets (from torchvision), EfficientNets. '
                 'For DenseNets, add prefix "mem-" for memory efficient version')
    parser.add_argument('--head-hidden', type=lambda x: None if not x else list(map(int, x.split(','))),
            help='hidden layers sizes in the head. Defaults to absence of hidden layers')
    parser.add_argument('--concat-cell-type', type=bool_type, default=True)
    parser.add_argument('--metric-loss-coeff', type=float, default=0.2)
    parser.add_argument('--embedding-size', type=int, default=1024)
    parser.add_argument('--bn-mom', type=float, default=0.05)
    parser.add_argument('--wd', '--weight-decay', type=float, default=1e-5)
    parser.add_argument('--label-smoothing', '--ls', type=float, default=0)
    parser.add_argument('--mixup', type=float, default=0,
            help='alpha parameter for mixup. 0 means no mixup')
    parser.add_argument('--cutmix', type=float, default=1,
            help='parameter for beta distribution. 0 means no cutmix')

    parser.add_argument('--classes', type=int, default=1139,
            help='number of classes predicting by the network')
    parser.add_argument('--fp16', type=bool_type, default=True,
            help='mixed precision training/inference')
    parser.add_argument('--disp-batches', type=int, default=50,
            help='frequency (in iterations) of printing statistics of training / inference '
                 '(e.g. accuracy, loss, speed)')

    parser.add_argument('--tta', type=int,
            help='number of TTAs. Flips, 90 degrees rotations and resized crops (for --tta-size != 1) are applied')
    parser.add_argument('--tta-size', type=float, default=1,
            help='crop percentage for TTA')

    parser.add_argument('--save',
            help='path for the checkpoint with best accuracy. '
                 'Checkpoint for each epoch will be saved with suffix .<number of epoch>')
    parser.add_argument('--load',
            help='path to the checkpoint which will be loaded for inference or fine-tuning')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--pred-suffix', default='',
            help='suffix for prediction output. '
                 'Predictions output will be stored in <loaded checkpoint path>.output<pred suffix>')

    parser.add_argument('--pw-aug', type=lambda x: tuple(map(float, x.split(','))), default=(0.1, 0.1),
            help='pixel-wise augmentation in format (scale std, bias std). scale will be sampled from N(1, scale_std) '
                 'and bias from N(0, bias_std) for each channel independently')
    parser.add_argument('--scale-aug', type=float, default=0.5,
            help='zoom augmentation. Scale will be sampled from uniform(scale, 1). '
                 'Scale is a scale for edge (preserving aspect)')
    parser.add_argument('--all-controls-train', type=bool_type, default=True,
            help='train using all control images (also these from the test set)')
    parser.add_argument('--data-normalization', choices=('global', 'experiment', 'sample'), default='sample',
            help='image normalization type: '
                 'global -- use statistics from entire dataset, '
                 'experiment -- use statistics from experiment, '
                 'sample -- use mean and std calculated on given example (after normalization)')
    parser.add_argument('--data', type=Path, default=Path('../data'),
            help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--cv-number', type=int, default=0, choices=(-1, 0, 1, 2, 3, 4, 5),
            help='number of fold in 6-fold split. '
                 'For number of given cell type experiment in certain fold see dataset.py file. '
                 '-1 means not using validation set (training on all data)')
    parser.add_argument('--data-split-seed', type=int, default=0,
            help='seed for splitting experiments for folds')
    parser.add_argument('--num-data-workers', type=int, default=10,
            help='number of data loader workers')
    parser.add_argument('--seed', type=int,
            help='global seed (for weight initialization, data sampling, etc.). '
                 'If not specified it will be randomized (and printed on the log)')

    parser.add_argument('--pl-epoch', type=int, default=None,
            help='first epoch where pseudo-labeling starts')
    parser.add_argument('--pl-size-func', type=str, default='x',
            help='function indicating percentage of the test set transferred to the training set. '
                 'Function is called once an epoch and argument "x" is number from 0 to 1 indicating '
                 'training progress (0 is first epoch of pseudo-labeling, and 1 is last epoch of traning). '
                 'For example: "x" -- constant number of test examples is added each epoch; '
                 '"x*0.6+0.4" -- 40% of test set added at the begining of pseudo-labeling and '
                 'then constant number each epoch')

    parser.add_argument('-b', '--batch_size', type=int, default=24)
    parser.add_argument('--gradient-accumulation', type=int, default=2,
            help='number of iterations for gradient accumulation')
    parser.add_argument('-e', '--epochs', type=int, default=90)
    parser.add_argument('-l', '--lr', type=lr_type, default=('cosine', [1.5e-4]),
            help='learning rate values and schedule given in format: schedule,value1,epoch1,value2,epoch2,...,value{n}. '
                 'in epoch range [0, epoch1) initial_lr=value1, in [epoch1, epoch2) initial_lr=value2, ..., '
                 'in [epoch{n-1}, total_epochs) initial_lr=value{n}, '
                 'in every range the same learning schedule is used. Possible schedules: cosine, const')
    args = parser.parse_args()

    if args.mode == 'train':
        assert args.save is not None
    if args.mode == 'val':
        assert args.save is None
    if args.mode == 'predict':
        assert args.load is not None
        assert args.save is None

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    return args

def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.mode == 'train':
        handlers.append(logging.FileHandler(args.save + '.log', mode='w'))
    if args.mode == 'predict':
        handlers.append(logging.FileHandler(args.load + '.output.log', mode='w'))
    logging.basicConfig(level=logging.DEBUG, format=head, style='{', handlers=handlers)
    logging.info('Start with arguments {}'.format(args))

def setup_determinism(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


@torch.no_grad()
def infer(args, model, loader):
    """Infer and return prediction in dictionary formatted {sample_id: logits}"""

    if not len(loader):
        return {}
    res = {}

    model.eval()
    tic = time.time()
    for i, (X, S, I, *_) in enumerate(loader):
        X = X.cuda()
        S = S.cuda()

        Xs = dataset.tta(args, X) if args.tta else [X]
        ys = [model.eval_forward(X, S) for X in Xs]
        y = torch.stack(ys).mean(0).cpu()

        for j in range(len(I)):
            assert I[j].item() not in res
            res[I[j].item()] = y[j].numpy()

        if (i + 1) % args.disp_batches == 0:
            logging.info('Infer Iter: {:4d}  ->  speed: {:6.1f}'.format(
                i + 1, args.disp_batches * args.batch_size / (time.time() - tic)))
            tic = time.time()

    return res


def predict(args, model):
    """Entrypoint for predict mode"""

    test_loader = dataset.get_test_loader(args)
    train_loader, val_loader = dataset.get_train_val_loader(args, predict=True)

    if args.fp16:
        model = amp.initialize(model, opt_level='O1')

    logging.info('Starting prediction')

    output = {}
    for k, loader in [('test', test_loader),
                      ('val', val_loader)]:
        output[k] = {}
        res = infer(args, model, loader)

        for i, v in res.items():
            d = loader.dataset.data[i]
            name = '{}_{}_{}'.format(d[0], d[1], d[2])
            if name not in output[k]:
                output[k][name] = []
            output[k][name].append(v)

    logging.info('Saving predictions to {}'.format(args.load + '.output' + args.pred_suffix))
    with open(args.load + '.output' + args.pred_suffix, 'wb') as file:
        pickle.dump(output, file)


def score(args, model, loader):
    """Return accuracy of the model on validation set"""

    logging.info('Starting validation')

    res = infer(args, model, loader)

    cell_type_c = np.array([0, 0, 0, 0])  # number of examples for given cell type
    cell_type_s = np.array([0, 0, 0, 0])  # number of correctly classified examples for given cell type
    for i, v in res.items():
        d = loader.dataset.data[i]
        r = v[:loader.dataset.treatment_classes].argmax() == d[-1]

        ser = loader.dataset.cell_types.index(d[4])
        cell_type_c[ser] += 1
        cell_type_s[ser] += r

    acc = (cell_type_s.sum() / cell_type_c.sum()).item() if cell_type_c.sum() != 0 else 0
    logging.info('Eval: acc: {} ({})'.format(cell_type_s / cell_type_c, acc))
    return acc


def get_learning_rate(args, epoch):
    assert len(args.lr[1][1::2]) + 1 == len(args.lr[1][::2])
    for start, end, lr, next_lr in zip([0] + args.lr[1][1::2],
                                       args.lr[1][1::2] + [args.epochs],
                                       args.lr[1][::2],
                                       args.lr[1][2::2] + [0]):
        if start <= epoch < end:
            if args.lr[0] == 'cosine':
                return lr * (math.cos((epoch - start) / (end - start) * math.pi) + 1) / 2
            elif args.lr[0] == 'const':
                return lr
            else:
                assert 0
    assert 0

@torch.no_grad()
def smooth_label(args, Y):
    nY = nn.functional.one_hot(Y, args.classes).float()
    nY += args.label_smoothing / (args.classes - 1)
    nY[range(Y.size(0)), Y] -= args.label_smoothing / (args.classes - 1) + args.label_smoothing
    return nY

@torch.no_grad()
def transform_input(args, X, S, Y):
    """Apply mixup, cutmix, and label-smoothing"""

    Y = smooth_label(args, Y)

    if args.mixup != 0 or args.cutmix != 0:
        perm = torch.randperm(args.batch_size).cuda()

    if args.mixup != 0:
        coeffs = torch.tensor(np.random.beta(args.mixup, args.mixup, args.batch_size), dtype=torch.float32).cuda()
        X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm,]
        S = coeffs.view(-1, 1) * S + (1 - coeffs.view(-1, 1)) * S[perm,]
        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm,]

    if args.cutmix != 0:
        img_height, img_width = X.size()[2:]
        lambd = np.random.beta(args.cutmix, args.cutmix)
        column = np.random.uniform(0, img_width)
        row = np.random.uniform(0, img_height)
        height = (1 - lambd) ** 0.5 * img_height
        width = (1 - lambd) ** 0.5 * img_width
        r1 = round(max(0, row - height / 2))
        r2 = round(min(img_height, row + height / 2))
        c1 = round(max(0, column - width / 2))
        c2 = round(min(img_width, column + width / 2))
        if r1 < r2 and c1 < c2:
            X[:, :, r1:r2, c1:c2] = X[perm, :, r1:r2, c1:c2]

            lambd = 1 - (r2 - r1) * (c2 - c1) / (img_height * img_width)
            S = S * lambd + S[perm] * (1 - lambd)
            Y = Y * lambd + Y[perm] * (1 - lambd)

    return X, S, Y

def pseudo_label(args, epoch, pl_data, model, val_loader, test_loader, train_loader):
    """Pseudo-label some test and validation examples and move them to the training set"""

    if args.pl_epoch is None or epoch < args.pl_epoch:
        return

    logging.info('Starting pseudo-labeling')

    test_loader.dataset.filter(lambda i, d: ('test', i) not in pl_data)
    test_res = infer(args, model, test_loader)
    test_loader.dataset.filter()

    val_loader.dataset.filter(lambda i, d: ('val', i) not in pl_data)
    val_res = infer(args, model, val_loader)
    val_loader.dataset.filter()

    test_res = sorted(test_res.items())
    val_res = sorted(val_res.items())


    set_classes = defaultdict(lambda: [])  # classes that are already in the training set for the plate
    for j in range(len(train_loader.dataset.data)):
        experiment_plate = train_loader.dataset.data[j][:2]
        sirna = train_loader.dataset.data[j][-1]
        set_classes[experiment_plate].append(sirna)

    confs = []
    last = None
    for k, (i, v) in itertools.chain(
            zip(itertools.repeat('val'), val_res),
            zip(itertools.repeat('test'), test_res)):
        loader = val_loader if k == 'val' else test_loader

        # assumes that both sides of an example will be next to each other
        if i % 2 == 0:
            assert last is None
            last = i, v
            continue
        else:
            last_i, last_v = last
            assert last_i == i - 1
            last = None

            logits = v + last_v  # ensemble two sites
            plate = loader.dataset.data[i][1] - 1
            experiment = loader.dataset.data[i][0]
            class_group_id = P.group_assignment[experiment][plate]
            possible_classes = P.groups[class_group_id]
            remaining_classes = list(set(range(loader.dataset.treatment_classes)) - possible_classes)
            logits[remaining_classes] = -10e6

            experiment_plate = loader.dataset.data[i][:2]
            if set_classes[experiment_plate]:
                logits[set_classes[experiment_plate]] = -10e6
            logits = logits[:loader.dataset.treatment_classes]
            r = logits.argmax().item()

            logits.sort()
            c = logits[-1] - logits[-2]
            confs.append(((k, i - 1), c, r))


    x = (epoch - args.pl_epoch + 1) / (args.epochs - args.pl_epoch + 1)
    val_test_examples = len(val_loader.dataset.data) // 2 + len(test_loader.dataset.data) // 2
    added_examples = len(pl_data) // 2
    n = round(eval('lambda x: ' + args.pl_size_func)(x) * val_test_examples) - added_examples
    n = max(n, 0)

    confs = list(filter(lambda x: x[0] not in pl_data, confs))
    confs.sort(key=lambda x: -x[1])
    confs = confs[:n]

    val_misclass = 0
    val_count = 0
    test_count = 0
    not_added_count = 0
    added_sirnas = defaultdict(set)
    for (k, i), c, r in confs:
        if k == 'val':
            d1 = val_loader.dataset.data[i]
            d2 = val_loader.dataset.data[i + 1]
        elif k == 'test':
            d1 = test_loader.dataset.data[i]
            d2 = test_loader.dataset.data[i + 1]
        else:
            assert 0
        assert d1[:3] == d2[:3] and d1[-2:] == d2[-2:]

        if r in added_sirnas[d1[:2]]:
            not_added_count += 1
            continue

        if k == 'val':
            val_count += 1
            if d1[-1] != r:
                val_misclass += 1
        elif k == 'test':
            test_count += 1
        else:
            assert 0

        added_sirnas[d1[:2]].add(r)
        pl_data.add((k, i))
        pl_data.add((k, i + 1))
        train_loader.dataset.data.append((*d1[:-1], r))
        train_loader.dataset.data.append((*d2[:-1], r))

    logging.info('Pseudo-labeling: Added {} ({} val, {} test), {} ({:.3f}%) val misclassified, '
                 '{} ({:.3f}%) not added, pl_data size {}, train size {}, threshold {}'.format(
                     n, val_count, test_count, val_misclass, val_misclass / val_count * 100 if val_count != 0 else 0,
                     not_added_count, not_added_count / (not_added_count + n) * 100 if not_added_count + n != 0 else 0,
                     len(pl_data), len(train_loader.dataset.data), confs[-1][1] if len(confs) != 0 else 'None'))


def train(args, model):
    train_loader, val_loader = dataset.get_train_val_loader(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0, weight_decay=args.wd)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if args.load is not None:
        best_acc = score(args, model, val_loader)
    else:
        best_acc = float('-inf')

    if args.mode == 'val':
        return

    if args.pl_epoch is not None:
        test_loader = dataset.get_test_loader(args, exclude_leak=True)
        pl_data = set()

    for epoch in range(args.start_epoch, args.epochs):
        if args.pl_epoch is not None:
            pseudo_label(args, epoch, pl_data, model, val_loader, test_loader, train_loader)

        with torch.no_grad():
            avg_norm = np.mean([v.norm().item() for v in model.parameters()])

        logging.info('Train: epoch {}   avg_norm: {}'.format(epoch, avg_norm))

        model.train()
        optimizer.zero_grad()

        cum_loss = 0
        cum_acc = 0
        cum_count = 0
        tic = time.time()
        for i, (X, S, _, Y) in enumerate(train_loader):
            lr = get_learning_rate(args, epoch + i / len(train_loader))
            for g in optimizer.param_groups:
                g['lr'] = lr

            X = X.cuda()
            S = S.cuda()
            Y = Y.cuda()
            X, S, Y = transform_input(args, X, S, Y)

            loss, acc = model.train_forward(X, S, Y)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (i + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            cum_count += 1
            cum_loss += loss.item()
            cum_acc += acc
            if (i + 1) % args.disp_batches == 0:
                logging.info('Epoch: {:3d} Iter: {:4d}  ->  speed: {:6.1f}   lr: {:.9f}   loss: {:.6f}   acc: {:.6f}'.format(
                    epoch, i + 1, cum_count * args.batch_size / (time.time() - tic), optimizer.param_groups[0]['lr'],
                    cum_loss / cum_count, cum_acc / cum_count))
                cum_loss = 0
                cum_acc = 0
                cum_count = 0
                tic = time.time()

        acc = score(args, model, val_loader)
        torch.save(model.state_dict(), str(args.save + '.{}'.format(epoch)))
        if acc >= best_acc:
            best_acc = acc
            logging.info('Saving best to {} with score {}'.format(args.save, best_acc))
            torch.save(model.state_dict(), str(args.save))

def main(args):
    model = ModelAndLoss(args).cuda()
    logging.info('Model:\n{}'.format(str(model)))

    if args.load is not None:
        logging.info('Loading model from {}'.format(args.load))
        model.load_state_dict(torch.load(str(args.load)))

    if args.mode in ['train', 'val']:
        train(args, model)
    elif args.mode == 'predict':
        predict(args, model)
    else:
        assert 0



if __name__ == '__main__':
    args = parse_args()
    setup_logging(args)
    setup_determinism(args)
    main(args)
