import math

import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        kwargs = {}
        backbone = args.backbone
        if args.backbone.startswith('mem-'):
            kwargs['memory_efficient'] = True
            backbone = args.backbone[4:]

        if backbone.startswith('densenet'):
            channels = 96 if backbone == 'densenet161' else 64
            first_conv = nn.Conv2d(6, channels, 7, 2, 3, bias=False)
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True, **kwargs)
            self.features = pretrained_backbone.features
            self.features.conv0 = first_conv
            features_num = pretrained_backbone.classifier.in_features
        elif backbone.startswith('resnet') or backbone.startswith('resnext'):
            first_conv = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True, **kwargs)
            self.features = nn.Sequential(
                first_conv,
                pretrained_backbone.bn1,
                pretrained_backbone.relu,
                pretrained_backbone.maxpool,
                pretrained_backbone.layer1,
                pretrained_backbone.layer2,
                pretrained_backbone.layer3,
                pretrained_backbone.layer4,
            )
            features_num = pretrained_backbone.fc.in_features
        elif backbone.startswith('efficientnet'):
            from efficientnet_pytorch import EfficientNet
            self.efficientnet = EfficientNet.from_pretrained(backbone)
            first_conv = nn.Conv2d(6, self.efficientnet._conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.efficientnet._conv_stem = first_conv
            self.features = self.efficientnet.extract_features
            features_num = self.efficientnet._conv_head.out_channels
        else:
            raise ValueError('wrong backbone')

        self.concat_cell_type = args.concat_cell_type
        self.classes = args.classes

        features_num = features_num + (4 if self.concat_cell_type else 0)

        self.neck = nn.Sequential(
            nn.BatchNorm1d(features_num),
            nn.Linear(features_num, args.embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(args.embedding_size),
            nn.Linear(args.embedding_size, args.embedding_size, bias=False),
            nn.BatchNorm1d(args.embedding_size),
        )
        self.arc_margin_product = ArcMarginProduct(args.embedding_size, args.classes)

        if args.head_hidden is None:
            self.head = nn.Linear(args.embedding_size, args.classes)
        else:
            self.head = []
            for input_size, output_size in zip([args.embedding_size] + args.head_hidden, args.head_hidden):
                self.head.extend([
                    nn.Linear(input_size, output_size, bias=False),
                    nn.BatchNorm1d(output_size),
                    nn.ReLU(),
                ])
            self.head.append(nn.Linear(args.head_hidden[-1], args.classes))
            self.head = nn.Sequential(*self.head)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = args.bn_mom

    def embed(self, x, s):
        x = self.features(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.concat_cell_type:
            x = torch.cat([x, s], dim=1)

        embedding = self.neck(x)
        return embedding

    def metric_classify(self, embedding):
        return self.arc_margin_product(embedding)

    def classify(self, embedding):
        return self.head(embedding)


class ModelAndLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.model = Model(args)
        self.metric_crit = ArcFaceLoss()
        self.crit = DenseCrossEntropy()

    def train_forward(self, x, s, y):
        embedding = self.model.embed(x, s)

        metric_output = self.model.metric_classify(embedding)
        metric_loss = self.metric_crit(metric_output, y)

        output = self.model.classify(embedding)
        loss = self.crit(output, y)

        acc = (output.max(1)[1] == y.max(1)[1]).float().mean().item()

        coeff = self.args.metric_loss_coeff
        return loss * (1 - coeff) + metric_loss * coeff, acc

    def eval_forward(self, x, s):
        embedding = self.model.embed(x, s)
        output = self.model.classify(embedding)
        return output

    def embed(self, x, s):
        return self.model.embed(x, s)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss / 2


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine
