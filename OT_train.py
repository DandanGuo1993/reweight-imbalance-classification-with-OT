import os
import time
import argparse
import random
import copy
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data_utils import *
from resnet import *
import shutil
from Sinkhorn_distance import SinkhornDistance
from Sinkhorn_distance_fl import SinkhornDistance as SinkhornDistance_fl
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10[default] or cifar100)')
parser.add_argument('--cost', default='combined', type=str,
                    help='[combined, label, feature, twoloss]')
parser.add_argument('--meta_set', default='prototype', type=str,
                    help='[whole, prototype]')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--imb_factor', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--save_name', default='OT_cifar10_imb0.005', type=str)
parser.add_argument('--idx', default='ours', type=str)


args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
kwargs = {'num_workers': 0, 'pin_memory': False}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

train_data_meta, train_data, test_dataset = build_dataset(args.dataset, args.num_meta)

print(f'length of meta dataset:{len(train_data_meta)}')
print(f'length of train dataset: {len(train_data)}')

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

np.random.seed(42)
random.seed(42)
torch.manual_seed(args.seed)
classe_labels = range(args.num_classes)

data_list = {}


for j in range(args.num_classes):
    data_list[j] = [i for i, label in enumerate(train_loader.dataset.targets) if label == j]


img_num_list = get_img_num_per_cls(args.dataset, args.imb_factor, args.num_meta*args.num_classes)
print(img_num_list)
print(sum(img_num_list))

im_data = {}
idx_to_del = []
for cls_idx, img_id_list in data_list.items():
    random.shuffle(img_id_list)
    img_num = img_num_list[int(cls_idx)]
    im_data[cls_idx] = img_id_list[img_num:]
    idx_to_del.extend(img_id_list[img_num:])

print(len(idx_to_del))
imbalanced_train_dataset = copy.deepcopy(train_data)
imbalanced_train_dataset.targets = np.delete(train_loader.dataset.targets, idx_to_del, axis=0)
imbalanced_train_dataset.data = np.delete(train_loader.dataset.data, idx_to_del, axis=0)
print(len(imbalanced_train_dataset))

imbalanced_train_loader = DataLoader(new_dataset(imbalanced_train_dataset, train=True),
                                     batch_size=args.batch_size, shuffle=True, **kwargs)
validation_loader = DataLoader(new_dataset(train_data_meta, train=True),
                               batch_size=args.num_classes*args.num_meta, shuffle=False, **kwargs)
test_loader = DataLoader(new_dataset(test_dataset, train=False),
                         batch_size=args.batch_size, shuffle=False, **kwargs)

best_prec1 = 0

beta = 0.9999
effective_num = 1.0 - np.power(beta, img_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
weights = torch.tensor(per_cls_weights).float()
weightsbuffer = torch.tensor([per_cls_weights[cls_i] for cls_i in imbalanced_train_dataset.targets]).to('cuda')

eplisons = 0.1
criterion = SinkhornDistance(eps=eplisons, max_iter=200, reduction=None, dis='cos').to('cuda')
criterion_label = SinkhornDistance(eps=eplisons, max_iter=200, reduction=None, dis='euc').to('cuda')
criterion_fl = SinkhornDistance_fl(eps=eplisons, max_iter=200, reduction=None).to('cuda')

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        if args.imb_factor == 0.005:
            ckpt_path = r'checkpoint/ours/pretrain/..'
       
    else:
        if args.imb_factor == 0.005:
            ckpt_path = r'checkpoint/ours/pretrain/..'
 

    model = build_model(load_pretrain=True, ckpt_path=ckpt_path)
    optimizer_a = torch.optim.SGD([model.linear.weight,model.linear.bias], args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True
    criterion_classifier = nn.CrossEntropyLoss(reduction='none').cuda()

    for epoch in range(160, args.epochs):

        train_OT(imbalanced_train_loader, validation_loader, weightsbuffer,
                 model, optimizer_a, epoch, criterion_classifier)

        prec1, preds, gt_labels = validate(test_loader, model)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            weightsbuffer_bycls = []
            for i_cls in range(args.num_classes):
                weightsbuffer_bycls.extend(weightsbuffer[imbalanced_train_dataset.targets == i_cls].data.cpu().numpy())
        
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_prec1,
            'optimizer': optimizer_a.state_dict(),
            'weights': weightsbuffer_bycls
        }, is_best)

    print('Best accuracy: ', best_prec1)


def train_OT(train_loader, validation_loader, weightsbuffer,  model, optimizer, epoch, criterion_classifier):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    val_data, val_labels, _ = next(iter(validation_loader))
    val_data = to_var(val_data, requires_grad=False)
    val_labels = to_var(val_labels, requires_grad=False).squeeze()

    if args.meta_set == 'whole':
        val_data_bycls = val_data
        val_labels_bycls = val_labels
    elif args.meta_set == 'prototype':
        val_data_bycls = torch.zeros([args.num_classes, args.num_meta, 3, 32, 32]).cuda()
        for i_cls in range(args.num_classes):
            val_data_bycls[i_cls, ::] = val_data[val_labels == i_cls]
        val_data_bycls = torch.mean(val_data_bycls, dim=1)
        val_labels_bycls = torch.tensor([i_l for i_l in range(args.num_classes)]).cuda()

    val_labels_onehot = to_categorical(val_labels_bycls).cuda()
    feature_val, _ = model(val_data_bycls)

    for i, batch in enumerate(train_loader):

        inputs, labels, ids = tuple(t.to('cuda') for t in batch)
        labels = labels.squeeze()
        labels_onehot = to_categorical(labels).cuda()


        weights = to_var(weightsbuffer[ids])
        model.eval()
        Attoptimizer = torch.optim.SGD([weights], lr=0.01, momentum=0.9, weight_decay=5e-4)

        for ot_epoch in range(1):
            feature_train, _ = model(inputs)
            probability_train = softmax_normalize(weights)

            if args.cost == 'feature':
                OTloss = criterion(feature_val.detach(), feature_train.detach(), probability_train.squeeze())
            elif args.cost == 'label':
                OTloss = criterion_label(torch.tensor(val_labels_onehot, dtype=float).cuda(),
                                         torch.tensor(labels_onehot, dtype=float).cuda(),
                                         probability_train.squeeze())
            elif args.cost == 'combined':
                OTloss = criterion_fl(feature_val.detach(), feature_train.detach(),
                                      torch.tensor(val_labels_onehot, dtype=float).cuda(),
                                      torch.tensor(labels_onehot, dtype=float).cuda(),
                                      probability_train.squeeze())
            elif args.cost == 'twoloss':
                OTloss1 = criterion(feature_val.detach(), feature_train.detach(), probability_train.squeeze())
                OTloss2 = criterion_label(torch.tensor(val_labels_onehot, dtype=float).cuda(),
                                          torch.tensor(labels_onehot, dtype=float).cuda(),
                                          probability_train.squeeze())
                OTloss = OTloss1 + OTloss2

            Attoptimizer.zero_grad()
            OTloss.backward()
            Attoptimizer.step()

        weightsbuffer[ids] = weights.data  


        model.train()
        optimizer.zero_grad()
        _, logits = model(inputs)
        loss_train = criterion_classifier(logits, labels.long())
        _, logits_val = model(val_data)        
        loss_val = F.cross_entropy(logits_val, val_labels.long(), reduction='none')
 
        loss = torch.sum(loss_train * weights.data) + 10*torch.mean(loss_val)
        loss.backward()
        optimizer.step()

        prec_train = accuracy(logits.data, labels, topk=(1,))[0]

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec_train.item(), inputs.size(0))

        if i==len(train_loader)-1 or i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, top1=top1))



def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, batch in enumerate(val_loader):
        input, target, _ = tuple(t.to('cuda') for t in batch)
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            _, output = model(input_var)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output


        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i==len(val_loader)-1: #i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, preds, true_labels


def build_model(load_pretrain, ckpt_path=None):
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if load_pretrain == True:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def softmax_normalize(weights, temperature=1.):
    return nn.functional.softmax(weights / temperature, dim=0)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(args, state, is_best):
    path = 'checkpoint/ours/'
    save_name = args.save_name
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + save_name + '_ckpt.pth.tar'
    if is_best:
        torch.save(state, filename)

def to_categorical(labels):
    labels_onehot = torch.zeros([labels.shape[0], args.num_classes])
    for label_epoch in range(labels.shape[0]):
        labels_onehot[label_epoch, labels[label_epoch]] = 1

    return labels_onehot

if __name__ == '__main__':
    main()
