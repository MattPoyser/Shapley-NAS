import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import dyn_utils
import csv
import torch.nn.functional as F
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random

from torch.autograd import Variable
from model_search import Network

import time

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--celltype', type=str, default='normal', help='experiment name')
parser.add_argument('--resume', type=str, default='', help='resume from pretrained')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--shapley_momentum', type=float, default=0.8, help='momentum for updating shapley')
parser.add_argument('--step_size', type=float, default=0.1, help='step size for updating shapley')
parser.add_argument('--samples', type=int, default=10, help='number of samples for estimation')
parser.add_argument('--threshold', type=float, default=0.5, help='early truncation threshold')

parser.add_argument('--subset_size', type=int, default=100, help='subset parameter determining size of subdataset in dynamic loader')
parser.add_argument('--hardness', type=float, default=0.9, help='hardness parameter')
parser.add_argument('--mastery', type=float, default=0.1, help='mastery parameter')
parser.add_argument('--dynamic', type=bool, default=False, help='are we doing dynamic dataset')
parser.add_argument('--vanilla', type=bool, default=False, help='are we doing vanilla')
parser.add_argument('--isbad', type=bool, default=False, help='are we using bad autoencoder (ablation)')
parser.add_argument('--isablation', type=bool, default=False, help='are we using restricted dataset size? (ablation)')
parser.add_argument('--isTree', type=bool, default=False, help='do we use tree in dynamic dataloader (probably true)')
parser.add_argument('--init_train_epochs', type=int, default=5, help='minimum no. epochs to train before updating dynamic subset')
parser.add_argument('--is_csv', type=bool, default=False, help='saving with csv?')
parser.add_argument('--is_detection', type=bool, default=False, help='object detection?')
parser.add_argument('--ncc', type=bool, default=False, help='are we on ncc?')
parser.add_argument('--visualize', type=bool, default=False, help='are we visualizing results')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
if args.set == 'cifar100':
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    grayscale = False
    if args.set == "mnist" or args.set == "fashion":
        grayscale = True

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, grayscale=grayscale)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        model.show_arch_parameters()
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

    arch_params = list(map(id, model.arch_parameters()))
    weight_params = filter(lambda p: id(p) not in arch_params,
                           model.parameters())
    optimizer = torch.optim.SGD(
        weight_params,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # if args.set == 'cifar100':
    #     train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    # else:
    #     train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    train_data, val_data = dyn_utils.get_data(args)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if args.isablation:
        assert not args.dynamic
        assert args.vanilla
        train_indices = random.sample(list(range(num_train)), 1000)
        val_indices = random.sample(list(range(len(val_data))), 1000)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True)

        valid_queue = torch.utils.data.DataLoader(
            # train_data, batch_size=1024,
            val_data, batch_size=1024,
            sampler=val_sampler,
            pin_memory=True, num_workers=8)
    else:
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True)

        valid_queue = torch.utils.data.DataLoader(
            # train_data, batch_size=1024,
            val_data, batch_size=1024,
            # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=8)

    infer_queue = torch.utils.data.DataLoader(
        # train_data, batch_size=args.batch_size // 2,
        val_data, batch_size=args.batch_size // 2,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    ops = []
    for cell_type in ['normal', 'reduce']:
        for edge in range(model.num_edges):
            ops.append(['{}_{}_{}'.format(cell_type, edge, i) for i in
                        range(0, model.num_ops)])
    ops = np.concatenate(ops)

    if args.resume:
        train_epochs = [0, 25]
    else:
        pretrain_epochs = 25
        train_epochs = [pretrain_epochs, args.epochs - pretrain_epochs]
    epoch = 0
    accum_shaps = [1e-3 * torch.randn(model.num_edges, model.num_ops).cuda(),
                   1e-3 * torch.randn(model.num_edges, model.num_ops).cuda()]

    hardness = None
    just_updated = True
    valid_acc = 0

    for i, current_epochs in enumerate(train_epochs):
        assert i == 1 or i == 0 # sanity check
        for e in range(current_epochs):
            epoch_type = get_epoch_type(epoch, hardness, valid_acc, is_pretrain=1 if i==0 else 0)
            if epoch_type or just_updated or not args.dynamic:  # 1 is train, as normal (0 is dataset update)
                just_updated = False
                scheduler.step()
                lr = scheduler.get_lr()[0]
                logging.info('epoch %d lr %e', epoch, lr)

                model.show_arch_parameters()
                genotype = model.genotype()
                logging.info('genotype = %s', genotype)

                if i == len(train_epochs) - 1:
                    shap_normal, shap_reduce = shap_estimation(valid_queue, model, criterion,
                                                               ops, num_samples=args.samples, threshold=args.threshold)
                    accum_shaps = change_alpha(model, [shap_normal, shap_reduce], accum_shaps,
                                               momentum=args.shapley_momentum, step_size=args.step_size)

                train_acc, train_obj, hardness, correct  = train(train_queue, model, criterion, optimizer)
                logging.info('train_acc %f', train_acc)

                # validation
                if epoch == args.epochs - 1 or epoch % 2 == 0:
                    valid_acc, valid_obj = infer(infer_queue, model, criterion)
                    logging.info('valid_acc %f', valid_acc)

                if not args.resume and epoch == pretrain_epochs - 1:
                    utils.save(model, os.path.join(args.save, 'weights_pretrain.pt'))

                utils.save(model, os.path.join(args.save, 'weights.pt'))
                epoch += 1
            else:
                print("updating subset")
                train_queue.dataset.update_subset(hardness, epoch)
                save_indices(train_queue.dataset.get_printable(), epoch, [train_queue.dataset.full_set.__getitem__(idx) for idx in train_queue.dataset.idx])
                just_updated = True
                if args.ncc and args.visualize:
                    train_queue.dataset.visualize(framework="shapleydarts")

    model.show_arch_parameters()
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)


def remove_players(normal_weights, reduce_weights, op):
    selected_cell = str(op.split('_')[0])
    selected_eid = int(op.split('_')[1])
    opid = int(op.split('_')[-1])
    proj_mask = torch.ones_like(normal_weights[selected_eid])
    proj_mask[opid] = 0
    if selected_cell in ['normal']:
        normal_weights[selected_eid] = normal_weights[selected_eid] * proj_mask
    else:
        reduce_weights[selected_eid] = reduce_weights[selected_eid] * proj_mask


def shap_estimation(valid_queue, model, criterion, players, num_samples, threshold=0.5):
    """
    Implementation of Monte-Carlo sampling of Shapley value for operation importance evaluation

    """

    permutations = None
    n = len(players)
    sv_acc = np.zeros((n, num_samples))

    with torch.no_grad():

        if permutations is None:
            # Keep the same permutations for all batches
            permutations = [np.random.permutation(n) for _ in range(num_samples)]

        for j in range(num_samples):
            x, y = next(iter(valid_queue))
            x, y = x.cuda(), y.cuda(non_blocking=True)
            logits = model(x, weights_dict=None)
            ori_prec1, = utils.accuracy(logits, y, topk=(1,))

            normal_weights = model.get_projected_weights('normal')
            reduce_weights = model.get_projected_weights('reduce')

            acc = ori_prec1.data.item()
            print('MC sampling %d times' % (j + 1))

            for idx, i in enumerate(permutations[j]):

                remove_players(normal_weights, reduce_weights, players[i])

                logits = model(x, weights_dict={'normal': normal_weights, 'reduce': reduce_weights})
                prec1, = utils.accuracy(logits, y, topk=(1,))
                new_acc = prec1.item()
                delta_acc = acc - new_acc
                sv_acc[i][j] = delta_acc
                acc = new_acc
                # print(players[i], delta_acc)

                if acc < threshold * ori_prec1:
                    break

        result = np.mean(sv_acc, axis=-1) - np.std(sv_acc, axis=-1)
        shap_acc = np.reshape(result, (2, model.num_edges, model.num_ops))
        shap_normal, shap_reduce = shap_acc[0], shap_acc[1]
        return shap_normal, shap_reduce


def change_alpha(model, shap_values, accu_shap_values, momentum=0.8, step_size=0.1):
    assert len(shap_values) == len(model.arch_parameters())

    shap = [torch.from_numpy(shap_values[i]).cuda() for i in range(len(model.arch_parameters()))]

    for i, params in enumerate(shap):
        mean = params.data.mean()
        std = params.data.std()
        params.data.add_(-mean).div_(std)

    updated_shap = [
        accu_shap_values[i] * momentum \
        + shap[i] * (1. - momentum)
        for i in range(len(model.arch_parameters()))]

    for i, p in enumerate(model.arch_parameters()):
        p.data.add_((step_size * updated_shap[i]).to(p.device))

    return updated_shap


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    hardness = [None for i in range(len(train_queue))]
    correct = [None for i in range(len(train_queue))]
    batch_size = args.batch_size

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        new_hardness, new_correct = get_hardness(logits.cpu(), target.cpu(), False)
        hardness[(step * batch_size):(step * batch_size) + batch_size] = new_hardness  # assumes batch 1 takes idx 0-8, batch 2 takes 9-16, etc.
        correct[(step * batch_size):(step * batch_size) + batch_size] = new_correct

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, hardness, correct


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):

        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

################################# dynamic functions #####################################
def save_indices(data, epoch, images=None):
    if not args.badpath and not args.nosave:
        if args.ncc:
            with open(f'/home2/lgfm95/nas/pdarts/tempSave/curriculums/{args.dataset}/indices_{args.dataset}_{epoch}.csv', 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=' ')
                csv_writer.writerow(data)
            if images is not None:
                image_dir = f'/home2/lgfm95/nas/pdarts/tempSave/curriculums/{args.dataset}/indices_{args.dataset}_{epoch}'
                os.makedirs(image_dir)
                for q, image in enumerate(images):
                    image.save(image_dir + f"{q}.png")

        else:
            with open(f'/hdd/PhD/nas/pdarts/tempSave/curriculums/{args.dataset}/indices_{args.dataset}_{epoch}.csv', 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=' ')
                csv_writer.writerow(data)


# low value for hardness means harder.
def get_hardness(output, target, is_multi):
    if not is_multi:
        # currently a binary association between correct classication => 0.8
        # we want it to be a softmax representation. if we instead take crossentropy loss of each individual cf target
        _, predicted = torch.max(output.data, 1)
        confidence = F.softmax(output, dim=1)
        try:
            hardness_scaler = np.where((predicted == target), 1, 0.1) # if correct, simply use confidence as measure of hardness
        except RuntimeError:
            raise AttributeError(output.shape, target.shape, predicted.shape, confidence.shape)
        # therefore if model can easily say yep this is object X, then confidence will be high. if it only just manages to identify
        # object X, confidence if lower
        # if object X is misclassified, hardness needs to be lower still.
        # assumes that it does not confidently misclassify.
        # raise AttributeError(output.shape, target.shape, predicted.shape, confidence.shape, hardness_scaler)
        hardness = [(confidence[i][predicted[i]] * hardness_scaler[i]).item() for i in range(output.size(0))]
    else:
        output = torch.sigmoid(output.float()).detach()
        output[output>0.5] = 1
        output[output<=0.5] = 0
        confidence = F.softmax(output, dim=1)

        hardness_scaler = []
        hardness = []
        assert len(output) == len(target) # should both be equal to batch size
        for q in range(len(output)):
            assert len(output[q]) == len(target[q]) # should both be equal to num_classes eg 184
            correct_avg = (np.array(output[q]) == np.array(target[q])).sum() / len(output[q])
            if correct_avg > 0.5: # this could be another threshold we change, or have it == hardness threshold
                hardness_scaler.append(1)
            else:
                hardness_scaler.append(0.1)

            correct = np.where(np.array(output[q]) == np.array(target[q]))[0]
            hardness_value = [confidence[q][i] * 1 if i in correct else confidence[q][i] * 0.1 for i in range(len(output[q]))]
            hardness.append(sum(hardness_value) / len(output[q]))


        hardness_scaler = np.asarray(hardness_scaler)
        hardness = np.array(hardness)
        # raise AttributeError(output, target, hardness_scaler, hardness)

    return hardness, hardness_scaler


def get_epoch_type(epoch, hardness, top1, is_pretrain):
    # naive alternate, starting with normal training
    if not args.dynamic or epoch < args.init_train_epochs or is_pretrain:
        return 1
    is_mastered = get_mastered(hardness, top1)
    if is_mastered:
        print("mastered, therefore epoch type 0")
        return 0
    print("not mastered, therefore epoch type 1")
    return 1


def get_mastered(hardness, top1):
    # if fraction of times where image is unconfidently/mis-classified is less than mastery threshold
    # TODO use hardness across history eg mean hardness over last 5
    # print("ahard", "\n")
    # for aHard in hardness:
        # print("ahard", aHard)
    # print("len hardness", len(hardness))
    # print("len hard ones", np.where(np.array(hardness) > 0.5))
    # print("len hard ones", len(np.where(np.array(hardness) > 0.5)[0]))
    # print("hardness calculations: ", (len(np.where(np.array(hardness) > g_config.hardness)[0]) / len(hardness)), g_config.mastery)

    #if percentage of items considered hard exceeds a mastery threshold, update the subset.
    if top1 is None:
        if (len(np.where(np.array(hardness) > args.hardness)[0]) / len(hardness)) < args.mastery:
            print("therefore not mastered")
            return 0
    else:
        # print("grep working", top1)
        if top1 < args.mastery:
            return 0
    # if len(np.where(np.array(hardness) < g_config.mastery)) > len(hardness)-2:
        # a lot of images still being misclassified
        # return 0
    print("therefore mastered")
    return 1


if __name__ == '__main__':
    start_time = time.time()
    print("start time", start_time)
    main()
    print("end time", time.time() - start_time)
