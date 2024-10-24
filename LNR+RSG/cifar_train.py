import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import LDAMLoss, FocalLoss
from collections import OrderedDict
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_thresh', default=59, type=int, metavar='N',
                    help='the epoch threshold for generating rare samples')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_step', default=0.01, type=float,
                    help='learning rate decay step')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 2e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--head_tail_ratio', type=float, default=0.2)
parser.add_argument('--transfer_strength', type=float, default=1.0)
parser.add_argument('--addnoise', type=int, default=0)
parser.add_argument('--onlyldam', type=int, default=0)


best_acc1 = 0


def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 1:
       args.gpu=0
    if args.addnoise or args.onlyldam:
        args.epoch_thresh = 10000
        print(args.epoch_thresh)
    for k in [1,2,3,4,5,6,7,8,9,0]:
        args.seed = k
        main_worker(args.gpu, ngpus_per_node, args, seed = k)

import pickle

def get_noise_index_refined(train_dataloader,criterion,net,net_id, args, thre = 3, class_num = 100, read = False, store = False, epoch = 0):
    if not read and not store:
        return None
    if read:
        if os.path.exists('net_'+str(net_id)+'_noise_info_cifar10.pkl'):
            print('read')
            with open('net_'+str(net_id)+'_noise_info_cifar10.pkl','rb') as j:
                noise_info = pickle.load(j)
            return noise_info

    pre_dict = {}
    loss_clean = {}
    for e in range(5):
        for batch_idx, (index, x, target) in enumerate(train_dataloader):
            index, x, target = index, x.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
            out, cesc_loss, total_mv_loss, combine_target= net(x, epoch, target)
            if type(out) == type((1,2,3)):
                out = out[2]
            pospre = F.softmax(out, dim=1).cpu().detach().numpy()
            target = target.long()
            loss = criterion(out, target)
            for i, value in enumerate(x.cpu()):
                if str(index[i]) in pre_dict:
                    pre_dict[str(index[i])].append((pospre[i],target.cpu().detach().numpy()[i]))
                else:
                    pre_dict[str(index[i])] = [(pospre[i],target.cpu().detach().numpy()[i])]
            loss = criterion(out, target)
            loss_clean[batch_idx] = loss.cpu().detach().numpy()
    
    loss_last_mean = np.mean([t for t in loss_clean.values()])
    targets = np.array([t[0][1] for t in pre_dict.values()])
    for k,v in pre_dict.items():
        pre_dict[k] = (np.mean(np.array([item[0] for item in v]),axis = 0), v[0][1])
    preds = np.array([t[0] for t in pre_dict.values()])
    classes = np.array(list(range(class_num)))
    cls,cls_cnt = np.unique(targets, return_counts=True)
    print(cls,cls_cnt)
    priors = []
    for c in classes:
        if c in cls:
            priors.append(cls_cnt[np.where(cls == c)[0]][0])
        else:
            priors.append(0)
    priors = np.array(priors)
    priors = 1- (priors - np.min(priors))/(np.max(priors)-np.min(priors))
    preds_mean, preds_std = [],[]
    preds_wrong = {}
    comp = {}
    for k in classes:
        preds_wrong[k] = []
        comp[k] = {}
    for k in classes:
        if len(preds[np.where(targets != k)[0],k]) > 1:
            preds_mean.append(np.mean(preds[np.where(targets != k)[0],k]))
            preds_std.append(np.std(preds[np.where(targets != k)[0],k]))
            for j in classes:
                if j == k:
                    preds_wrong[k].append(0)
                else:
                    wrong = np.mean(preds[np.where(targets == j)[0],k])
                    preds_wrong[k].append(wrong)
        else:
            preds_mean.append(0)
            preds_std.append(-1)

    v = 1
    zscores = {}
    fliprate = {}
    noisecnt = 0
    noise_flag = {}
    print(priors)
    label_cnt = {key: 0 for key in range(len(classes))}
    zscore_m = []
    for key,value in pre_dict.items():
        (pred_prob, c) = value
        zscores[key] = (pred_prob - np.array(preds_mean)) / np.array(preds_std)
        prior_weight = np.array([max(p-priors[c],0) for p in priors])
        zscore_m.append(zscores[key])
    zscore_m = np.array(zscore_m)
    threclass = np.partition(zscore_m, -thre, axis=0)[-thre]

    for key,value in pre_dict.items():
        (pred_prob, c) = value
        prior_weight = np.array([max(p-priors[c],0) for p in priors])
        fliprate[key] = np.tanh((zscores[key]-threclass))*prior_weight

        uniform_rand = np.random.rand(len(classes))
        noise_class = np.where(fliprate[key] > uniform_rand)[0]
        if len(noise_class):
            compare_flip = fliprate[key][noise_class]
            highest_flip = np.where(compare_flip == max(compare_flip))[0]
            noise_idx = noise_class[highest_flip]
            noisecnt += len(noise_idx)
            noise_flag[key] = classes[noise_idx]
            label_cnt[classes[noise_idx][0]] += 1
            if classes[noise_idx][0] not in comp[c]:
                comp[c][classes[noise_idx][0]] = 1
            else:
                comp[c][classes[noise_idx][0]] += 1
            #print(net_id,c,fliprate[key],compare_flip,highest_flip,noise_idx,noise_flag[key],key)
        else:
            label_cnt[c] += 1

    print(comp)
    noise_info={}
    noise_info['noise_flag'] = noise_flag

    if store:
        with open('net_'+str(net_id)+'_noise_info_cifar10.pkl', 'wb') as file:
            pickle.dump(noise_info, file)
    print(net_id, noisecnt,label_cnt)
    return noise_info

def main_worker(gpu, ngpus_per_node, args, seed):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type == 'LDAM' else False

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm, head_tail_ratio = args.head_tail_ratio, transfer_strength = args.transfer_strength,  phase_train=True, epoch_thresh=args.epoch_thresh)

    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    
    train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    f1s = {}
    noise_info = None
    is_best = 0
    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None 
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 60
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
        
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
             criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return
        if args.addnoise and epoch > 60:
            if is_best:
                read = 0
            else:
                read = 1
            noise_info = get_noise_index_refined(train_loader,criterion,model,0, args, thre = 200, class_num = 10, read = read, store= is_best, epoch=epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer,noise_info)
        
        # evaluate on validation set
        acc1, f1s = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer, f1s= f1s, round=epoch)
        #print(f1s)
        if epoch > 0:
            plot_round_result(f1s,args=args)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        # remove RSG module, since RSG is not used during testing.
        new_state_dict = OrderedDict()
        for k in model.state_dict().keys():
            name = k[7:] # remove `module.`
            if 'RSG' in k:
                continue
            new_state_dict[name] = model.state_dict()[k]

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': new_state_dict,
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer,noise_info):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode

    model.train()
    fflag = 0
    end = time.time()
    for i, (index, input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.addnoise and epoch > 79:
            for j, value in enumerate(index):#str(torch.sum(value))+
                if str(index[j]) in noise_info['noise_flag'].keys():
                    fflag = 1
                    #print(str(index[i]), value[0][0][0], target[i],noise_info['noise_flag'][str(index[i])][0])
                    target[j] = noise_info['noise_flag'][str(index[j])][0]
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        
        output, cesc_loss, total_mv_loss, combine_target= model(input, epoch, target)
        if args.addnoise or args.onlyldam:# or args.onlyldam
            loss = criterion(output, combine_target)# + 0.1 * cesc_loss.mean() + 0.01 * total_mv_loss.mean()
        else:
            loss = criterion(output, combine_target) + 0.1 * cesc_loss.mean() + 0.01 * total_mv_loss.mean()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, combine_target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])) 
            print(output)
            log.write(output + '\n')
            log.flush()
    print('addnoise ', fflag)
    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def plot_round_result(results,args):
    x = results.keys()
    f1 = [value[1][0] for value in results.values()]
    gmean = [value[1][1] for value in results.values()]
    precision = [value[1][2] for value in results.values()]
    recall = [value[1][3] for value in results.values()]
    acc = [value[1][4] for value in results.values()]
    best_idx = np.where(np.array(acc) == max(acc))[0][0]
    print(best_idx)
    plt.plot(x,f1,label = 'f1'+str(np.round(f1[best_idx],4)),linestyle='dashed')
    plt.plot(x,gmean,label = 'gmean'+str(np.round(gmean[best_idx],4)),linestyle='dashdot')
    plt.plot(x,precision,label = 'precision'+str(np.round(precision[best_idx],4)),linestyle='dotted')
    plt.plot(x,recall,label = 'recall'+str(np.round(recall[best_idx],4)),linestyle='dotted')
    plt.plot(x,acc,label = 'accuracy'+str(np.round(acc[best_idx],4)),linestyle='solid')
    plt.grid(True)  # 显示网格
    plt.yticks([i/20 for i in range(21)])#str(args.onlyldam)
    plt.title(str(args.addnoise)+'cifar10_'+str(args.imb_type)+str(args.onlyldam)+'_'+str(args.seed), fontsize=10)
    plt.legend()
    #plt.show()
    plt.savefig(str(args.addnoise)+'cifar10_'+str(args.imb_type)+str(args.onlyldam)+'_'+str(args.seed)+'.pdf', format="pdf", bbox_inches="tight")
    plt.close()

def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val', round=0, f1s = None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input, phase_train=False)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        print(cf)
        classes = list(range(cf.shape[0]))
        F1_test,precision,recall,Gmean,acc = np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        for c in classes:
            TP = cf[c, c]
            TN = np.sum(cf[[nc for nc in classes if nc != c], [nc for nc in classes if nc != c]])
            FP = np.sum(cf[[nc for nc in classes if nc != c], c])
            FN = np.sum(cf[c, [nc for nc in classes if nc != c]])
            F1_test = np.append(F1_test,2*TP/(2*TP+FP + FN) if (2*TP+FP + FN) else 0)
            precision = np.append(precision,TP/(TP+FP) if (TP+FP) else 0)
            recall = np.append(recall,TP/(TP+FN) if (TP+FN) else 0)
            Gmean=np.append(Gmean,np.sqrt(precision[-1]*recall[-1]) if (TP+FP) else 0)
            acc=np.append(acc,(TP+TN)/(np.sum(cf)))
        if args.imb_type == 'step':
            minority_class = np.array(list(range(50,100)))
            F1_test = np.mean(F1_test[minority_class])
            precision = np.mean(precision[[idx for idx in classes if idx not in minority_class]])
            recall = np.mean(recall[minority_class])
            Gmean = np.mean(Gmean[minority_class])
            acc_total = np.mean(acc)
            f1s[round] = (cf,[F1_test,Gmean,precision,recall,acc_total])
        else:#python cifar_train.py --imb_type exp --imb_factor 0.01 --epoch_thresh 59 --loss_type LDAM --train_rule DRW --seed 1 --dataset cifar100 --epochs 100 --workers 2
            print('exp')
            many = np.array(list(range(0,4)))
            medium = np.array(list(range(4,7)))
            few = np.array(list(range(7,10)))
            F1_test = np.mean(recall[many])
            precision = np.mean(recall[medium])
            recall = np.mean(recall[few])
            Gmean = np.mean(Gmean[few])
            acc_total = np.mean(acc)
            f1s[round] = (cf,[F1_test,Gmean,precision,recall,acc_total])
        #python cifar_train.py --imb_type step --imb_factor 0.01 --epoch_thresh 159 --loss_type LDAM --train_rule DRW --seed 1 --dataset cifar100 --epochs 100 --addnoise 1 --workers 2
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg, f1s

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 79:
        if args.addnoise:
           lr = args.lr * 0.0002
        else:
           lr = args.lr * 0.0001
        #lr = args.lr * 0.0001
    elif epoch > 59:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

#python cifar_train.py --imb_type step --imb_factor 0.01 --transfer_strength 0 --epoch_thresh 200 --loss_type LDAM --train_rule DRW --seed 7 
#python cifar_train.py --imb_type step --imb_factor 0.01 --loss_type LDAM --train_rule DRW --seed 7 
#python python cifar_train.py --imb_type step --imb_factor 0.01 --epoch_thresh 2000 --loss_type LDAM --train_rule DRW --seed 6 
#python cifar_train.py --imb_type one --imb_factor 0.01 --epoch_thresh 2000 --loss_type LDAM --train_rule DRW --seed 1 --dataset cifar100 --epochs 100