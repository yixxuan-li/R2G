"""
Handle the model trainging.
include: set_gpu_to_zero_position(), create_logger(), seed_training_code(), Visualizer(), make_batch_keys()
        single_epoch_train(), compute_losses(), evaluate_on_dataset(), detailed_predictions_on_dataset(), 
        save_predictions_for_visualization(), prediction_stats(), cls_pred_stats(), save_state_dicts()
        load_state_dicts().

        Class: AverageMeter(), Visualizer(), GradualWarmupScheduler().




"""
import os
import os.path as osp
import sys
import logging
import random
import numpy as np
import pandas as pd
import torch
import tqdm
from collections import OrderedDict

from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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

class Visualizer():
    def __init__(self, top_out_path):  # This will cause error in the very old train scripts
        self.writer = SummaryWriter(top_out_path)

    # |visuals|: dictionary of images to save
    def log_images(self, visuals, step):
        for label, image_numpy in visuals.items():
            self.writer.add_images(
                label, [image_numpy], step)

    # scalars: dictionary of scalar labels and values
    def log_scalars(self, scalars, step, main_tag='metrics'):
        self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict=scalars, global_step=step)

    def log_scalar(self, tag, scalar_value, epoch):
        self.writer.add_scalar(tag = tag, scalar_value = scalar_value, global_step = epoch)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                # return self.after_scheduler.get_last_lr()
                return [group['lr'] for group in self.optimizer.param_groups]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            # warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            assert(self.multiplier==1.)
            warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                # print(epoch,lr,warmup_lr,self.last_epoch, self.total_epoch)
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                # self._last_lr = self.after_scheduler.get_last_lr()
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def seed_training_code(manual_seed, strict=False):
    """Control pseudo-randomness for reproducibility.
    :param manual_seed: (int) random-seed
    :param strict: (boolean) if True, cudnn operates in a deterministic manner
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu_to_zero_position(real_gpu_loc):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_loc)


def create_logger(log_dir, std_out=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add logging to file handler
    file_handler = logging.FileHandler(osp.join(log_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos', 'color_onehot', 'mc_attr', 'tb_attr']  # all models use these
    # batch_keys = ['objects', 'tokens', 'target_pos', 'lang_mask', 'color_feature', 'size_feature', 'position_features', 'color_token', 'object_mask']  # cause segmentation fault 
    
    if args.use_LLM:
        batch_keys += ['ins_token', 'ins_mask']

    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.target_cls_alpha > 0:
        batch_keys.append('target_class')


    return batch_keys


def single_epoch_train(model, data_loader, criteria, optimizer, device, pad_idx, args):
    """
    :param model:
    :param data_loader:
    :param criteria: (dict) holding all modules for computing the losses.
    :param optimizer:
    :param device:
    :param pad_idx: (int)
    :param args:
    :return:
    """
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    target_acc_mtr = AverageMeter()
    anchor_acc_mtr = AverageMeter()
    anchor_loss_mtr = AverageMeter()
    relation_acc_mtr = AverageMeter()
    relation_loss_mtr = AverageMeter()

    # Set the model in training mode
    model.train()
    model.object_encoder.eval()
    model.object_clf.eval()

    np.random.seed()  # call this to change the sampling of the point-clouds
    batch_keys = make_batch_keys(args)

    ## debug 
    # all = np.zeros(9)
    # count = np.zeros(9)
    # rela_dis = np.zeros((9, 10))
    # rela_all = np.zeros((9, 10))

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        res = model(batch)

        ## debug
        # rela_dis = rela_dis + res['rela_dis']
        # all = all + batch['target_pos'].size(0)
        # count = count + res['edge_correct']
        # rela_all = rela_all + res['rela_sum']

        # Backward
        optimizer.zero_grad()
        all_losses = compute_losses(batch, res, criteria, args)
        total_loss = all_losses['total_loss']
        total_loss.backward()
        optimizer.step()

        # for name, parms in model.named_parameters():
        #     if 'nsm' in name:
        #         # print('-->name:', name, '-->grad_requirs:', parms.requires_grad,)
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(total_loss.item(), batch_size)

        # referential_loss_mtr.update(all_losses['referential_loss'].item(), batch_size)
        # TODO copy the ref-loss to homogeneize the code
        referential_loss_mtr.update(all_losses['referential_loss'], batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()

        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            obj_loss_mtr.update(all_losses['obj_clf_loss'].item(), batch_size)

        if args.target_cls_alpha > 0:
            batch_guess = torch.argmax(res['target_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            target_acc_mtr.update(cls_b_acc, batch_size)

        if args.anchor_cls_alpha > 0:
            batch_guess = torch.argmax(res['anchor_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['anchor_class'].cuda()).double())
            anchor_acc_mtr.update(cls_b_acc, batch_size)
            anchor_loss_mtr.update(all_losses['anchor_clf_loss'].item(), batch_size)

        
        if args.relation_cls_alpha > 0:
            batch_guess = torch.argmax(res['relation_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['sr_type'].cuda()).double())
            relation_acc_mtr.update(cls_b_acc, batch_size)
            relation_loss_mtr.update(all_losses['relation_clf_loss'].item(), batch_size)            
    ## debug
    # print('edge_correct:', count/all, "rela_acc:", 1 - rela_dis/rela_all, "rela_sum:", rela_all)
    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_loss'] = referential_loss_mtr.avg
    metrics['train_obj_clf_loss'] = obj_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_target_cls_acc'] = target_acc_mtr.avg
    metrics['train_anchor_cls_acc'] = anchor_acc_mtr.avg
    metrics['train_anchor_loss'] = anchor_loss_mtr.avg
    metrics['train_relation_cls_loss'] = relation_loss_mtr.avg
    metrics['train_relation_cls_acc'] = relation_acc_mtr.avg

    return metrics


def compute_losses(batch, res, criterion_dict, args):
    """Calculate the loss given the model logits and the criterion
    :param batch:
    :param res: dict of logits
    :param criterion_dict: dict of the criterion should have key names same as the logits
    :param args, argparse.Namespace
    :return: scalar loss value
    """
    # Get the object language classification loss and the object classification loss
    criterion = criterion_dict['logits']
    logits = res['logits']

    # Panos-note investigating tb output (if you do it like this, it does not separate, later additions
    # to total_loss from TODO POST DEADLINE.
    # referential_loss)
    # referential_loss = criterion(logits, batch['target_pos'])
    # total_loss = referential_loss
    if args.s_vs_n_weight is not None:
        total_loss = criterion(logits, batch)
    else:
        total_loss = criterion(logits, batch['target_pos'])
    referential_loss = total_loss.item()
    obj_clf_loss = target_clf_loss = anchor_clf_loss  = relation_clf_loss = 0

    if args.obj_cls_alpha > 0:
        criterion = criterion_dict['class_logits']
        obj_clf_loss = criterion(res['class_logits'].transpose(2, 1), batch['class_labels'])
        total_loss += obj_clf_loss * args.obj_cls_alpha

    if args.target_cls_alpha > 0:
        criterion = criterion_dict['target_logits']
        target_clf_loss = criterion(res['target_logits'], batch['target_class'].long().cuda())
        total_loss += target_clf_loss * args.target_cls_alpha

    if args.anchor_cls_alpha > 0:
        criterion = criterion_dict['anchor_logits']
        anchor_clf_loss = criterion(res['anchor_logits'], batch['anchor_class'].long().cuda())
        total_loss += anchor_clf_loss * args.anchor_cls_alpha

        
    if args.relation_cls_alpha > 0:
        criterion = criterion_dict['relation_logits']
        relation_clf_loss = criterion(res['relation_logits'], batch['sr_type'].long().cuda())
        total_loss += relation_clf_loss * args.relation_cls_alpha
        
    return {'total_loss': total_loss, 'referential_loss': referential_loss,
            'obj_clf_loss': obj_clf_loss, 'target_clf_loss': target_clf_loss, 
            'anchor_clf_loss': anchor_clf_loss, 'relation_clf_loss': relation_clf_loss}


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criteria, device, pad_idx, args, randomize=False):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    target_acc_mtr = AverageMeter()
    anchor_acc_mtr = AverageMeter()
    anchor_loss_mtr = AverageMeter()
    relation_acc_mtr = AverageMeter()
    relation_loss_mtr = AverageMeter()

    # Set the model in training mode
    model.eval()

    if randomize:
        np.random.seed()  # call this to change the sampling of the point-clouds #TODO-A talk about it.
    else:
        np.random.seed(args.random_seed)

    batch_keys = make_batch_keys(args)
    
    ## debug
    # all = 0
    # count = 0 
    # rela_dis = np.zeros(10)
    # rela_all = np.zeros(10)
    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        res = model(batch)

        ## debug
        # rela_dis = rela_dis + res['rela_dis']
        # all = all + batch['target_pos'].size(0)
        # count = count + res['edge_correct']
        # rela_all = rela_all + res['rela_sum']

        all_losses = compute_losses(batch, res, criteria, args)

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(all_losses['total_loss'].item(), batch_size)

        # referential_loss_mtr.update(all_losses['referential_loss'].item(), batch_size)
        referential_loss_mtr.update(all_losses['referential_loss'], batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()

        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            obj_loss_mtr.update(all_losses['obj_clf_loss'].item(), batch_size)

        if args.target_cls_alpha > 0:
            batch_guess = torch.argmax(res['target_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            target_acc_mtr.update(cls_b_acc, batch_size)

        if args.anchor_cls_alpha > 0:
            batch_guess = torch.argmax(res['anchor_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['anchor_class'].cuda()).double())
            anchor_acc_mtr.update(cls_b_acc, batch_size)
            anchor_loss_mtr.update(all_losses['anchor_clf_loss'].item(), batch_size)


        if args.relation_cls_alpha > 0:
            batch_guess = torch.argmax(res['relation_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['sr_type'].cuda()).double())
            relation_acc_mtr.update(cls_b_acc, batch_size)
            relation_loss_mtr.update(all_losses['relation_clf_loss'].item(), batch_size)   
    ## debug  
    # print('edge_correct:', count/all,  "rela_acc:", 1 - rela_dis/rela_all, "rela_sum:", rela_all)
    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_loss'] = referential_loss_mtr.avg
    metrics['test_obj_clf_loss'] = obj_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_target_cls_acc'] = target_acc_mtr.avg
    metrics['test_anchor_cls_acc'] = anchor_acc_mtr.avg
    metrics['test_anchor_cls_loss'] = anchor_loss_mtr.avg
    metrics['test_relation_cls_loss'] = relation_loss_mtr.avg
    metrics['test_relation_cls_acc'] = relation_acc_mtr.avg

    return metrics


@torch.no_grad()
def detailed_predictions_on_dataset(model, data_loader, args, device, FOR_VISUALIZATION=True):
    model.eval()

    res = dict()
    res['guessed_correctly'] = list()
    res['confidences_probs'] = list()
    res['contrasted_objects'] = list()
    res['target_pos'] = list()
    res['context_size'] = list()
    res['guessed_correctly_among_true_class'] = list()

    batch_keys = make_batch_keys(args, extras=['context_size', 'target_class_mask'])

    if FOR_VISUALIZATION:
        res['utterance'] = list()
        res['stimulus_id'] = list()
        res['object_ids'] = list()
        res['target_object_id'] = list()
        res['distrators_pos'] = list()

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        out = model(batch)

        if FOR_VISUALIZATION:
            n_ex = len(out['logits'])
            c = batch['context_size']
            n_obj = out['logits'].shape[1]
            for i in range(n_ex):
                if c[i] < n_obj:
                    out['logits'][i][c[i]:] = -10e6

        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly'].append((predictions == batch['target_pos']).cpu().numpy())
        res['confidences_probs'].append(F.softmax(out['logits'], dim=1).cpu().numpy())
        res['contrasted_objects'].append(batch['class_labels'].cpu().numpy())
        res['target_pos'].append(batch['target_pos'].cpu().numpy())
        res['context_size'].append(batch['context_size'].cpu().numpy())

        if FOR_VISUALIZATION:
            res['utterance'].append(batch['utterance'])
            res['stimulus_id'].append(batch['stimulus_id'])
            res['object_ids'].append(batch['object_ids'])
            res['target_object_id'].append(batch['target_object_id'])
            res['distrators_pos'].append(batch['distrators_pos'])
        # print(batch.keys())
        # print("******************")
        #print(batch['target_pos'])
        #print(predictions)
        #print(predictions == batch['target_pos'])
        
    
        
        #for i,xx in enumerate(batch['object_ids']):
            #print(xx[predictions[i]])
        # also see what would happen if you where to constraint to the target's class.
        cancellation = -1e6
        mask = batch['target_class_mask']
        out['logits'] = out['logits'].float() * mask.float() + (~mask).float() * cancellation
        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly_among_true_class'].append((predictions == batch['target_pos']).cpu().numpy())

    res['guessed_correctly'] = np.hstack(res['guessed_correctly'])
    res['confidences_probs'] = np.vstack(res['confidences_probs'])
    res['contrasted_objects'] = np.vstack(res['contrasted_objects'])
    res['target_pos'] = np.hstack(res['target_pos'])
    res['context_size'] = np.hstack(res['context_size'])
    res['guessed_correctly_among_true_class'] = np.hstack(res['guessed_correctly_among_true_class'])

    #print(res.keys())
    return res


@torch.no_grad()
def save_predictions_for_visualization(model, data_loader, device, channel_last, seed=2020):
    """
    Return the predictions along with the scan data for further visualization
    """
    # batch_keys = ['objects', 'tokens', 'class_labels', 'target_pos', 'scan', 'bboxes']
    batch_keys = ['objects', 'tokens', 'class_labels', 'target_pos', 'tb_attr', 'mc_attr']#model origin return 

    # Set the model in eval mode
    model.eval()

    # Create table
    res_list = []

    # Fix the test random seed
    np.random.seed(seed)
    ind = 0
    for batch in data_loader:
        if ind == 0:
            ind += 1
            continue
        # Move the batch to gpu
        for k in batch_keys:
            if len(batch[k]) > 0:
                batch[k] = batch[k].to(device)

        if not channel_last:
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward Pass
        res = model(batch)

        batch_size = batch['target_pos'].size(0)
        for i in range(batch_size):
            # print(res['simi'][i, :].shape, res['ins_simi'][i, :].shape)
            # adata, aindex = torch.sort(res['token'][i], dim = -1, descending = True) 
            # print(adata.shape, aindex.shape)
            res_list.append({
                'scan_id': batch['scan_id'][i],
                'utterance': batch['utterance'][i],
                'target_pos': batch['target_pos'][i].cpu(),
                'confidences': res['logits'][i].cpu().numpy(),
                #'bboxes': batch['objects_bboxes'][i].cpu().numpy(),
                # 'predicted_classes': res['class_logits'][i].argmax(dim=-1).cpu().numpy(),
                'predicted_target_pos': res['logits'][i].argmax(-1).cpu().numpy(),
                'object_ids': batch['object_ids'][i],
                'context_size': batch['context_size'][i],
                'correct': batch['target_pos'][i].cpu() == res['logits'][i].argmax(-1).cpu(),
                # 'prob': res['prob'][i].cpu().numpy(),
                # 'obj_prob': res['class_logits'][i].cpu().numpy(),
                'class_label': batch['class_labels'][i].cpu().numpy(),
                # 'attention_data': adata.cpu().numpy(),
                # 'attention_index': aindex.cpu().numpy(),
                # 'attention': res['attention'][i].cpu().numpy(),
                # 'instruction_prop': res['instruction_prop'][i].cpu().numpy(),
                # 'simi': res['simi'][i, :].cpu().numpy(),
                # 'simi_index': res['simi_index'][i, :].cpu().numpy(),
                # 'ins_simi': F.softmax(res['ins_simi'][i, :], dim = -1).cpu().numpy(),
                # 'ins_simi_index': res['ins_simi_index'][i, :].cpu().numpy(),
                # 'obj_attr': res['obj_attr'][i, :].cpu().numpy(),
                # 'edge_prob':res['edge_prob'][i,:, :, :].cpu().numpy(),
                'obj_center': batch['obj_position'][i].cpu().numpy(),
                'obj_size': batch['obj_size'][i].cpu().numpy(),
                'scene_center': batch['scene_center'][i].cpu().numpy(),
                'scene_size': batch['scene_size'][i].cpu().numpy(),
                # 'sr_prob': res['sr_prob'][i].cpu().numpy(),
                # "tar_anc_sr": res['tar_anc_sr'][i].cpu().numpy(),
                "sr_type": batch['sr_type'][i].cpu().numpy(),
                'target_class': batch['target_class'][i].cpu().numpy(),
                # "pred_sr": res['relation_logits'][i],
                'anchor_class': batch['anchor_class'][i].cpu().numpy()
                # 'edge_prob': res['edge_prob_logits'][i].cpu().numpy()
                #'is_easy': batch['is_easy'][i]
                }
            )
        if ind ==1:
            break 

    return res_list


def prediction_stats(logits, gt_labels):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects
    :param gt_labels: The ground truth labels of size: B x 1
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=1)
    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """ 
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples



def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, obj_cls_path, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """

    # if map_location is None:
    #     checkpoint = torch.load(checkpoint_file)
    # else:
    #     checkpoint = torch.load(checkpoint_file, map_location=map_location)

    # for key, value in kwargs.items():
    #     if key == 'model':
    #         value.load_state_dict(checkpoint[key], strict=False)
    #     # # else:
    #     # value.load_state_dict(checkpoint[key])

    # epoch = checkpoint.get('epoch')
    # if epoch:
    #     return epoch

    epoch = None
    if checkpoint_file is not None:
        if map_location is None:
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(checkpoint_file, map_location=map_location)
        epoch = checkpoint.get('epoch')

    if obj_cls_path is not None:
        if map_location is None:
            obj_pre = torch.load(obj_cls_path)
        else:
            obj_pre = torch.load(obj_cls_path, map_location=map_location)
    
    nsm = OrderedDict()

    if checkpoint_file is not None:
        nsm = checkpoint['model']
        
    if obj_cls_path is not None: 
        # object_class_pth =  dict()
        for k, v in obj_pre['model'].items():
            if "scene_graph.single_object_encoder" in k:
                _k = k.replace("scene_graph.single_object_encoder", "object_encoder")
                nsm[_k] = v
            if "scene_graph.object_mlp" in k:
                _k = k.replace("scene_graph.object_mlp", "object_clf")
                nsm[_k] = v


    for key, value in kwargs.items():
        if key == 'model':
            value.load_state_dict(nsm, strict=False)
        else:
            value.load_state_dict(checkpoint[key])

    if epoch is not None:
        return epoch



