#!/usr/bin/env python
# coding: utf-8
import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored
import pickle
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import json


from config.arguments import parse_arguments
from datasets.neural_net_oriented import load_scan_related_data, load_referential_data
from datasets.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from datasets.loading_dataset import make_data_loaders
from utils import set_gpu_to_zero_position, create_logger, seed_training_code, Visualizer, analyze_predictions
from models.r2g import create_r2g_net
from utils import single_epoch_train, evaluate_on_dataset, load_state_dicts, save_state_dicts, save_predictions_for_visualization
from datasets.utils import dataset_to_dataloader
from utils import GradualWarmupScheduler

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


object_class1 = ['table', 'pad', 'alarm', 'alarm', 'armchair', 'pad', 'backpack', 'bag', 'bag', 'ball', 'pad', 'pad', 'pad', 'banner', 'bar', 'pad', 'basket', 'pad', 'wall', 'pad', 'cabinet', 'counter', 'pad', 'door', 'pad', 'bathtub', 'pad', 'ball', 'chair', 'bear', 'bed', 'bottles', 'bench', 'bicycle', 'pad', 'pad', 'bin', 'blackboard', 'blanket', 'blinds', 'pad', 'board', 'boards', 'pad', 'pad', 'book', 'rack', 'books', 'bookshelf', 'bookshelves', 'pad', 'bottle', 'bowl', 'box', 'boxes', 'boxes', 'bar', 'pad', 'pad', 'bucket', 'board', 'bed', 'cabinet', 'door', 'doors', 'cabinets', 'pad', 'calendar', 'camera', 'can', 'pad', 'pad', 'car', 'pad', 'cardboard', 'carpet', 'pad', 'cart', 'pad', 'case', 'bottle', 'box', 'case', 'ceiling', 'fan', 'light', 'pad', 'chair', 'pad', 'pad', 'chest', 'clock', 'closet', 'ceiling', 'door', 'pad', 'doors', 'floor', 'rod', 'shelf', 'wall', 'wall', 'cloth', 'clothes', 'pad', 'pad', 'pad', 'pad', 'clothing', 'rack', 'coat', 'rack', 'rack', 'box', 'pad', 'pad', 'table', 'column', 'bin', 'pad', 'bottle', 'container', 'pad', 'pad', 'pot', 'copier', 'pad', 'couch', 'cushion', 'counter', 'box', 'crate', 'pad', 'cup', 'cups', 'curtain', 'curtains', 'cushion', 'board', 'board', 'decoration', 'desk', 'lamp', 'bin', 'table', 'rack', 'dishwasher', 'bottle', 'dispenser', 'display', 'case', 'rack', 'pad', 'pad', 'pad', 'dolly', 'door', 'pad', 'doors', 'drawer', 'rack', 'dresser', 'pad', 'pad', 'rack', 'bag', 'pad', 'pad', 'easel', 'pad', 'pad', 'pad', 'machine', 'table', 'pad', 'pad', 'machine', 'sign', 'fan', 'faucet', 'cabinet', 'alarm', 'pad', 'fireplace', 'pad', 'pad', 'floor', 'stand', 'pad', 'chair', 'chair', 'ladder', 'table', 'folder', 'bag', 'container', 'display', 'table', 'footrest', 'footstool', 'frame', 'pad', 'pad', 'furniture', 'box', 'futon', 'door', 'bag', 'doors', 'globe', 'bag', 'bar', 'bag', 'guitar', 'case', 'pad', 'pad', 'hamper', 'pad', 'rail', 'dispenser', 'towel', 'bar', 'rail', 'hanging', 'hat', 'rack', 'headboard', 'headphones', 'heater', 'pad', 'pad', 'pad', 'pad', 'bag', 'case', 'pad', 'pad', 'board', 'jacket', 'pad', 'pad', 'keyboard', 'piano', 'pad', 'cabinet', 'cabinets', 'counter', 'pad', 'pad', 'pad', 'ladder', 'lamp', 'pad', 'laptop', 'bag', 'basket', 'pad', 'hamper', 'ledge', 'legs', 'light', 'switch', 'bed', 'pad', 'luggage', 'rack', 'stand', 'box', 'machine', 'magazine', 'rack', 'mail', 'tray', 'pad', 'pad', 'map', 'chair', 'mat', 'mattress', 'pad', 'bag', 'pad', 'microwave', 'pad', 'mirror', 'doors', 'monitor', 'mouse', 'bottle', 'mug', 'book', 'stand', 'pad', 'lamp', 'nightstand', 'notepad', 'object', 'chair', 'cabinet', 'pad', 'shelf', 'ottoman', 'oven', 'pad', 'painting', 'shelf', 'wall', 'wall', 'pad', 'paper', 'bag', 'pad', 'pad', 'towel', 'dispenser', 'towel', 'tray', 'papers', 'person', 'photo', 'piano', 'pad', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'table', 'pipe', 'pipes', 'pad', 'boxes', 'plant', 'bin', 'container', 'container', 'bin', 'pad', 'pad', 'pad', 'pad', 'table', 'poster', 'pad', 'printer', 'pad', 'pot', 'plant', 'pad', 'pad', 'printer', 'projector', 'screen', 'purse', 'pad', 'rack', 'stand', 'radiator', 'rail', 'railing', 'pad', 'chair', 'bin', 'refrigerator', 'pad', 'pad', 'rod', 'poster', 'roomba', 'pad', 'table', 'rug', 'pad', 'pad', 'pad', 'pad', 'screen', 'seat', 'seating', 'machine', 'shampoo', 'bottle', 'shelf', 'shirt', 'shoe', 'rack', 'shoes', 'bag', 'shorts', 'shower', 'pad', 'curtain', 'rod', 'door', 'doors', 'floor', 'pad', 'wall', 'walls', 'pad', 'sign', 'sink', 'door', 'pad', 'pad', 'soap', 'bottle', 'pad', 'dispenser', 'pad', 'pad', 'bed', 'chair', 'pad', 'pad', 'bottle', 'chair', 'cups', 'chair', 'stair', 'rail', 'staircase', 'stairs', 'stand', 'pad', 'cup', 'statue', 'step', 'stool', 'sticker', 'stool', 'bin', 'box', 'container', 'pad', 'shelf', 'stove', 'pad', 'light', 'pad', 'suitcase', 'suitcases', 'sweater', 'pad', 'switch', 'table', 'tank', 'pad', 'pad', 'pad', 'pad', 'bear', 'telephone', 'pad', 'thermostat', 'pad', 'box', 'toaster', 'oven', 'toilet', 'pad', 'pad', 'paper', 'dispenser', 'pad', 'pad', 'pad', 'dispenser', 'pad', 'pad', 'pad', 'pad', 'towel', 'rack', 'towels', 'pad', 'piano', 'pad', 'bag', 'bin', 'cabinet', 'can', 'tray', 'rack', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'tv', 'stand', 'umbrella', 'pad', 'pad', 'vase', 'machine', 'vent', 'wall', 'hanging', 'lamp', 'rack', 'wardrobe', 'cabinet', 'closet', 'pad', 'machine', 'machine', 'bottle', 'pad', 'pad', 'heater', 'pad', 'sign', 'pad', 'whiteboard', 'pad', 'window', 'windowsill', 'wood', 'pad', 'pad', 'pad', 'pad']


vocabs = ['above', 'air', 'airplane', 'alarm', 'allocentric', 'animal', 'apron', 'armchair', 'baby', 'back', 'backpack', 'bag', 'ball', 'banana', 'bananas', 'banister', 'banner', 'bar', 'barricade', 'base', 'basket', 'bath', 'bathrobe', 'bathroom', 'bathtub', 'battery', 'beachball', 'beam', 'beanbag', 'beans', 'bear', 'bed', 'beer', 'below', 'bench', 'between', 'bicycle', 'bike', 'bin', 'black', 'blackboard', 'blanket', 'blinds', 'block', 'blue', 'board', 'boards', 'boat', 'boiler', 'book', 'books', 'bookshelf', 'bookshelves', 'boots', 'bottle', 'bottles', 'bottom', 'bowl', 'box', 'boxes', 'breakfast', 'briefcase', 'broom', 'brown', 'brush', 'bucket', 'bulletin', 'bunk', 'button', 'cabinet', 'cabinets', 'cable', 'calendar', 'camera', 'can', 'candle', 'canopy', 'car', 'card', 'cardboard', 'carpet', 'carseat', 'cart', 'carton', 'case', 'cat', 'cd', 'ceiling', 'chain', 'chair', 'chairs', 'chandelier', 'changing', 'chest', 'cleaner', 'clock', 'closest', 'closet', 'cloth', 'clothes', 'clothing', 'coat', 'coatrack', 'coffee', 'color', 'column', 'compost', 'computer', 'conditioner', 'cone', 'container', 'containers', 'controller', 'cooker', 'cooking', 'cooler', 'copier', 'corner', 'costume', 'couch', 'counter', 'covered', 'crate', 'crib', 'cup', 'cups', 'curtain', 'curtains', 'curve', 'cushion', 'cushions', 'cutter', 'cutting', 'dart', 'decoration', 'desk', 'detector', 'detergent', 'diaper', 'dining', 'dinosaur', 'dish', 'dishwasher', 'dishwashing', 'dispenser', 'display', 'divider', 'doll', 'dollhouse', 'dolly', 'door', 'doorframe', 'doors', 'drawer', 'dress', 'dresser', 'drum', 'dryer', 'dryers', 'drying', 'duffel', 'dumbbell', 'dustpan', 'easel', 'electric', 'elevator', 'elliptical', 'end', 'envelope', 'eraser', 'exercise', 'exit', 'extinguisher', 'fan', 'farthest', 'faucet', 'file', 'fire', 'fireplace', 'flag', 'flip', 'floor', 'flops', 'flower', 'flowerpot', 'folded', 'folder', 'food', 'foosball', 'footrest', 'footstool', 'fountain', 'frame', 'fridge', 'front', 'frying', 'function', 'furnace', 'furniture', 'fuse', 'futon', 'garage', 'garbage', 'glass', 'globe', 'gold', 'golf', 'grab', 'green', 'grey', 'grocery', 'guitar', 'gun', 'hair', 'hamper', 'hand', 'handicap', 'handrail', 'hanger', 'hangers', 'hanging', 'hat', 'hatrack', 'head', 'headboard', 'headphones', 'heater', 'height', 'helmet', 'holder', 'hood', 'hose', 'hoverboard', 'humidifier', 'identity', 'ikea', 'instrument', 'ipad', 'iron', 'ironing', 'island', 'jacket', 'jar', 'kettle', 'keyboard', 'kitchen', 'kitchenaid', 'knife', 'ladder', 'lamp', 'laptop', 'large', 'laundry', 'ledge', 'leftmost', 'legs', 'length', 'light', 'lock', 'loft', 'long', 'loofa', 'lower', 'luggage', 'lunch', 'machine', 'machines', 'magazine', 'mail', 'mailbox', 'mailboxes', 'maker', 'map', 'massage', 'mat', 'mattress', 'medal', 'messenger', 'metronome', 'microwave', 'middle', 'mini', 'mirror', 'mitt', 'mixer', 'mobile', 'monitor', 'mouse', 'mouthwash', 'mug', 'music', 'nerf', 'night', 'nightstand', 'notepad', 'object', 'office', 'open', 'orange', 'organizer', 'orientation', 'ottoman', 'outlet', 'oven', 'package', 'pad', 'painting', 'pan', 'panel', 'pantry', 'pants', 'paper', 'papers', 'person', 'photo', 'piano', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'ping', 'pink', 'pipe', 'pipes', 'pitcher', 'pizza', 'plant', 'plastic', 'plate', 'plates', 'plunger', 'podium', 'pool', 'position', 'poster', 'pot', 'potted', 'power', 'printer', 'products', 'projector', 'pump', 'purple', 'purse', 'quadcopter', 'rack', 'radiator', 'rail', 'railing', 'range', 'recliner', 'recycling', 'red', 'refrigerator', 'relations', 'remote', 'rice', 'rightmost', 'rod', 'roll', 'rolled', 'rolls', 'roomba', 'rope', 'round', 'rug', 'salt', 'santa', 'scale', 'scanner', 'screen', 'seat', 'seating', 'set', 'sewing', 'shampoo', 'sheets', 'shelf', 'shirt', 'shoe', 'shoes', 'shopping', 'short', 'shorts', 'shower', 'shredder', 'sign', 'sink', 'size', 'sliding', 'slippers', 'sliver', 'small', 'smoke', 'soap', 'sock', 'soda', 'sofa', 'speaker', 'sponge', 'spray', 'stack', 'stair', 'staircase', 'stairs', 'stall', 'stand', 'stapler', 'starbucks', 'station', 'statue', 'step', 'sticker', 'stool', 'storage', 'stove', 'stream', 'strip', 'structure', 'studio', 'stuffed', 'suitcase', 'suitcases', 'support', 'supported', 'sweater', 'swiffer', 'switch', 'table', 'tall', 'tank', 'tap', 'tape', 'tea', 'teapot', 'teddy', 'telephone', 'telescope', 'thermostat', 'tire', 'tissue', 'toaster', 'toilet', 'toiletry', 'toolbox', 'toothbrush', 'toothpaste', 'top', 'towel', 'towels', 'tower', 'toy', 'traffic', 'trash', 'tray', 'treadmill', 'tripod', 'trolley', 'trunk', 'tube', 'tupperware', 'tv', 'umbrella', 'urinal', 'vacuum', 'valve', 'vanity', 'vase', 'vending', 'vent', 'wall', 'walls', 'wardrobe', 'washcloth', 'washing', 'water', 'wet', 'wheel', 'white', 'whiteboard', 'window', 'windowsill', 'wood', 'workbench', 'yellow', 'yoga', 'ccccccc']



if __name__ == '__main__':

    def log_train_test_information():
        """Helper logging function.
        Note uses "global" variables defined below.
        """
        logger.info('Epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            if phase == 'train':
                meters = train_meters
            else:
                meters = test_meters

            info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(phase,
                                                                        meters[phase + '_total_loss'],
                                                                        meters[phase + '_referential_acc'])
            if args.obj_cls_alpha > 0:
                info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

            if args.target_cls_alpha > 0:
                info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_target_cls_acc'])

            if args.anchor_cls_alpha > 0:
                info += ', Anchor-Clf-Acc: {:.4f}'.format(meters[phase + '_anchor_cls_acc'])       

            if args.relation_cls_alpha > 0:
                info += ', Relation-Clf-Acc: {:.4f}'.format(meters[phase + '_relation_cls_acc'])        

            logger.info(info)
            logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
        logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))

    def get_group_parameters(model):
        params = list(model.named_parameters())
        for n,p in params:
            if 'clf' in n:
                print(n)
        param_group = [
            {'params':[p for n,p in params if 'clf' in n],'lr':5e-4},
            # {'params':[p for n,p in params if 'target_clf' in n],'lr':1e-3},
            # {'params':[p for n,p in params if 'relation_clf' in n],'lr':1e-3},
            # {'params':[p for n,p in params if 'anchor_clf' in n],'lr':1e-3},
            {'params':[p for n,p in params if 'clf' not in n ], 'lr':5e-4}
        ]
        return param_group

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.current_device()
    torch.cuda._initialized = True


    # Parse arguments
    args = parse_arguments()

    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)

    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)

    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)

    # Prepare GPU environment
    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    seed_training_code(args.random_seed)

    # Losses:
    criteria = dict()

    # Referential, "find the object in the scan" loss
    if args.s_vs_n_weight is not None:  # TODO - move to a better place
        assert args.augment_with_sr3d is not None
        ce = nn.CrossEntropyLoss(reduction='none').to(device)
        s_vs_n_weight = args.s_vs_n_weight


        def weighted_ce(logits, batch):
            loss_per_example = ce(logits, batch['target_pos'])
            sr3d_mask = ~batch['is_nr3d']
            loss_per_example[sr3d_mask] *= s_vs_n_weight
            loss = loss_per_example.sum() / len(loss_per_example)
            return loss


        criteria['logits'] = weighted_ce
    else:
        criteria['logits'] = nn.CrossEntropyLoss().to(device)

    # Object-type classification
    if args.obj_cls_alpha > 0:
        criteria['class_logits'] = nn.CrossEntropyLoss(ignore_index=class_to_idx['pad']).to(device)

    # # Target-in-language guessing
    if args.target_cls_alpha > 0:
        criteria['target_logits'] = nn.CrossEntropyLoss().to(device)

    if args.anchor_cls_alpha > 0:
        criteria['anchor_logits'] = nn.CrossEntropyLoss().to(device)


    if args.self_supervision_alpha > 0:
        print('Adding a self-supervised loss.')
        criteria['self_sv_logits'] = My_Loss().to(device)
        
    if args.relation_cls_alpha > 0:
        print("Adding language-relation-pred loss.")
        criteria['relation_logits'] = nn.CrossEntropyLoss().to(device)
        
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']
    model = create_r2g_net(args, vocab, n_classes, class_to_idx).to(device)
    
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
        # print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))


    # if args.resume_path:
    #     group_parameters = get_group_parameters(model)
    #     optimizer = optim.Adam(group_parameters, lr=args.init_lr)
    #     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
    #                                                             patience=5, verbose=True)        
    # else:
    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
    #                                                         patience=5, verbose=True)
    # group_parameters = get_group_parameters(model)
    # optimizer = optim.Adam(group_parameters, lr=args.init_lr)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
    #                                                                 patience=5, verbose=True)  

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
                                                              patience=5, verbose=True)

    print("----------------------------------")
    print("model's parameters:{}".format(np.sum([p.numel() for p in model.parameters()]).item()))
    print("----------------------------------")

    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    no_improvement = 0

    if args.resume_path or args.obj_cls_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, args.obj_cls_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not args.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, args.obj_cls_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            # start_training_epoch = loaded_epoch + 1
            # best_test_epoch = loaded_epoch
            # best_test_acc = lr_scheduler.best
            # print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
            #     best_test_acc))    
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = False
            for name, param in model.named_parameters():
                # if "object" in name:
                #     param.requires_grad = False
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))

    # Training.
    if args.mode == 'train':
        train_vis = Visualizer(args.tensorboard_dir)
        logger = create_logger(args.log_dir)
        logger.info('Starting the training. Good luck!')
        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                # Train:
                tic = time.time()
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, args=args)
                toc = time.time()
                timings['train'] = (toc - tic) / 60

                # Evaluate:
                tic = time.time()
                test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
                toc = time.time()
                timings['test'] = (toc - tic) / 60

                eval_acc = test_meters['test_referential_acc']
                lr_scheduler.step(eval_acc)

                if best_test_acc < eval_acc:
                    logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch

                    # Save the model (overwrite the best one)
                    save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red'))

                log_train_test_information()
                train_meters.update(test_meters)
                # train_vis.log_scalars({k: v for k, v in train_meters.items() if '_acc' in k}, step=epoch,
                #                       main_tag='acc')
                # train_vis.log_scalars({k: v for k, v in train_meters.items() if '_loss' in k},
                #                       step=epoch, main_tag='loss')

                ## update the log
                for k,v in train_meters.items():
                    train_vis.log_scalar(tag = k, scalar_value = v, epoch = epoch)

                bar.refresh()

                if no_improvement == args.patience:
                    logger.warning(colored('Stopping the training @epoch-{} due to lack of progress in test-accuracy '
                                           'boost (patience hit {} epochs)'.format(epoch, args.patience),
                                           'red', attrs=['bold', 'underline']))
                    break

        with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
            msg = ('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch))
            f_out.write(msg)

        logger.info('Finished training successfully. Good job!')

    elif args.mode == 'evaluate':
        # meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
        # print('Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
        # print('Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
        # exit()

        # out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
        # res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
        #                          args, out_file=out_file)
        # print(res)
        

        def collate_my(batch_data):
            batch_data.sort(key= lambda data: len(data['tokens']), reverse=True)
            out = {}
            for key in batch_data[0].keys():
                out[key] = [x[key] for x in batch_data]

            for key in out.keys():
                if key in ['object_mask', 'object_diag_mask', 'edge_attr', 'gt_class', 'tb_attr', 'mc_attr']:
                    out[key] = torch.stack(out[key])
                elif key in ['lang_mask', 'tokens', 'token_embedding']:
                    out[key] = pad_sequence(out[key], batch_first=True)
                elif key in ['context_size']:
                    out[key] = torch.Tensor(np.array(out[key])).int()
                elif key in ['target_pos', 'class_labels']:
                    out[key] = torch.LongTensor(np.array(out[key]))
                elif key in ['utterance', 'stimulus_id', 'scan_id']:
                    continue
                else:
                    out[key] = torch.Tensor(np.array(out[key]))

            return out


        # #prepare for 3d visual 
        references = data_loaders['test'].dataset.references
        d_loader = dataset_to_dataloader(data_loaders['test'].dataset, 'test', args.batch_size, n_workers=5, seed=2020)
        assert d_loader.dataset.references is references
        vis_res = save_predictions_for_visualization(model, d_loader, device, channel_last=True, seed=2020)
        anchor_right = 0
        pred_right = 0
        print("get the scenes for visualization")

        for i_index in range(len(vis_res)): ##i_index: per utturance index
            obj_class = vis_res[i_index]['class_label']
            obj_pred_index = np.argsort(vis_res[i_index]['obj_prob'], axis = -1)
        #     # if vis_res[i_index]['utterance'] == "the cabinets that are far away from the recycling bin":
        #     #     print(i_index)
        #     #     print("____________")
            prob = vis_res[i_index]['prob']
        #     if prob[1, vis_res[i_index]['anchor_pos']] > 0.9:
        #         anchor_right = anchor_right + 1
        #     if obj_class[vis_res[i_index]['anchor_pos']] ==  obj_pred_index[vis_res[i_index]['anchor_pos']][-1]:
        #         pred_right = pred_right + 1
        #     continue
        # print("**************")
        # print(anchor_right / len(vis_res), '\n', len(vis_res), '\n', pred_right / len(vis_res))

        # print("**************")
            if not (prob[1, vis_res[i_index]['anchor_pos']] < 0.9 and obj_class[vis_res[i_index]['anchor_pos']] ==  obj_pred_index[vis_res[i_index]['anchor_pos']][-1]):
                continue
            obj_class = vis_res[i_index]['class_label']
            out = {}
            # get scan
            scan = data_loaders['test'].dataset.scans[vis_res[i_index]['scan_id']]
            ppos = vis_res[i_index]['predicted_target_pos']
            pos = vis_res[i_index]['target_pos']
            # edge = vis_res[i_index]['edge_prob']

            out['scene_size'] = vis_res[i_index]['scene_size'].tolist()
            out['scene_center'] = vis_res[i_index]['scene_center'].tolist()
            # edges = ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported', 'between', 'allocentric']
            # edge = {}
            # for i, _edge in enumerate(edges):
            #     edge[_edge] = vis_res[i_index]['edge_prob'][ppos, pos][i].tolist()

            # out['edge'] = edge
            # if np.sum(vis_res[i_index]['edge_prob'][ppos, pos]) <= 1:
            #     continue


            # ins_simi = {}
            # token_simi = {}
            # for i in range(len(vis_res[i_index]['utterance'].split())):
            #     t = {}
            #     for j in range(20):
            #         t[vocabs[vis_res[i_index]['attention_index'][i][j]]] = vis_res[i_index]['attention_data'][i][j].tolist()
            #     token_simi[vis_res[i_index]['utterance'].split()[i]] = t
            # out['token_simi'] = token_simi

            # for j in range(5):
            #     out_ins_attention = {}
            #     for i in range(len(vis_res[i_index]['utterance'].split())):
            #         out_ins_attention[vis_res[i_index]['utterance'].split()[i]] = vis_res[i_index]['attention'][j][i].tolist()

            #     ins_simi[str(j)+' simi'] = out_ins_attention
            #     ins_simi[str(j)+' dist'] = vis_res[i_index]['instruction_prop'][j].tolist()

            # out['ins'] = ins_simi



            
            # print(ppos, pos)
            prob = vis_res[i_index]['prob']
            
            # pos_prob = vis_res[i_index]['confidences']
            
            # anchor_pos = vis_res[i_index]['anchor_pos']
            
            id_to_class = {v:k for k, v in class_to_idx.items()}
            # out = {}
            # out['class_label'] = id_to_class[int(vis_res[i_index]['target_class'])]
            # out['anchor_class'] = id_to_class[int(vis_res[i_index]['anchor_class'])]
            # out['sr_type'] = edges[int(vis_res[i_index]['sr_type'])]
            
            # anchor_tar_edge = vis_res[i_index]['edge_prob'][pos][anchor_pos]
            
            objects = {}
            obj_pred_index = np.argsort(vis_res[i_index]['obj_prob'], axis = -1)
            for i in range(vis_res[i_index]['context_size']):
                object_info = {}
                object_info["attention"] = prob[:,i].tolist()
                # simi = {}
                # for j in range(20):
                #     simi[str(j) + vocabs[vis_res[i_index]['simi_index'][i][j]]] = vis_res[i_index]['simi'][i][j].tolist()
                   
                # object_info["simi"] = simi
                # for j in range(9):
                #     object_info[id_to_class[int(vis_res[i_index]['class_index'][i][j])]] = vis_res[i_index]['objects_pred'][i][j].tolist()
                # attention['id:' + str(i) + ' gt:' + id_to_class[obj_class[i]] + ' pred:' +  id_to_class[obj_predclass[i]]] = prob[:,i].tolist()
                

                for j in range(3):
                    pred_class = {
                                    id_to_class[obj_pred_index[j][-1]]: (obj_pred_index[j][-1]).tolist(),
                                    id_to_class[obj_pred_index[j][-2]]: (obj_pred_index[j][-2]).tolist(),
                                    id_to_class[obj_pred_index[j][-3]]: (obj_pred_index[j][-3]).tolist(),
                                    }
                
                object_info['pred_class'] = pred_class
                objects['id:' + str(i) + ' gt:' + id_to_class[obj_class[i]]] = object_info
            
            # ins_simi = {}
            # print(vis_res[i_index]['ins_simi_index'].shape, vis_res[i_index]['ins_simi_index'])

            # for i in range(5):
            #     ins1= {}
            #     for j in range(20):
            #         ins1[vocabs[vis_res[i_index]['ins_simi_index'][i, j]]] = vis_res[i_index]['ins_simi'][i, j].tolist()
            #     ins_simi[i] = ins1
                # if object_class2[vis_res[i_index]['ins_simi_index'][0, j]] not in ins1.keys():
                # ins1[object_class1[vis_res[i_index]['ins_simi_index'][0, j]]] = vis_res[i_index]['ins_simi'][0, j].tolist()
                # # else:
                #     # ins1[object_class2[vis_res[i_index]['ins_simi_index'][0, j]]] = ins1[object_class2[vis_res[i_index]['ins_simi_index'][0, j]]] + vis_res[i_index]['ins_simi'][0, j].tolist()
                # # ins2[object_class2[vis_res[i_index]['ins_simi_index'][1, j]]] = vis_res[i_index]['ins_simi'][1, j].tolist()
                # # if object_class2[vis_res[i_index]['ins_simi_index'][2, j]] not in ins3.keys():
                # ins3[object_class1[vis_res[i_index]['ins_simi_index'][2, j]]] = vis_res[i_index]['ins_simi'][2, j].tolist()
                # else: 
                    # ins3[object_class2[vis_res[i_index]['ins_simi_index'][2, j]]] = ins3[object_class2[vis_res[i_index]['ins_simi_index'][2, j]]] + vis_res[i_index]['ins_simi'][2, j].tolist()
                    
            # ins_simi['ins1'] = ins1
            # ins_simi['ins1_em'] = vis_res[i_index]['intruction'][0].tolist()
            # # ins_simi['ins2'] = ins2
            # ins_simi['ins3'] = ins3
            # ins_simi['ins3_em'] = vis_res[i_index]['intruction'][2].tolist()
            
            # out['ins_simi'] = ins_simi
            
            out['object_info'] =objects

            #pred_class
            # print(type(vis_res[i_index]['obj_prob'][atten_obj[0]]), prob, atten_obj, np.shape(vis_res[i_index]['obj_prob'][atten_obj[0]]), np.shape(vis_res[i_index]['obj_prob']))


            ## SR
            # relation_semantic = ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported', 'between', 'allocentric']
            
            # t_a_sr = dict(zip(relation_semantic, anchor_tar_edge.tolist()))
            # out['target_anchor_relation'] = t_a_sr
            
            # id_to_class = {v:k for k, v in class_to_idx.items()}
            # obj_class = vis_res[i_index]['class_label']
            # t_obj = {}
            # # p_obj = {}
            # t_obj['target class'] = id_to_class[obj_class[pos]]
            # p_obj['ptarget class'] = id_to_class[obj_class[ppos]]
            # attr = ['large', 'small', 'tall', 'lower', 'middle', 'corner', 'top', 'bottom', 'leftmost', 'rightmost', 'long', 'short']
            # obj_attr = vis_res[i_index]['obj_attr']
            
            # for i in range(len(attr)):
            #     t_obj[attr[i]] = obj_attr[pos][i].tolist()
            #     p_obj[attr[i]] = obj_attr[ppos][i].tolist()

            # t_obj['center'] = vis_res[i_index]['obj_center'][pos].tolist()
            # t_obj['size'] = vis_res[i_index]['obj_size'][pos].tolist()
            # p_obj['center'] = vis_res[i_index]['obj_center'][ppos].tolist()
            # p_obj['size'] = vis_res[i_index]['obj_size'][ppos].tolist()
            
            # out['tobj'] = t_obj
            # out['pobj'] = p_obj


    
            # attention = {}
            # for i in range(vis_res[i_index]['context_size']):
            #     attention['gt:' + id_to_class[obj_class[i]] + ' pred:' +  id_to_class[obj_predclass[i]]] = prob[:,i].tolist()
                
            # out['attention'] =attention 
            
        
            
            # pred_sr = {}
            # sr = vis_res[i_index]['sr_prob'][:9, :9]
            # pred_sr['tar-anc relation'] = { vis_res[i_index]['sr_type'].tolist(): vis_res[i_index]['tar_anc_sr'].tolist()} 
            # for i in range(9):
            #     pred_sr['{}'.format(i)] = sr[i].tolist()
                
 
            # top_id = vis_res[i_index]['object_ids'][atten_obj].int().cpu().numpy()

            # use id to identify the object
            p_id = vis_res[i_index]['object_ids'][ppos].int().cpu().numpy()
            t_id = vis_res[i_index]['object_ids'][pos].int().cpu().numpy()
            # anchor_id = vis_res[i_index]['object_ids'][anchor_pos].int().cpu().numpy()

            # SR
            # p_id = vis_res[i_index]['object_ids'][:9].int().cpu().numpy()
            # t_id = vis_res[i_index]['object_ids'][pos].int().cpu().numpy()
            
            
            

            if not vis_res[i_index]['correct']:
                # pos_id = vis_res[i_index]['object_ids']
                director = '/data1/liyixuan/R2G/vis/{}_{}_{}'.format(vis_res[i_index]['correct'],vis_res[i_index]['scan_id'],vis_res[i_index]['utterance'].replace("/", "or"))
                scan.visualize_heatmap(prob, pid= p_id, id = t_id, pos_id = None, filedir = director, utterance = vis_res[i_index]['utterance'].replace("/", "or"))
                with open(director + '/{}.json'.format(vis_res[i_index]['utterance'].replace("/", "or")), 'w') as file:
                    json.dump(out, file, indent = 4)
                # print("***************")
                # scan.visualize_ground(p_id, t_id, '/data1/liyixuan/referit_my/vis/{}_{}_{}.ply'.format(vis_res[i_index]['correct'],vis_res[i_index]['scan_id'],vis_res[i_index]['utterance'].replace("/", "or")))