import numpy as np
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers
import torch

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform, decode_stimulus_string
from models import token_embeder
from torch.nn.utils.rnn import pad_sequence

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

class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, args = None):

        self.references = references
        self.scans = scans
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.embedder = token_embeder(vocab=vocab, word_embedding_dim=args.word_embedding_dim, random_seed=args.random_seed)
        
        self.args= args 

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        raw_token_filtered = []
        for token in ref['tokens']:
            # print(token)
            if token in self.vocab.word2idx.keys():
                raw_token_filtered.append(token)
        tokens, lang_mask = self.vocab.encode(ref['tokens'], None, add_begin_end=False)
        tokens_len = len(ref['tokens'])
        tokens_filterd, tokens_filterd_mask =  self.vocab.encode(raw_token_filtered, None, add_begin_end=False)
        tokens = np.array(tokens, dtype=np.long)

        lang_mask = np.array(lang_mask)
        tokens_filterd = np.array(tokens_filterd, dtype=np.long)
        tokens_filterd_mask = np.array(tokens_filterd_mask)
        is_nr3d = ref['dataset'] == 'nr3d'

        anchors_id = None
        anchor = None
        if 'anchor_ids' in ref.keys():
            anchors_id = int(ref['anchor_ids'].replace('[', '').replace(']', '').split(',')[0])
            anchor = scan.three_d_objects[anchors_id]
            

        sr_type = None
        if 'reference_type' in ref.keys():
            sr_type = ref['reference_type']
            

        return scan, target, tokens, tokens_len, is_nr3d, lang_mask, tokens_filterd, tokens_filterd_mask, anchor, sr_type

    def prepare_distractors(self, scan, target, anchor = None):
        target_label = target.instance_label

        # # First add all objects with the same instance-label as the target
        # distractors = [o for o in scan.three_d_objects if
        #                (o.instance_label == target_label and (o != target))]

        # # Then all more objects up to max-number of distractors
        # already_included = {target_label}
        # if anchor is not None:
        #     clutter = [o for o in scan.three_d_objects if ((o.instance_label not in already_included) and (o != anchor))]
        # else:
        #     clutter = [o for o in scan.three_d_objects if (o.instance_label not in already_included)]
            
            
        already_included = {target_label}    
        distractors = []
        clutter = []
        for o in scan.three_d_objects:
            # First add all objects with the same instance-label as the target
            if (o.instance_label == target_label and (o != target)):
                distractors.append(o)
            # and all more objects up to max-number of distractors   
            elif o != target :  
                if anchor is not None:
                    if  (o != anchor):
                        clutter.append(o)
                else:
                        clutter.append(o)

        np.random.shuffle(clutter)

        distractors.extend(clutter)
        if anchor is not None:
            distractors = distractors[:self.max_distractors-1]
        else:
            distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, tokens_len, is_nr3d, lang_mask, tokens_filterd, tokens_filterd_mask, anchor, sr_type = self.get_reference_data(index)
        # Make a context of distractors
        context = self.prepare_distractors(scan, target, anchor)


        # Add anchor object in 'context' list
        anchor_pos = None
        if anchor is not None:
            anchor_pos = np.random.randint(len(context) + 1)
            context.insert(anchor_pos, anchor)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        if anchor_pos is not None:
            if target_pos <= anchor_pos:
                anchor_pos += 1
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])

        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)
 
        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res['context_size'] = len(samples)

        res['scene_pc'] = np.array(scan.sample_scene_pc(n_samples = 1024))
        
        res['scene_center'], res['scene_size'] = scan.get_center_size()
        res['scene_center'] = np.array(res['scene_center'])
        res['scene_size'] = np.array(res['scene_size'])

        

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)
        
        res['object_mask'] = torch.zeros(self.max_context_size)
        res['object_mask'][len(context):] = torch.tensor(-np.inf)
        res['object_diag_mask'] = torch.ones((self.max_context_size, self.max_context_size))
        diag = torch.eye(len(context))
        res['object_diag_mask'][:len(context), :len(context)] = res['object_diag_mask'][:len(context), :len(context)] - diag

        # get the explicity node feature and padding it
        res['color_onehot'] = np.array([o.get_mean_color() for o in context]) ## use one-hot to represent color
        res['color_token'] = [np.array(self.vocab.encode(o.get_mean_color_token(), 1, add_begin_end = False)[0], dtype=np.long) for o in context]   # use token to represent color
        res['obj_size'] = np.array([o.get_size() for o in context]) #    [lx_,l_y,l_z]
        res['obj_position'] = np.array([o.get_center_position() for o in context])

        color_f_tmp = np.zeros((self.max_context_size, 13), dtype=np.float32)
        color_t_tmp = np.zeros((self.max_context_size, 1), dtype=np.long)
        size_tmp = np.zeros((self.max_context_size, 3), dtype=np.float32)
        position_tmp = np.zeros((self.max_context_size, 3), dtype=np.float32)
        color_f_tmp[:len(context)] = res['color_onehot']
        color_t_tmp[:len(context)] = res['color_token']
        size_tmp[:len(context)] = res['obj_size']
        position_tmp[:len(context)] = res['obj_position']

        res['color_onehot'] = color_f_tmp
        res['color_token'] = color_t_tmp
        res['obj_size'] = size_tmp
        res['obj_position'] = position_tmp
        res['token_embedding'] = self.embedder(torch.LongTensor(tokens))
        
        res['edge_vector'] = np.zeros((self.max_context_size, self.max_context_size, 3), dtype=np.float32)
        
        # ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported', 'between', 'allocentric']
        res['edge_distance'] = np.zeros((self.max_context_size, self.max_context_size, 1), dtype=np.float32)
        res['edge_touch'] = np.zeros((self.max_context_size, self.max_context_size, 1), dtype=np.float32)
        res['edge_attr'] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).view(1,1,-1).repeat(self.max_context_size, self.max_context_size, 1).float()
        # model the top and bottom
        res['tb_attr'] = torch.zeros([self.max_context_size, 2])
        res['tb_attr'][:len(context), 1] = 2
        res['tb_attr'][:len(context), 0] = 1
        # model middle or corner
        res['mc_attr'] = torch.zeros([self.max_context_size, 2])


        for i, o in enumerate(context):
            i_size = context[i].get_size()
            i_position = context[i].get_center_position()
            i_scene_translation = i_position - res['scene_center'] 
            # model middle or corner
            if np.abs(i_scene_translation[0])  < i_size[0] and np.abs(i_scene_translation[1]) < i_size[1]:
                res['mc_attr'][i, 0] = 1
            if np.abs(res['scene_size'][0]/2 - np.abs(i_scene_translation[0])) < i_size[0] and np.abs(res['scene_size'][1]/2 - np.abs(i_scene_translation[1])) < i_size[1]:
                res['mc_attr'][i, 1] = 1


            for j in range(i+1, len(context)):  
                j_size = context[j].get_size()  #[lx_,l_y,l_z]
                j_position = context[j].get_center_position()
                res['edge_vector'][i][j] = i_position - j_position
                res['edge_vector'][j][i] = - res['edge_vector'][i][j]
                if  True:#self.args is not None and not self.args.relation_pred:
                    res['edge_distance'][i][j] = np.sqrt(np.sum(np.square(res['edge_vector'][i][j]), axis = 0))
                    res['edge_distance'][j][i] = res['edge_distance'][i][j]
                    if res['edge_distance'][i][j] < context[i].get_object_radius() + context[j].get_object_radius():
                        # front back
                        if (np.abs(res['edge_vector'][i][j][1])*2 < j_size[1] or np.abs(res['edge_vector'][i][j][1])*2 < i_size[1]) \
                        and (np.abs(res['edge_vector'][i][j][2])*2 < j_size[2] or np.abs(res['edge_vector'][i][j][2])*2 < i_size[2]):
                            if res['edge_vector'][i][j][0] > 0:
                                res['edge_attr'][i][j] += torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
                                res['edge_attr'][j][i] += torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                            elif res['edge_vector'][i][j][0] < 0:
                                res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                                res['edge_attr'][j][i] += torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
                        # support supported
                        if (np.abs(res['edge_vector'][i][j][1])*2 < j_size[1] or np.abs(res['edge_vector'][i][j][1])*2 < i_size[1]) \
                        and (np.abs(res['edge_vector'][i][j][0])*2 < j_size[0] or np.abs(res['edge_vector'][i][j][0])*2 < i_size[0]):
                            # res['edge_touch'][i][j] = 1
                            if res['edge_vector'][i][j][2] > 0:
                                res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                                res['edge_attr'][j][i] += torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                            elif res['edge_vector'][i][j][2] < 0:
                                res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                                res['edge_attr'][j][i] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

                    # above below
                    if (np.abs(res['edge_vector'][i][j][1])*2 < j_size[1] or np.abs(res['edge_vector'][i][j][1])*2 < i_size[1]) \
                    and (np.abs(res['edge_vector'][i][j][0])*2 < j_size[0] or np.abs(res['edge_vector'][i][j][0])*2 < i_size[0]):
                        if res['edge_vector'][i][j][2] > 0:
                            res['edge_attr'][i][j] += torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            res['edge_attr'][j][i] += torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                            res['tb_attr'][i, 1] -= 1
                            res['tb_attr'][j, 0] = 0
                        elif res['edge_vector'][i][j][2] < 0:
                            res['edge_attr'][i][j] += torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                            res['edge_attr'][j][i] += torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            res['tb_attr'][i, 0] = 0
                            res['tb_attr'][j, 1] -= 1
            if res['tb_attr'][i, 1] != 1:
                res['tb_attr'][i, 1] = 0
            if (res['tb_attr'][i, 0] == 1) and (res['tb_attr'][i, 1] == 1):
                res['tb_attr'][i, 0] = 0
                res['tb_attr'][i, 1] = 0




        # model spatial relations in sr3d
        if sr_type is not None:
            ## get the spatial relation type
            relation = {
                'above': 0, 
                'below': 1, 
                'front': 2, 
                'back': 3, 
                'farthest': 4, 
                'closest': 5, 
                'supporting': 6, 
                'supported-by': 7, 
                'between': 8,
                'allocentric': 9
                }
            if sr_type in relation.keys():
                sr = relation[sr_type]
            else:
                sr = relation['allocentric']
            
            res['sr_type'] = sr

        res['lang_mask'] = torch.Tensor(lang_mask)
        res['tokens_len'] = tokens_len
        
        if anchor_pos is not None:
            res['anchors_pos'] = anchor_pos


        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        res['target_class'] = self.class_to_idx[target.instance_label]
        if anchor is not None:
            res['anchor_class'] = self.class_to_idx[anchor.instance_label]
        res['gt_class'] = torch.zeros([self.max_context_size, len(self.class_to_idx)])
        for i in range(res['context_size']):
            res['gt_class'][i, res['class_labels'][i]] = 1

        
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = torch.Tensor(tokens)
        res['is_nr3d'] = is_nr3d

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

            # get scan_id 
            res['scan_id'] = self.references.loc[index].scan_id

        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   args = args)

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed, collate_fn = collate_my)

    return data_loaders