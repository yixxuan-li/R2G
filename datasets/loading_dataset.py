import numpy as np
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers
import torch

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform, decode_stimulus_string
from .utils import get_allocentric_relation
from referit3d.utils import unpickle_data
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
                 visualization=False, args = None, scan_relation = None):

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
        self.scan_relation = scan_relation
        # self.embedder = token_embeder(vocab=vocab, word_embedding_dim=args.word_embedding_dim, random_seed=args.random_seed)
        
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
        tokens, lang_mask = self.vocab.encode(ref['tokens'],self.max_seq_len)
        tokens_len = len(ref['tokens'])
        tokens_filterd, tokens_filterd_mask =  self.vocab.encode(raw_token_filtered, self.max_seq_len, add_begin_end=False)
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

        instruction_tokens = None
        instructions_mask = None

        if 'instruction' in ref.keys():
            instructions = []
            for k, v in eval(ref['instruction']).items():
                instructions.append(v.lower())
            instructions.reverse()
            instruction_tokens, _ = self.vocab.encode(instructions, add_begin_end=False)
            instruction_tokens = np.array(instruction_tokens)
            instructions_mask = np.array([1, 1, 1])
        # if 'instruction' in ref.keys():
        #     instructions = []
        #     instructions_mask = []
            # if "}}}}" in ref['instruction']:
            #     if ref['instruction'].split('.')[0].split('AND')[0].split('OR')[0].split('(Note')[0].split('Note')[0].split('}}}}')[0][-1] != "}":
            #         ins = eval(str(ref['instruction']).split('AND')[0].split('(Note')[0].split('Note')[0].split('}}}}')[0]+"}}}}")
            # elif "}}}" in ref['instruction']:
            #     if ref['instruction'].split('.')[0].split('AND')[0].split('OR')[0].split('(Note')[0].split('Note')[0].split('}}}')[0][-1] != "}":
            #         ins =eval(str(ref['instruction']).split('AND')[0].split('(Note')[0].split('Note')[0].split('}}}')[0]+"}}}")
            # else:
            #     ins =eval(str(ref['instruction']).split('AND')[0].split('OR')[0].split('(Note')[0].split('Note')[0].split('}}}')[0])
            # if not isinstance(ins, dict):
            #     ins = ins[0]
            # keys = ['anchor', 'relation', 'target']
            # sub_keys = ['identity', 'attribute']
            # # print(ins)
            # for key in keys:
            #     if key != 'relation':
            #         if key in ins.keys() and isinstance(ins[key], dict):
            #             if isinstance(ins[key][sub_keys[0]], dict):
            #                 instructions.append(str(ins[key][sub_keys[0]].values())[0].split()[0].split('-')[0].split('_')[0].lower())
            #                 instructions_mask.append(int(instructions[-1]!='null'))
            #             elif sub_keys[0] in ins[key].keys() and ins[key][sub_keys[0]] is not None and ins[key][sub_keys[0]] != '':
            #                 instructions.append(ins[key][sub_keys[0]].split()[0].split('-')[0].split('_')[0].lower())
            #                 instructions_mask.append(int(instructions[-1]!='null'))
            #             else:
            #                 instructions.append('null')
            #                 instructions_mask.append(0)       
            #             try:        
            #                 # if isinstance(ins[key], dict):
            #                 if sub_keys[1] in ins[key].keys() and ins[key][sub_keys[0]] is not None and  ins[key][sub_keys[0]] != '' and  ins[key][sub_keys[0]] != 'null' and isinstance(ins[key][sub_keys[0]], dict):
            #                     if  '0' in ins[key][sub_keys[1]].keys() and isinstance(ins[key][sub_keys[1]]['0'], dict):
            #                         instructions.append(str(ins[key][sub_keys[1]]['0'].values())[0].split()[0].split('-')[0].split('_')[0].lower())
            #                         instructions_mask.append(int(instructions[-1]!='null'))
            #                     elif '0' in ins[key][sub_keys[1]].keys() and ins[key][sub_keys[1]]['0'] is not None and ins[key][sub_keys[1]]['0'] != '':
            #                         instructions.append(ins[key][sub_keys[1]]['0'].split()[0].split('-')[0].split('_')[0].lower())
            #                         instructions_mask.append(int(instructions[-1]!='null'))
            #                     else:
            #                         instructions.append('null')
            #                         instructions_mask.append(0)

            #                     if '1' in ins[key][sub_keys[1]].keys() and isinstance(ins[key][sub_keys[1]]['1'], dict):
            #                         instructions.append(str(ins[key][sub_keys[1]]['1'].values())[0].split()[0].split('-')[0].split('_')[0].lower())
            #                         instructions_mask.append(int(instructions[-1]!='null'))
            #                     elif '1' in ins[key][sub_keys[1]].keys() and ins[key][sub_keys[1]]['1'] is not None and ins[key][sub_keys[1]]['1'] != '':
            #                         instructions.append(ins[key][sub_keys[1]]['1'].split()[0].split('-')[0].split('_')[0].lower())
            #                         instructions_mask.append(int(instructions[-1]!='null'))
            #                     else:
            #                         instructions.append('null')
            #                         instructions_mask.append(0)
            #                 else:
            #                     for i in range(2):
            #                         instructions.append('null')
            #                         instructions_mask.append(0)
            #             except:
            #                 print(type(ins[key]), ins[key])
            #         else:
            #             for i in range(3):
            #                 instructions.append('null')
            #                 instructions_mask.append(0)
            #     else:
            #         if key in ins.keys() and ins[key] is not None and ins[key] != '' :
            #             if isinstance(ins[key], dict):
            #                 instructions.append(str(ins[key].values()).split()[0].lower())
            #             elif len(ins[key].split()) == 3:
            #                 instructions.append(ins[key].split()[1].lower())
            #             else:
            #                 instructions.append(ins[key].split()[0].lower())
            #             instructions_mask.append(int(instructions[-1]!='null'))
            #         else:
            #             instructions.append('null')
            #             instructions_mask.append(0)
            # instruction_tokens, _ = self.vocab.encode(instructions, add_begin_end=False)
            # instruction_tokens = np.array(instruction_tokens)
            # instructions_mask = np.array(instructions_mask)


        
            

        return scan, target, tokens, tokens_len, is_nr3d, lang_mask, tokens_filterd, tokens_filterd_mask, anchor, sr_type, ref['target_id'], anchors_id, instruction_tokens, instructions_mask, self.scan_relation[ref['scan_id']]

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
        distractors_ind = []
        clutter_ind = []
        for ind, o in enumerate(scan.three_d_objects):
            # First add all objects with the same instance-label as the target
            if (o.instance_label == target_label and (o != target)):
                distractors.append(o)
                distractors_ind.append(ind)
            # and all more objects up to max-number of distractors   
            elif o != target :  
                if anchor is not None:
                    if  (o != anchor):
                        clutter.append(o)
                        clutter_ind.append(ind)
                else:
                        clutter.append(o)
                        clutter_ind.append(ind)


        state = np.random.get_state()
        np.random.shuffle(clutter)
        np.random.set_state(state)
        np.random.shuffle(clutter_ind)

        distractors.extend(clutter)
        distractors_ind.extend(clutter_ind)
        if anchor is not None:
            distractors = distractors[:self.max_distractors-1]
            distractors_ind = distractors_ind[:self.max_distractors-1]
        else:
            distractors = distractors[:self.max_distractors]
            distractors_ind = distractors_ind[:self.max_distractors]

        state = np.random.get_state()
        np.random.shuffle(distractors)
        np.random.set_state(state)
        np.random.shuffle(distractors_ind)
      

        return distractors, distractors_ind

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, tokens_len, is_nr3d, lang_mask, tokens_filterd, tokens_filterd_mask, anchor, sr_type, target_id, anchor_id, instruction_tokens, instructions_mask, scan_relation = self.get_reference_data(index)
        # Make a context of distractors
        context, context_ind_of_scan = self.prepare_distractors(scan, target, anchor)
        # print(scan.relation_matrix)


        # Add anchor object in 'context' list
        anchor_pos = None
        if anchor is not None:
            anchor_pos = np.random.randint(len(context) + 1)
            context.insert(anchor_pos, anchor)
            context_ind_of_scan.insert(anchor_pos, anchor_id)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        if anchor_pos is not None:
            if target_pos <= anchor_pos:
                anchor_pos += 1
        context.insert(target_pos, target)
        context_ind_of_scan.insert(target_pos, target_id)

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
        # res['token_embedding'] = self.embedder(torch.LongTensor(tokens))
        
        res['edge_vector'] = np.zeros((self.max_context_size, self.max_context_size, 3), dtype=np.float32)
        context_ind_of_scan = np.array([np.where(scan_relation['obj_id'][0] == o.object_id)[0][0] for o in context])
        relation_matrix = scan_relation['rela_dis'][0][context_ind_of_scan, :, :]
        relation_matrix = relation_matrix[:, context_ind_of_scan, :]

        # ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported', 'between', 'allocentric']
        res['edge_distance'] = np.zeros((self.max_context_size, self.max_context_size, 1), dtype=np.float32)
        res['edge_touch'] = np.zeros((self.max_context_size, self.max_context_size, 1), dtype=np.float32)
        res['edge_attr'] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).view(1,1,-1).repeat(self.max_context_size, self.max_context_size, 1).float()
        res['edge_attr'][:res['context_size'], :res['context_size'], :-1] = torch.tensor(relation_matrix)

        # # model the top and bottom
        res['tb_attr'] = torch.zeros([self.max_context_size, 2])
        if self.args.model_attr or self.args.multi_attr:
            res['tb_attr'][:res['context_size']] = torch.tensor(scan.tb_attr[context_ind_of_scan,:])
        # res['tb_attr'][:len(context), 1] = 2
        # res['tb_attr'][:len(context), 0] = 1
        
        # # model middle or corner
        res['mc_attr'] = torch.zeros([self.max_context_size, 2])
        if self.args.model_attr or self.args.multi_attr:
            res['mc_attr'][:res['context_size']] = torch.tensor(scan.mc_attr[context_ind_of_scan,:])


        # for j, o in enumerate(context):
        #     if context[j].has_front_direction:
        #         for i in range(0, len(context)):  
        #             allo_relation = get_allocentric_relation(context[j], context[i]) 
        #             if sr_type in ['front', 'back', 'left', 'right']:
        #                 if i == target_pos and j == anchor_pos:
        #                     print(sr_type, allo_relation)
        #             if allo_relation == 0:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        #             elif allo_relation == 2:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])   
        #             elif allo_relation == 1:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        #             elif allo_relation == 3:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])



        for i, o in enumerate(context):
            i_size = res['obj_size'][i]
            i_position = res['obj_position'][i]
            i_scene_translation = i_position - res['scene_center'] 
        #     # model middle or corner
        #     if np.abs(i_scene_translation[0])  < i_size[0] and np.abs(i_scene_translation[1]) < i_size[1]:
        #         res['mc_attr'][i, 0] = 1
        #     if np.abs(res['scene_size'][0]/2 - np.abs(i_scene_translation[0])) < i_size[0] and np.abs(res['scene_size'][1]/2 - np.abs(i_scene_translation[1])) < i_size[1]:
        #         res['mc_attr'][i, 1] = 1
        #     # for j in range(i+1, len(context)):  
            for j in range(0, len(context)):
                if i == j:
                    continue
        #         if context[j].has_front_direction:
        #             allo_relation = get_allocentric_relation(context[j], context[i]) 
        #             if allo_relation == 0:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        #             elif allo_relation == 2:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])   
        #             elif allo_relation == 1:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        #             elif allo_relation == 3:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                if i < j:
                    j_size = res['obj_size'][j]  #[lx_,l_y,l_z]
                    j_position = res['obj_position'][j]
                    res['edge_vector'][i][j] = i_position - j_position
                    res['edge_vector'][j][i] = - res['edge_vector'][i][j]
                    res['edge_distance'][i][j] = np.sqrt(np.sum(np.square(res['edge_vector'][i][j]), axis = 0))
                    res['edge_distance'][j][i] = res['edge_distance'][i][j]
        #             # if res['edge_distance'][i][j] < context[i].get_object_radius() + context[j].get_object_radius():
        #             #     # support supported
        #             #     if (np.abs(res['edge_vector'][i][j][1])*2 < j_size[1] or np.abs(res['edge_vector'][i][j][1])*2 < i_size[1]) \
        #             #     and (np.abs(res['edge_vector'][i][j][0])*2 < j_size[0] or np.abs(res['edge_vector'][i][j][0])*2 < i_size[0]):
        #             #         # res['edge_touch'][i][j] = 1
        #             #         if res['edge_vector'][i][j][2] > 0:
        #             #             res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        #             #             res['edge_attr'][j][i] += torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        #             #         elif res['edge_vector'][i][j][2] < 0:
        #             #             res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        #             #             res['edge_attr'][j][i] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

        #             # above below
        #             target_extrema = context[i].get_axis_align_bbox().extrema
        #             target_zmin = target_extrema[2]
        #             target_zmax = target_extrema[5]
        #             anchor_extrema = context[j].get_axis_align_bbox().extrema
        #             anchor_zmin = anchor_extrema[2]
        #             anchor_zmax = anchor_extrema[5]
        #             target_bottom_anchor_top_dist = target_zmin - anchor_zmax
        #             target_top_anchor_bottom_dist = anchor_zmin - target_zmax
        #             iou_2d, i_ratios, a_ratios = context[i].iou_2d(context[j])
        #             i_target_ratio, i_anchor_ratio = i_ratios
        #             target_anchor_area_ratio, anchor_target_area_ratio = a_ratios
        #             # Above, Below 
        #             if target_bottom_anchor_top_dist > 0.06 and max(i_anchor_ratio, i_target_ratio) > 0.2:
        #                 res['edge_attr'][i][j] += torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #                 res['edge_attr'][j][i] += torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        #                 res['tb_attr'][i, 1] -= 1
        #                 res['tb_attr'][j, 0] = 0
        #             elif target_top_anchor_bottom_dist > 0.06 and max(i_anchor_ratio, i_target_ratio) > 0.2:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        #                 res['edge_attr'][j][i] += torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #                 res['tb_attr'][i, 0] = 0
        #                 res['tb_attr'][j, 1] -= 1
        #             # supported, support
        #             if i_target_ratio > 0.2 and abs(target_bottom_anchor_top_dist) <= 0.15 and target_anchor_area_ratio < 1.5:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        #                 res['edge_attr'][j][i] += torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        #             if i_anchor_ratio > 0.2 and abs(target_top_anchor_bottom_dist) <= 0.15 and anchor_target_area_ratio < 1.5:
        #                 res['edge_attr'][i][j] += torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        #                 res['edge_attr'][j][i] += torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

        #     if res['tb_attr'][i, 1] != 1:
        #         res['tb_attr'][i, 1] = 0
        #     if (res['tb_attr'][i, 0] == 1) and (res['tb_attr'][i, 1] == 1):
        #         res['tb_attr'][i, 0] = 0
        #         res['tb_attr'][i, 1] = 0





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
                'left': 8,
                'right': 9,
                'between': 10
                }
            if sr_type in relation.keys():
                sr = relation[sr_type]
            
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
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d

        if instruction_tokens is not None:
            res['ins_token'] = instruction_tokens
        if instructions_mask is not None:
            res['ins_mask'] = instructions_mask

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
    
    scan_relation = None
    if args.scan_relation_path is not None:
        file = unpickle_data(args.scan_relation_path)
        scan_relation = next(file)

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
                                   args = args,
                                   scan_relation = scan_relation)

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders