import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np
from itertools import combinations
import sys
sys.path.append('/data1/liyixuan/referit_my/referit3d/models/backbone/example/build')
import relation
 
# # nr3d
# nr3dclass = ['table', 'alarm', 'armchair', 'backpack', 'bag', 'ball', 'banner', 'bar', 'basket', 'wall', 'cabinet', 'counter', 'door', 'bathtub', 'chair', 'bear', 'bed', 'bottles', 'bench', 'bicycle', 'bin', 'blackboard', 'blanket', 'blinds', 'board', 'boards', 'book', 'rack', 'books', 'bookshelf', 'bookshelves', 'bottle', 'bowl', 'box', 'boxes', 'bucket', 'doors', 'cabinets', 'calendar', 'camera', 'can', 'car', 'cardboard', 'carpet', 'cart', 'case', 'ceiling', 'fan', 'light', 'chest', 'clock', 'closet', 'floor', 'rod', 'shelf', 'cloth', 'clothes', 'clothing', 'coat', 'column', 'container', 'pot', 'copier', 'couch', 'cushion', 'crate', 'cup', 'cups', 'curtain', 'curtains', 'decoration', 'desk', 'lamp', 'dishwasher', 'dispenser', 'display', 'dolly', 'drawer', 'dresser', 'easel', 'machine', 'sign', 'faucet', 'fireplace', 'stand', 'ladder', 'folder', 'footrest', 'footstool', 'frame', 'furniture', 'futon', 'globe', 'guitar', 'hamper', 'rail', 'towel', 'hanging', 'hat', 'headboard', 'headphones', 'heater', 'jacket', 'keyboard', 'piano', 'laptop', 'ledge', 'legs', 'switch', 'luggage', 'magazine', 'mail', 'tray', 'map', 'mat', 'mattress', 'microwave', 'mirror', 'monitor', 'mouse', 'mug', 'nightstand', 'notepad', 'object', 'ottoman', 'oven', 'painting', 'paper', 'papers', 'person', 'photo', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'pipe', 'pipes', 'plant', 'poster', 'printer', 'projector', 'screen', 'purse', 'radiator', 'railing', 'refrigerator', 'roomba', 'rug', 'seat', 'seating', 'shampoo', 'shirt', 'shoe', 'shoes', 'shorts', 'shower', 'walls', 'sink', 'soap', 'stair', 'staircase', 'stairs', 'statue', 'step', 'stool', 'sticker', 'stove', 'suitcase', 'suitcases', 'sweater', 'tank', 'telephone', 'thermostat', 'toaster', 'toilet', 'towels', 'tv', 'umbrella', 'vase', 'vent', 'wardrobe', 'whiteboard', 'window', 'windowsill', 'wood']
nr_index = [[0, 116, 144, 173, 192, 197, 328, 342, 371, 446], [2, 3, 181], [4], [6], [7, 8, 165, 194, 207, 210, 212, 234, 255, 281, 311, 388, 479], [9, 27], [13], [14, 56, 211, 222], [16, 256], [18, 100, 101, 307, 308, 398, 499], [20, 62, 180, 245, 299, 481, 504], [21, 129, 247], [23, 63, 94, 156, 206, 394, 403], [25], [28, 87, 189, 190, 277, 298, 362, 413, 417, 419], [29, 452], [30, 61, 263, 412], [31], [32], [33], [36, 118, 143, 334, 337, 363, 432, 480], [37], [38], [39], [41, 60, 138, 139, 238], [42], [45, 291], [46, 109, 111, 112, 145, 151, 160, 164, 226, 266, 271, 356, 386, 474, 484, 502], [47], [48], [49], [51, 80, 120, 147, 289, 382, 407, 416, 509], [52], [53, 81, 113, 130, 204, 268, 433, 457], [54, 55, 332], [59], [64, 96, 158, 208, 286, 395], [65, 246], [67], [68], [69, 482], [72], [74], [75], [77], [79, 82, 150, 214, 235], [83, 93], [84, 178], [85, 261, 439], [90], [91], [92, 505], [97, 186, 396], [98, 367, 393], [99, 301, 306, 383, 436], [102], [103], [108], [110], [117], [121, 195, 335, 336, 434], [124, 347], [125], [127], [128, 137], [131], [133, 426], [134, 418], [135, 392], [136], [140], [141], [142, 252, 294, 501], [146], [148, 220, 315, 409, 464, 468], [149, 196], [155], [159], [161], [168], [172, 176, 269, 380, 497, 507, 508], [177, 401, 514], [179], [183], [187, 267, 292, 357, 424, 492], [191, 251], [193], [198], [199], [200], [203], [205], [209], [213], [217, 258], [219, 223, 359, 421], [221, 314, 316, 473], [224, 500], [225], [227], [228], [229, 512], [239], [242], [243, 321, 477], [254], [259], [260], [262, 445], [265], [270], [272], [273, 317, 483], [276], [278], [279], [283], [285], [287], [288], [290], [295], [296], [297], [302], [303, 459], [305], [310, 463], [318], [319], [320], [323], [324], [325], [326], [327], [329], [330], [333, 348], [343, 368], [345, 351], [352], [353, 377], [354], [358], [360], [364], [369], [372], [378], [379], [381], [384], [385], [387], [389], [390], [399], [402], [406], [420], [422], [423], [427], [428], [429, 431], [430], [437], [441], [442], [443], [447], [453], [455], [458], [460], [475], [491], [493], [496], [498], [503], [516], [518], [519], [520]]


# #sr3d
# sr3dclass = ['bag', 'clothes', 'mug', 'printer', 'blinds', 'mirror', 'soap', 'refrigerator', 'crate', 'closet', 'ledge', 'banner', 'shoes', 'rug', 'mail', 'drawer', 'seat', 'blackboard', 'rack', 'footrest', 'bottles', 'ladder', 'hanging', 'stairs', 'thermostat', 'bowl', 'suitcases', 'bed', 'cups', 'boxes', 'switch', 'books', 'shampoo', 'mattress', 'basket', 'furniture', 'notepad', 'telephone', 'radiator', 'shelf', 'guitar', 'pipes', 'globe', 'clothing', 'can', 'bicycle', 'frame', 'step', 'monitor', 'plant', 'photo', 'person', 'ottoman', 'calendar', 'walls', 'heater', 'futon', 'toaster', 'bin', 'chest', 'jacket', 'stall', 'cart', 'window', 'dresser', 'display', 'shoe', 'chair', 'camera', 'seating', 'shower', 'floor', 'case', 'statue', 'container', 'tank', 'curtains', 'table', 'pictures', 'counter', 'armchair', 'wardrobe', 'stair', 'book', 'couch', 'pillar', 'fan', 'rail', 'shorts', 'column', 'boards', 'pot', 'carpet', 'fireplace', 'suitcase', 'legs', 'headboard', 'dolly', 'bench', 'dishwasher', 'dispenser', 'tv', 'car', 'sign', 'copier', 'footstool', 'oven', 'headphones', 'towels', 'whiteboard', 'folder', 'alarm', 'microwave', 'board', 'vent', 'cup', 'bad', 'bucket', 'bar', 'rod', 'umbrella', 'coat', 'cabinets', 'pan', 'mat', 'purse', 'curtain', 'box', 'bookshelves', 'piano', 'bookshelf', 'clock', 'faucet', 'sink', 'keyboard', 'blanket', 'hat', 'object', 'staircase', 'light', 'tray', 'bike', 'door', 'cushion', 'ceiling', 'hamper', 'pillow', 'pillows', 'containers', 'railing', 'picture', 'poster', 'lamp', 'magazine', 'machine', 'cabinet', 'laptop', 'paper', 'papers', 'sweater', 'pipe', 'bottle', 'towel', 'ball', 'bear', 'desk', 'chairs', 'nightstand', 'painting', 'stool', 'backpack', 'mouse', 'stand', 'screen', 'doors', 'wall', 'cardboard', 'projector', 'stove', 'windowsill', 'luggage', 'vase', 'shirt', 'bathtub', 'stream', 'decoration', 'holder', ' wall', 'wood', 'cloth', 'map', 'sticker', 'easel', 'toilet']
sr_index = [[9, 10, 130, 148, 193, 232, 246, 250, 252, 274, 298, 329, 361, 447, 464, 533, 559], [120], [134, 339], [402, 408], [47], [333], [470], [421], [155], [108, 585], [302], [15], [446], [430], [317], [187], [436], [44], [54, 126, 128, 129, 171, 177, 188, 192, 266, 311, 316, 413, 445, 554, 564, 582], [236], [92], [229, 294], [264, 580], [492], [535], [61], [518], [34, 71, 306, 480], [159, 486, 488], [63, 64, 226, 387], [305, 521], [55], [439], [326], [20, 299], [241], [346], [530], [415], [115, 351, 356, 442, 510], [253], [384], [249], [125], [80, 477, 562], [38, 72], [238], [498], [335], [388, 405], [373], [372], [352], [78], [117], [269, 592], [243], [538], [41, 137, 169, 318, 389, 392, 420, 506, 560], [105], [279], [26], [89], [600], [189], [175, 234], [444], [32, 102, 227, 324, 348, 419, 424, 481, 485, 487], [79], [437], [449], [113, 223, 456], [91, 94, 95, 176, 254, 275], [497], [142, 233, 390, 508], [524], [162], [0, 135, 170, 203, 230, 235, 382, 398, 429, 461, 522], [377], [25, 152, 289], [5], [118, 583], [489], [53, 340], [150], [378], [97, 209], [259, 263, 416, 490], [448], [136], [50], [145, 404], [87], [219], [517], [303], [267], [182], [37, 375], [172], [174, 260, 366, 474, 544], [571], [84], [178, 208, 462, 595], [147], [237], [353, 539], [268], [555], [597], [231], [3, 215], [331], [49, 70, 164, 165, 272, 278], [578], [158, 496], [179], [68], [16, 65, 251, 262, 471], [114, 161, 425, 452], [573], [127], [76, 212, 288], [239], [325], [411], [160, 451], [62, 93, 131, 154, 242, 281, 313, 321, 322, 386, 507, 537], [57], [284, 374, 557], [56], [4, 107], [210], [463], [283], [46], [265], [347], [491], [99, 214, 304, 344, 514], [319, 320, 369, 563], [206], [27, 74, 110, 183, 245, 453, 465, 466], [151, 163], [96, 109], [257, 301], [379], [380], [391], [417], [376], [400, 426], [98, 168, 295, 343, 523, 581], [315], [202, 207, 314, 438, 577, 587, 588], [24, 73, 211, 287, 349, 561, 584], [297], [360], [370], [519], [383], [36, 59, 60, 139, 141, 173, 309, 338, 440, 472, 484, 589], [261, 365, 553], [11, 31, 205], [33, 529], [167], [228], [345], [355], [499, 504, 505], [8], [337], [224, 312, 341, 414, 493, 572], [410, 435], [75, 112, 186, 248, 334, 454], [22, 116, 184, 358, 458, 459, 579], [86], [409], [511], [601], [310], [576], [443], [29], [478], [166], [12], [357], [602], [119], [323], [503], [198], [540]]




def SR_Retrieval(mode = None, full_obj_prob = None, origin_relation = None, obj_distance = None, object_mask = None, context_size = None, n = 1):
    
    """used to generated the relation that need to be compared with objects in the same class for object
    Args:
        obj_prob (_type_): B x N x num_class, N is the number of objects in the scene, the probability of the object belong to a class
        origin_relation (_type_): B x N x N x num_relations, the realtion between objects calculated without comparence considered 
        obj_distance (_type_): B x N x N x 1, the object center, 
        object_mask: B x N, mask the padding objects nums in a scene using '-inf'
        context_size: B, scnen context size
    """
    if mode == 'sr3d':
        ind = sr_index
    elif mode == 'nr3d':
        ind = nr_index
    num_class = len(ind)
    bsz, num_obj = full_obj_prob.shape[:2]
    obj_prob = torch.zeros(num_class).unsqueeze(0).unsqueeze(0).repeat(bsz, num_obj, 1)# B x N x mini Class
    for i in range(num_class):
        obj_prob[:, :, i] = torch.sum(full_obj_prob[:, :, ind[i]], dim = -1)
    obj_prob =  F.softmax(obj_prob, dim = -1) # B x N x mini-Class
    
    
    if n == 1:
        # get top 1 class for object
        mask_obj_class = (torch.argmax(full_obj_prob, dim = -1) + object_mask) # B x N; represent the object class 
        # mask_obj_class = torch.where(torch.isinf(mask_obj_class), torch.full_like(mask_obj_class, -1), mask_obj_class)
        # ------------------------ pytorch -----------------------------
        # for i in range(bsz):
        #     batch_obj_prob = mask_obj_class[i]# N
        #     batch_obj_class_set = set(batch_obj_prob.numpy())
        #     batch_obj_class_set.discard(-np.inf)
        #     for tar_ind in range(context_size[i]):
        #         tar_class = batch_obj_prob[tar_ind]
        #         _batch_obj_class_set = batch_obj_class_set
        #         _batch_obj_class_set.discard(tar_class.numpy().tolist())
        #         for ref_class in _batch_obj_class_set:
        #             ref_obj_ind = torch.nonzero(torch.eq(batch_obj_prob, ref_class)).squeeze(-1)
        #             ref_obj_distance = obj_distance[i, ref_obj_ind, tar_ind]
        #             closet = ref_obj_ind[torch.argmin(ref_obj_distance)]
        #             farthest = ref_obj_ind[torch.argmax(ref_obj_distance)]
        #             if closet == farthest:
        #                 continue
        #             origin_relation[i, closet, tar_ind, -5] = 1
        #             origin_relation[i, farthest, tar_ind, -6] = 1
        # -------------------------------------------------------------
        
        # -------------------------- c++ ------------------------------

        origin_relation = torch.tensor(relation.get_relation(mask_obj_class.squeeze(-1), origin_relation, obj_distance.squeeze(-1), context_size))
        # -------------------------------------------------------------
                    

    else:
        
        # get top n class for objects
        topn_class, topn_class_indx = (torch.sort(obj_prob, dim = -1, descending = False))## B x  N x n_object_class
        topn_class_prob = F.softmax(topn_class[:, :, :n], dim =-1)# B x N x n, represent top n class probability
        topn_class_indx = topn_class_indx[:, :, :n]# B x N x n, represent top n class index
        mask_obj_class = topn_class_indx + object_mask.unsqueeze(-1).repeat(1, 1, n) # B x N x n; represent the object class 
        
        batch_obj_class_line = mask_obj_class.view(bsz, num_obj * n) # B x (N x n)
        batch_obj_prob_line = topn_class_prob.view(bsz, num_obj * n) # B x (N x n)
        
        
        # -------------------------- c++ ------------------------------
        origin_relation = torch.tensor(relation.get_relation_topn(batch_obj_class_line.numpy(), batch_obj_prob_line.numpy(), origin_relation, obj_distance.squeeze(-1), context_size, n))
        # -------------------------------------------------------------
        
        # ------------------------ pytorch -----------------------------
        # for i in range(bsz):
        #     batch_obj_class = mask_obj_class[i]# N x n
        #     topn_class_prob_batch = topn_class_prob[i]
        #     batch_obj_class_line = batch_obj_class.view(num_obj * n)# (N x n)
        #     batch_obj_prob_line = topn_class_prob_batch.view(num_obj * n)# (N x n)
        #     batch_obj_class_set = set(batch_obj_class_line.numpy())
        #     batch_obj_class_set.discard(-np.inf)
            
        #     for tar_ind in range(context_size[i]):## select target object 
        #         for j in range(n):## select target class
        #             tar_class = batch_obj_class[tar_ind, j] # n
        #             _batch_obj_class_set = batch_obj_class_set
        #             _batch_obj_class_set.discard(tar_class.numpy().tolist())
        #             for ref_class in _batch_obj_class_set:
        #                 ref_obj_line_ind = torch.nonzero(torch.eq(batch_obj_class_line, ref_class)).squeeze(-1)# index of the objects belonging to class ref_class in batch_obj_prob_line 
        #                 for num_com in range(1, len(ref_obj_line_ind)):

        #                     for com in combinations(ref_obj_line_ind, num_com):
        #                         obj_ind_complement = torch.tensor(list(set(ref_obj_line_ind) - set(com))).long()
        #                         ref_obj_ind = torch.tensor(com) / n
        #                         # ref_obj_ind_relative = torch.tensor(com) % n
        #                         ref_obj_distance = obj_distance[i, ref_obj_ind.long(), tar_ind]
        #                         closet = ref_obj_ind[torch.argmin(ref_obj_distance)]
        #                         farthest = ref_obj_ind[torch.argmax(ref_obj_distance)]
                                
        #                         p_include = batch_obj_prob_line[torch.tensor(com).long()]# com
        #                         p_notinclude = 1 - batch_obj_prob_line[obj_ind_complement]# (context - epoch)                                
                                
        #                         origin_relation[i, closet.long(), tar_ind, -5] += p_include.prod(dim = -1) * p_notinclude.prod(dim = -1)
        #                         origin_relation[i, farthest.long(), tar_ind, -6] += p_include.prod(dim = -1) * p_notinclude.prod(dim = -1)
        # -------------------------------------------------------------
        
    return origin_relation
                    
                     
    
def Attr_Compute(mode = None, batch = None, full_obj_prob = None, object_mask = None, context_size = None, n = 1):

    bsz, num_obj = full_obj_prob.shape[:2]

    obj_size = batch['obj_size']# B x N x 3
    obj_volume = batch['obj_size'][:, :, 0] * batch['obj_size'][:, :, 1] * batch['obj_size'][:, :, 2]
    obj_diagonal = batch['obj_size'][:, :, 0] * batch['obj_size'][:, :, 1]

    ls_attr = torch.zeros([bsz, num_obj, 2], dtype=torch.double)
    tl_attr = torch.zeros([bsz, num_obj, 2], dtype=torch.double)
    losh_attr = torch.zeros([bsz, num_obj, 2], dtype=torch.double)

    if mode == 'sr3d':
        ind = sr_index
    elif mode == 'nr3d':
        ind = nr_index
    num_class = len(ind)
    obj_prob = torch.zeros(num_class).unsqueeze(0).unsqueeze(0).repeat(bsz, num_obj, 1)# B x N x mini Class
    for i in range(num_class):
        obj_prob[:, :, i] = torch.sum(full_obj_prob[:, :, ind[i]], dim = -1)
    obj_prob =  F.softmax(obj_prob, dim = -1) # B x N x mini-Class

    if n == 1:
        # get top 1 class for object
        mask_obj_class = (torch.argmax(obj_prob, dim = -1) + object_mask) # B x N; represent the object class 
        # mask_obj_class = torch.where(torch.isinf(mask_obj_class), torch.full_like(mask_obj_class, -1), mask_obj_class)
        # ------------------------ pytorch -----------------------------
        for i in range(bsz):
            batch_obj_prob = mask_obj_class[i]# N
            batch_obj_class_set = set(batch_obj_prob.numpy())
            batch_obj_class_set.discard(-np.inf)
            for obj in batch_obj_class_set:
                obj_ind = torch.nonzero(torch.eq(batch_obj_prob, obj)).squeeze(-1)
                class_x = obj_size[i, obj_ind, 0]
                class_y = obj_size[i, obj_ind, 1]
                class_z = obj_size[i, obj_ind, 2]
                class_volume = obj_volume[i, obj_ind]
                class_dislogal = obj_diagonal[i, obj_ind]
                _, x_index = torch.sort(class_x, dim = -1)
                _, y_index = torch.sort(class_y, dim = -1)
                _, z_index = torch.sort(class_z, dim = -1)
                _, volume_index = torch.sort(class_volume, dim = -1)
                _, losh_index = torch.sort(class_dislogal, dim = -1)
                if obj_ind[volume_index[0]] != obj_ind[volume_index[-1]]:
                    ls_attr[i, obj_ind[volume_index[0]], 1] = 1
                    ls_attr[i, obj_ind[volume_index[-1]], 0] = 1
                if obj_ind[z_index[0]] != obj_ind[z_index[-1]]:
                    tl_attr[i, obj_ind[z_index[0]], 1] = 1
                    tl_attr[i, obj_ind[z_index[-1]], 0] = 1
                if obj_ind[losh_index[0]] != obj_ind[losh_index[-1]]:
                    losh_attr[i, obj_ind[losh_index[0]], 1] = 1
                    losh_attr[i, obj_ind[losh_index[-1]], 0] = 1
                # if obj_ind[x_index[0]] != obj_ind[x_index[-1]]:
                #     losh_attr[i, obj_ind[x_index[0]], 1] = 1
                #     losh_attr[i, obj_ind[x_index[-1]], 0] = 1
                # if obj_ind[y_index[0]] != obj_ind[y_index[-1]]:
                #     losh_attr[i, obj_ind[y_index[0]], 1] = 1
                #     losh_attr[i, obj_ind[y_index[-1]], 0] = 1
                    
                # if obj_ind[x_index[0]] != obj_ind[y_index[0]]:
                #     if class_x[x_index[0]] <= class_y[y_index[0]]:
                #         losh_attr[i, obj_ind[y_index[0]], 1] = 0
                #     else:
                #         losh_attr[i, obj_ind[x_index[0]], 1] = 0
                # if obj_ind[x_index[-1]] != obj_ind[y_index[-1]]:
                #     if class_x[x_index[-1]] >= class_y[y_index[-1]]:
                #         losh_attr[i, obj_ind[y_index[-1]], 0] = 0
                #     else:
                #         losh_attr[i, obj_ind[x_index[-1]], 0] = 0



        return ls_attr.cuda().double(), tl_attr.cuda().double(), losh_attr.cuda().double()







                





    

    
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
        