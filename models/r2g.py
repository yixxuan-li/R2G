import torch
import argparse
from torch import nn
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

from . import MLP
from . import NSM, Relation_Estimate, SR_Retrieval, Attr_Estimate, Attr_Compute
from .modules.default_blocks import single_object_encoder, token_encoder, token_embeder, object_decoder_for_clf, text_decoder_for_clf, object_lang_clf
from .modules.utils import get_siamese_features
from datasets.vocabulary import Vocabulary

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None

## raw
# object_class = ['air hockey table', 'airplane', 'alarm', 'alarm clock', 'armchair', 'baby mobile', 'backpack', 'bag', 'bag of coffee beans', 'ball', 'banana holder', 'bananas', 'banister', 'banner', 'bar', 'barricade', 'basket', 'bath products', 'bath walls', 'bathrobe', 'bathroom cabinet', 'bathroom counter','bathroom stall', 'bathroom stall door', 'bathroom vanity', 'bathtub', 'battery disposal jar', 'beachball', 'beanbag chair', 'bear', 'bed', 'beer bottles', 'bench', 'bicycle', 'bike lock', 'bike pump', 'bin', 'blackboard', 'blanket', 'blinds', 'block', 'board', 'boards', 'boat', 'boiler', 'book', 'book rack', 'books', 'bookshelf', 'bookshelves', 'boots', 'bottle', 'bowl', 'box', 'boxes', 'boxes of paper', 'breakfast bar', 'briefcase', 'broom', 'bucket', 'bulletin board', 'bunk bed', 'cabinet', 'cabinet door', 'cabinet doors', 'cabinets', 'cable', 'calendar', 'camera', 'can', 'candle', 'canopy', 'car', 'card', 'cardboard', 'carpet', 'carseat', 'cart', 'carton', 'case', 'case of water bottles', 'cat litter box', 'cd case', 'ceiling', 'ceiling fan', 'ceiling light', 'chain', 'chair', 'chandelier', 'changing station', 'chest', 'clock', 'closet', 'closet ceiling', 'closet door', 'closet doorframe', 'closet doors', 'closet floor', 'closet rod', 'closet shelf', 'closet wall', 'closet walls', 'cloth', 'clothes', 'clothes dryer', 'clothes dryers', 'clothes hanger', 'clothes hangers', 'clothing', 'clothing rack', 'coat', 'coat rack', 'coatrack', 'coffee box', 'coffee kettle', 'coffee maker', 'coffee table', 'column', 'compost bin', 'computer tower', 'conditioner bottle', 'container', 'controller', 'cooking pan', 'cooking pot', 'copier', 'costume', 'couch', 'couch cushions', 'counter', 'covered box', 'crate', 'crib', 'cup', 'cups', 'curtain', 'curtains', 'cushion', 'cutting board', 'dart board', 'decoration', 'desk', 'desk lamp', 'diaper bin', 'dining table', 'dish rack', 'dishwasher', 'dishwashing soap bottle', 'dispenser', 'display', 'display case', 'display rack', 'divider', 'doll', 'dollhouse', 'dolly', 'door', 'doorframe', 'doors', 'drawer', 'dress rack', 'dresser', 'drum set', 'dryer sheets', 'drying rack', 'duffel bag', 'dumbbell', 'dustpan', 'easel', 'electric panel', 'elevator', 'elevator button', 'elliptical machine', 'end table', 'envelope', 'exercise bike', 'exercise machine', 'exit sign', 'fan', 'faucet', 'file cabinet', 'fire alarm', 'fire extinguisher', 'fireplace', 'flag', 'flip flops', 'floor', 'flower stand', 'flowerpot', 'folded chair', 'folded chairs', 'folded ladder', 'folded table', 'folder', 'food bag', 'food container', 'food display', 'foosball table', 'footrest', 'footstool', 'frame', 'frying pan', 'furnace', 'furniture', 'fuse box', 'futon', 'garage door', 'garbage bag', 'glass doors', 'globe', 'golf bag', 'grab bar', 'grocery bag', 'guitar', 'guitar case', 'hair brush', 'hair dryer', 'hamper', 'hand dryer', 'hand rail', 'hand sanitzer dispenser', 'hand towel', 'handicap bar', 'handrail', 'hanging', 'hat', 'hatrack', 'headboard', 'headphones', 'heater', 'helmet', 'hose', 'hoverboard', 'humidifier', 'ikea bag', 'instrument case', 'ipad', 'iron', 'ironing board', 'jacket', 'jar', 'kettle', 'keyboard', 'keyboard piano', 'kitchen apron', 'kitchen cabinet', 'kitchen cabinets', 'kitchen counter', 'kitchen island', 'kitchenaid mixer', 'knife block', 'ladder', 'lamp', 'lamp base', 'laptop', 'laundry bag', 'laundry basket', 'laundry detergent', 'laundry hamper', 'ledge', 'legs', 'light', 'light switch', 'loft bed', 'loofa', 'luggage', 'luggage rack', 'luggage stand', 'lunch box', 'machine', 'magazine', 'magazine rack', 'mail', 'mail tray', 'mailbox', 'mailboxes', 'map', 'massage chair', 'mat', 'mattress', 'medal', 'messenger bag', 'metronome', 'microwave', 'mini fridge', 'mirror', 'mirror doors', 'monitor', 'mouse', 'mouthwash bottle', 'mug', 'music book', 'music stand', 'nerf gun', 'night lamp', 'nightstand', 'notepad', 'object', 'office chair', 'open kitchen cabinet', 'organizer', 'organizer shelf', 'ottoman', 'oven', 'oven mitt', 'painting', 'pantry shelf', 'pantry wall', 'pantry walls', 'pants', 'paper', 'paper bag', 'paper cutter', 'paper organizer', 'paper towel', 'paper towel dispenser', 'paper towel roll', 'paper tray', 'papers', 'person', 'photo', 'piano', 'piano bench', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'ping pong table', 'pipe', 'pipes', 'pitcher', 'pizza boxes', 'plant', 'plastic bin', 'plastic container', 'plastic containers', 'plastic storage bin', 'plate', 'plates', 'plunger', 'podium', 'pool table', 'poster', 'poster cutter', 'poster printer', 'poster tube', 'pot', 'potted plant', 'power outlet', 'power strip', 'printer', 'projector', 'projector screen', 'purse', 'quadcopter', 'rack', 'rack stand', 'radiator', 'rail', 'railing', 'range hood', 'recliner chair', 'recycling bin', 'refrigerator', 'remote', 'rice cooker', 'rod', 'rolled poster', 'roomba', 'rope', 'round table', 'rug', 'salt', 'santa', 'scale', 'scanner', 'screen', 'seat', 'seating', 'sewing machine', 'shampoo', 'shampoo bottle', 'shelf', 'shirt', 'shoe', 'shoe rack', 'shoes', 'shopping bag', 'shorts', 'shower', 'shower control valve', 'shower curtain', 'shower curtain rod', 'shower door', 'shower doors', 'shower floor', 'shower head', 'shower wall', 'shower walls', 'shredder', 'sign', 'sink', 'sliding wood door', 'slippers', 'smoke detector', 'soap', 'soap bottle', 'soap dish', 'soap dispenser', 'sock', 'soda stream', 'sofa bed', 'sofa chair', 'speaker', 'sponge', 'spray bottle', 'stack of chairs', 'stack of cups', 'stack of folded chairs', 'stair', 'stair rail', 'staircase', 'stairs', 'stand', 'stapler', 'starbucks cup', 'statue', 'step', 'step stool', 'sticker', 'stool', 'storage bin', 'storage box', 'storage container', 'storage organizer', 'storage shelf', 'stove', 'structure', 'studio light', 'stuffed animal', 'suitcase', 'suitcases', 'sweater', 'swiffer', 'switch', 'table', 'tank', 'tap', 'tape', 'tea kettle', 'teapot', 'teddy bear', 'telephone', 'telescope', 'thermostat', 'tire', 'tissue box', 'toaster', 'toaster oven', 'toilet', 'toilet brush', 'toilet flush button', 'toilet paper', 'toilet paper dispenser', 'toilet paper holder', 'toilet paper package', 'toilet paper rolls', 'toilet seat cover dispenser', 'toiletry', 'toolbox', 'toothbrush', 'toothpaste', 'towel', 'towel rack', 'towels', 'toy dinosaur', 'toy piano', 'traffic cone', 'trash bag', 'trash bin', 'trash cabinet', 'trash can', 'tray', 'tray rack', 'treadmill', 'tripod', 'trolley', 'trunk', 'tube', 'tupperware', 'tv', 'tv stand', 'umbrella', 'urinal', 'vacuum cleaner', 'vase', 'vending machine', 'vent', 'wall', 'wall hanging', 'wall lamp', 'wall mounted coat rack', 'wardrobe', 'wardrobe cabinet', 'wardrobe closet', 'washcloth', 'washing machine', 'washing machines', 'water bottle', 'water cooler', 'water fountain', 'water heater', 'water pitcher', 'wet floor sign', 'wheel', 'whiteboard', 'whiteboard eraser', 'window', 'windowsill', 'wood', 'wood beam', 'workbench', 'yoga mat', 'pad']

# object_class = ['table', '+airplane', 'alarm', 'alarm', 'armchair', '+baby mobile', 'backpack', 'bag', 'bag', 'ball', '+banana holder', '+bananas', '+banister', 'banner', 'bar', '+barricade', 'basket', '+bath products', 'wall', '+bathrobe', 'cabinet', 'counter', '+bathroom stall', 'door', '+bathroom vanity', 'bathtub', '+battery disposal jar', 'ball', 'chair', 'bear', 'bed', 'bottles', 'bench', 'bicycle', '+bike lock', '+bike pump', 'bin', 'blackboard', 'blanket', 'blinds', '+block', 'board', 'boards', '+boat', '+boiler', 'book', 'rack', 'books', 'bookshelf', 'bookshelves', '+boots', 'bottle', 'bowl', 'box', 'boxes', 'boxes', 'bar', '+briefcase', '+broom', 'bucket', 'board', 'bed', 'cabinet', 'door', 'doors', 'cabinets', '+cable', 'calendar', 'camera', 'can', '+candle', '+canopy', 'car', '+card', 'cardboard', 'carpet', '+carseat', 'cart', '+carton', 'case', 'bottle', 'box', 'case', 'ceiling', 'fan', 'light', '+chain', 'chair', '+chandelier', '+changing station', 'chest', 'clock', 'closet', 'ceiling', 'door', '+closet doorframe', 'doors', 'floor', 'rod', 'shelf', 'wall', 'wall', 'cloth', 'clothes', '+clothes dryer', '+clothes dryers', '+clothes hanger', '+clothes hangers', 'clothing', 'rack', 'coat', 'rack', 'rack', 'box', '+coffee kettle', '+coffee maker', 'table', 'column', 'bin', '+computer tower', 'bottle', 'container', '+controller', '+cooking pan', 'pot', 'copier', '+costume', 'couch', 'cushion', 'counter', 'box', 'crate', '+crib', 'cup', 'cups', 'curtain', 'curtains', 'cushion', 'board', 'board', 'decoration', 'desk', 'lamp', 'bin', 'table', 'rack', 'dishwasher', 'bottle', 'dispenser', 'display', 'case', 'rack', '+divider', '+doll', '+dollhouse', 'dolly', 'door', '+doorframe', 'doors', 'drawer', 'rack', 'dresser', '+drum set', '+dryer sheets', 'rack', 'bag', '+dumbbell', '+dustpan', 'easel', '+electric panel', '+elevator', '+elevator button', 'machine', 'table', '+envelope', '+exercise bike', 'machine', 'sign', 'fan', 'faucet', 'cabinet', 'alarm', '+fire extinguisher', 'fireplace', '+flag', '+flip flops', 'floor', 'stand', '+flowerpot', 'chair', 'chair', 'ladder', 'table', 'folder', 'bag', 'container', 'display', 'table', 'footrest', 'footstool', 'frame', '+frying pan', '+furnace', 'furniture', 'box', 'futon', 'door', 'bag', 'doors', 'globe', 'bag', 'bar', 'bag', 'guitar', 'case', '+hair brush', '+hair dryer', 'hamper', '+hand dryer', 'rail', 'dispenser', 'towel', 'bar', 'rail', 'hanging', 'hat', 'rack', 'headboard', 'headphones', 'heater', '+helmet', '+hose', '+hoverboard', '+humidifier', 'bag', 'case', '+ipad', '+iron', 'board', 'jacket', '+jar', '+kettle', 'keyboard', 'piano', '+kitchen apron', 'cabinet', 'cabinets', 'counter', '+kitchen island', '+kitchenaid mixer', '+knife block', 'ladder', 'lamp', '+lamp base', 'laptop', 'bag', 'basket', '+laundry detergent', 'hamper', 'ledge', 'legs', 'light', 'switch', 'bed', '+loofa', 'luggage', 'rack', 'stand', 'box', 'machine', 'magazine', 'rack', 'mail', 'tray', '+mailbox', '+mailboxes', 'map', 'chair', 'mat', 'mattress', '+medal', 'bag', '+metronome', 'microwave', '+mini fridge', 'mirror', 'doors', 'monitor', 'mouse', 'bottle', 'mug', 'book', 'stand', '+nerf gun', 'lamp', 'nightstand', 'notepad', 'object', 'chair', 'cabinet', '+organizer', 'shelf', 'ottoman', 'oven', '+oven mitt', 'painting', 'shelf', 'wall', 'wall', '+pants', 'paper', 'bag', '+paper cutter', '+paper organizer', 'towel', 'dispenser', 'towel', 'tray', 'papers', 'person', 'photo', 'piano', '+piano bench', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'table', 'pipe', 'pipes', '+pitcher', 'boxes', 'plant', 'bin', 'container', 'container', 'bin', '+plate', '+plates', '+plunger', '+podium', 'table', 'poster', '+poster cutter', 'printer', '+poster tube', 'pot', 'plant', '+power outlet', '+power strip', 'printer', 'projector', 'screen', 'purse', '+quadcopter', 'rack', 'stand', 'radiator', 'rail', 'railing', '+range hood', 'chair', 'bin', 'refrigerator', '+remote', '+rice cooker', 'rod', 'poster', 'roomba', '+rope', 'table', 'rug', '+salt', '+santa', '+scale', '+scanner', 'screen', 'seat', 'seating', 'machine', 'shampoo', 'bottle', 'shelf', 'shirt', 'shoe', 'rack', 'shoes', 'bag', 'shorts', 'shower', '+shower control valve', 'curtain', 'rod', 'door', 'doors', 'floor', '+shower head', 'wall', 'walls', '+shredder', 'sign', 'sink', 'door', '+slippers', '+smoke detector', 'soap', 'bottle', '+soap dish', 'dispenser', '+sock', '+soda stream', 'bed', 'chair', '+speaker', '+sponge', 'bottle', 'chair', 'cups', 'chair', 'stair', 'rail', 'staircase', 'stairs', 'stand', '+stapler', 'cup', 'statue', 'step', 'stool', 'sticker', 'stool', 'bin', 'box', 'container', '+storage organizer', 'shelf', 'stove', '+structure', 'light', '+stuffed animal', 'suitcase', 'suitcases', 'sweater', '+swiffer', 'switch', 'table', 'tank', '+tap', '+tape', '+tea kettle', '+teapot', 'bear', 'telephone', '+telescope', 'thermostat', '+tire', 'box', 'toaster', 'oven', 'toilet', '+toilet brush', '+toilet flush button', 'paper', 'dispenser', '+toilet paper holder', '+toilet paper package', '+toilet paper rolls', 'dispenser', '+toiletry', '+toolbox', '+toothbrush', '+toothpaste', 'towel', 'rack', 'towels', '+toy dinosaur', 'piano', '+traffic cone', 'bag', 'bin', 'cabinet', 'can', 'tray', 'rack', '+treadmill', '+tripod', '+trolley', '+trunk', '+tube', '+tupperware', 'tv', 'stand', 'umbrella', '+urinal', '+vacuum cleaner', 'vase', 'machine', 'vent', 'wall', 'hanging', 'lamp', 'rack', 'wardrobe', 'cabinet', 'closet', '+washcloth', 'machine', 'machine', 'bottle', '+water cooler', '+water fountain', 'heater', '+water pitcher', 'sign', '+wheel', 'whiteboard', '+whiteboard eraser', 'window', 'windowsill', 'wood', '+wood beam', '+workbench', '+yoga mat', 'pad']

# # nr3d
# nr3dclass = ['table', 'alarm', 'armchair', 'backpack', 'bag', 'ball', 'banner', 'bar', 'basket', 'wall', 'cabinet', 'counter', 'door', 'bathtub', 'chair', 'bear', 'bed', 'bottles', 'bench', 'bicycle', 'bin', 'blackboard', 'blanket', 'blinds', 'board', 'boards', 'book', 'rack', 'books', 'bookshelf', 'bookshelves', 'bottle', 'bowl', 'box', 'boxes', 'bucket', 'doors', 'cabinets', 'calendar', 'camera', 'can', 'car', 'cardboard', 'carpet', 'cart', 'case', 'ceiling', 'fan', 'light', 'chest', 'clock', 'closet', 'floor', 'rod', 'shelf', 'cloth', 'clothes', 'clothing', 'coat', 'column', 'container', 'pot', 'copier', 'couch', 'cushion', 'crate', 'cup', 'cups', 'curtain', 'curtains', 'decoration', 'desk', 'lamp', 'dishwasher', 'dispenser', 'display', 'dolly', 'drawer', 'dresser', 'easel', 'machine', 'sign', 'faucet', 'fireplace', 'stand', 'ladder', 'folder', 'footrest', 'footstool', 'frame', 'furniture', 'futon', 'globe', 'guitar', 'hamper', 'rail', 'towel', 'hanging', 'hat', 'headboard', 'headphones', 'heater', 'jacket', 'keyboard', 'piano', 'laptop', 'ledge', 'legs', 'switch', 'luggage', 'magazine', 'mail', 'tray', 'map', 'mat', 'mattress', 'microwave', 'mirror', 'monitor', 'mouse', 'mug', 'nightstand', 'notepad', 'object', 'ottoman', 'oven', 'painting', 'paper', 'papers', 'person', 'photo', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'pipe', 'pipes', 'plant', 'poster', 'printer', 'projector', 'screen', 'purse', 'radiator', 'railing', 'refrigerator', 'roomba', 'rug', 'seat', 'seating', 'shampoo', 'shirt', 'shoe', 'shoes', 'shorts', 'shower', 'walls', 'sink', 'soap', 'stair', 'staircase', 'stairs', 'statue', 'step', 'stool', 'sticker', 'stove', 'suitcase', 'suitcases', 'sweater', 'tank', 'telephone', 'thermostat', 'toaster', 'toilet', 'towels', 'tv', 'umbrella', 'vase', 'vent', 'wardrobe', 'whiteboard', 'window', 'windowsill', 'wood']
nr_index = [[0, 116, 144, 173, 192, 197, 328, 342, 371, 446], [2, 3, 181], [4], [6], [7, 8, 165, 194, 207, 210, 212, 234, 255, 281, 311, 388, 479], [9, 27], [13], [14, 56, 211, 222], [16, 256], [18, 100, 101, 307, 308, 398, 499], [20, 62, 180, 245, 299, 481, 504], [21, 129, 247], [23, 63, 94, 156, 206, 394, 403], [25], [28, 87, 189, 190, 277, 298, 362, 413, 417, 419], [29, 452], [30, 61, 263, 412], [31], [32], [33], [36, 118, 143, 334, 337, 363, 432, 480], [37], [38], [39], [41, 60, 138, 139, 238], [42], [45, 291], [46, 109, 111, 112, 145, 151, 160, 164, 226, 266, 271, 356, 386, 474, 484, 502], [47], [48], [49], [51, 80, 120, 147, 289, 382, 407, 416, 509], [52], [53, 81, 113, 130, 204, 268, 433, 457], [54, 55, 332], [59], [64, 96, 158, 208, 286, 395], [65, 246], [67], [68], [69, 482], [72], [74], [75], [77], [79, 82, 150, 214, 235], [83, 93], [84, 178], [85, 261, 439], [90], [91], [92, 505], [97, 186, 396], [98, 367, 393], [99, 301, 306, 383, 436], [102], [103], [108], [110], [117], [121, 195, 335, 336, 434], [124, 347], [125], [127], [128, 137], [131], [133, 426], [134, 418], [135, 392], [136], [140], [141], [142, 252, 294, 501], [146], [148, 220, 315, 409, 464, 468], [149, 196], [155], [159], [161], [168], [172, 176, 269, 380, 497, 507, 508], [177, 401, 514], [179], [183], [187, 267, 292, 357, 424, 492], [191, 251], [193], [198], [199], [200], [203], [205], [209], [213], [217, 258], [219, 223, 359, 421], [221, 314, 316, 473], [224, 500], [225], [227], [228], [229, 512], [239], [242], [243, 321, 477], [254], [259], [260], [262, 445], [265], [270], [272], [273, 317, 483], [276], [278], [279], [283], [285], [287], [288], [290], [295], [296], [297], [302], [303, 459], [305], [310, 463], [318], [319], [320], [323], [324], [325], [326], [327], [329], [330], [333, 348], [343, 368], [345, 351], [352], [353, 377], [354], [358], [360], [364], [369], [372], [378], [379], [381], [384], [385], [387], [389], [390], [399], [402], [406], [420], [422], [423], [427], [428], [429, 431], [430], [437], [441], [442], [443], [447], [453], [455], [458], [460], [475], [491], [493], [496], [498], [503], [516], [518], [519], [520]]


# #sr3d
# sr3dclass = ['bag', 'clothes', 'mug', 'printer', 'blinds', 'mirror', 'soap', 'refrigerator', 'crate', 'closet', 'ledge', 'banner', 'shoes', 'rug', 'mail', 'drawer', 'seat', 'blackboard', 'rack', 'footrest', 'bottles', 'ladder', 'hanging', 'stairs', 'thermostat', 'bowl', 'suitcases', 'bed', 'cups', 'boxes', 'switch', 'books', 'shampoo', 'mattress', 'basket', 'furniture', 'notepad', 'telephone', 'radiator', 'shelf', 'guitar', 'pipes', 'globe', 'clothing', 'can', 'bicycle', 'frame', 'step', 'monitor', 'plant', 'photo', 'person', 'ottoman', 'calendar', 'walls', 'heater', 'futon', 'toaster', 'bin', 'chest', 'jacket', 'stall', 'cart', 'window', 'dresser', 'display', 'shoe', 'chair', 'camera', 'seating', 'shower', 'floor', 'case', 'statue', 'container', 'tank', 'curtains', 'table', 'pictures', 'counter', 'armchair', 'wardrobe', 'stair', 'book', 'couch', 'pillar', 'fan', 'rail', 'shorts', 'column', 'boards', 'pot', 'carpet', 'fireplace', 'suitcase', 'legs', 'headboard', 'dolly', 'bench', 'dishwasher', 'dispenser', 'tv', 'car', 'sign', 'copier', 'footstool', 'oven', 'headphones', 'towels', 'whiteboard', 'folder', 'alarm', 'microwave', 'board', 'vent', 'cup', 'bad', 'bucket', 'bar', 'rod', 'umbrella', 'coat', 'cabinets', 'pan', 'mat', 'purse', 'curtain', 'box', 'bookshelves', 'piano', 'bookshelf', 'clock', 'faucet', 'sink', 'keyboard', 'blanket', 'hat', 'object', 'staircase', 'light', 'tray', 'bike', 'door', 'cushion', 'ceiling', 'hamper', 'pillow', 'pillows', 'containers', 'railing', 'picture', 'poster', 'lamp', 'magazine', 'machine', 'cabinet', 'laptop', 'paper', 'papers', 'sweater', 'pipe', 'bottle', 'towel', 'ball', 'bear', 'desk', 'chairs', 'nightstand', 'painting', 'stool', 'backpack', 'mouse', 'stand', 'screen', 'doors', 'wall', 'cardboard', 'projector', 'stove', 'windowsill', 'luggage', 'vase', 'shirt', 'bathtub', 'stream', 'decoration', 'holder', ' wall', 'wood', 'cloth', 'map', 'sticker', 'easel', 'toilet']
sr_index = [[9, 10, 130, 148, 193, 232, 246, 250, 252, 274, 298, 329, 361, 447, 464, 533, 559], [120], [134, 339], [402, 408], [47], [333], [470], [421], [155], [108, 585], [302], [15], [446], [430], [317], [187], [436], [44], [54, 126, 128, 129, 171, 177, 188, 192, 266, 311, 316, 413, 445, 554, 564, 582], [236], [92], [229, 294], [264, 580], [492], [535], [61], [518], [34, 71, 306, 480], [159, 486, 488], [63, 64, 226, 387], [305, 521], [55], [439], [326], [20, 299], [241], [346], [530], [415], [115, 351, 356, 442, 510], [253], [384], [249], [125], [80, 477, 562], [38, 72], [238], [498], [335], [388, 405], [373], [372], [352], [78], [117], [269, 592], [243], [538], [41, 137, 169, 318, 389, 392, 420, 506, 560], [105], [279], [26], [89], [600], [189], [175, 234], [444], [32, 102, 227, 324, 348, 419, 424, 481, 485, 487], [79], [437], [449], [113, 223, 456], [91, 94, 95, 176, 254, 275], [497], [142, 233, 390, 508], [524], [162], [0, 135, 170, 203, 230, 235, 382, 398, 429, 461, 522], [377], [25, 152, 289], [5], [118, 583], [489], [53, 340], [150], [378], [97, 209], [259, 263, 416, 490], [448], [136], [50], [145, 404], [87], [219], [517], [303], [267], [182], [37, 375], [172], [174, 260, 366, 474, 544], [571], [84], [178, 208, 462, 595], [147], [237], [353, 539], [268], [555], [597], [231], [3, 215], [331], [49, 70, 164, 165, 272, 278], [578], [158, 496], [179], [68], [16, 65, 251, 262, 471], [114, 161, 425, 452], [573], [127], [76, 212, 288], [239], [325], [411], [160, 451], [62, 93, 131, 154, 242, 281, 313, 321, 322, 386, 507, 537], [57], [284, 374, 557], [56], [4, 107], [210], [463], [283], [46], [265], [347], [491], [99, 214, 304, 344, 514], [319, 320, 369, 563], [206], [27, 74, 110, 183, 245, 453, 465, 466], [151, 163], [96, 109], [257, 301], [379], [380], [391], [417], [376], [400, 426], [98, 168, 295, 343, 523, 581], [315], [202, 207, 314, 438, 577, 587, 588], [24, 73, 211, 287, 349, 561, 584], [297], [360], [370], [519], [383], [36, 59, 60, 139, 141, 173, 309, 338, 440, 472, 484, 589], [261, 365, 553], [11, 31, 205], [33, 529], [167], [228], [345], [355], [499, 504, 505], [8], [337], [224, 312, 341, 414, 493, 572], [410, 435], [75, 112, 186, 248, 334, 454], [22, 116, 184, 358, 458, 459, 579], [86], [409], [511], [601], [310], [576], [443], [29], [478], [166], [12], [357], [602], [119], [323], [503], [198], [540]]



#nr3d
object_class1 = ['table', 'airplane', 'alarm', 'clock', 'armchair', 'mobile', 'backpack', 'bag', 'beans', 'ball', 'holder', 'bananas', 'banister', 'banner', 'bar', 'barricade', 'basket', 'products', 'walls', 'bathrobe', 'cabinet', 'counter', 'stall', 'door', 'vanity', 'bathtub', 'jar', 'beachball', 'chair', 'bear', 'bed', 'bottles', 'bench', 'bicycle', 'lock', 'pump', 'bin', 'blackboard', 'blanket', 'blinds', 'block', 'board', 'boards', 'boat', 'boiler', 'book', 'rack', 'books', 'bookshelf', 'bookshelves', 'boots', 'bottle', 'bowl', 'box', 'boxes', 'paper', 'bar', 'briefcase', 'broom', 'bucket', 'board', 'bed', 'cabinet', 'door', 'doors', 'cabinets', 'cable', 'calendar', 'camera', 'can', 'candle', 'canopy', 'car', 'card', 'cardboard', 'carpet', 'carseat', 'cart', 'carton', 'case', 'bottles', 'box', 'case', 'ceiling', 'fan', 'light', 'chain', 'chair', 'chandelier', 'station', 'chest', 'clock', 'closet', 'ceiling', 'door', 'doorframe', 'doors', 'floor', 'rod', 'shelf', 'wall', 'walls', 'cloth', 'clothes', 'dryer', 'dryers', 'hanger', 'hangers', 'clothing', 'rack', 'coat', 'rack', 'coatrack', 'box', 'kettle', 'maker', 'table', 'column', 'bin', 'tower', 'bottle', 'container', 'controller', 'pan', 'pot', 'copier', 'costume', 'couch', 'cushions', 'counter', 'box', 'crate', 'crib', 'cup', 'cups', 'curtain', 'curtains', 'cushion', 'board', 'board', 'decoration', 'desk', 'lamp', 'bin', 'table', 'rack', 'dishwasher', 'bottle', 'dispenser', 'display', 'case', 'rack', 'divider', 'doll', 'dollhouse', 'dolly', 'door', 'doorframe', 'doors', 'drawer', 'rack', 'dresser', 'set', 'sheets', 'rack', 'bag', 'dumbbell', 'dustpan', 'easel', 'panel', 'elevator', 'button', 'machine', 'table', 'envelope', 'bike', 'machine', 'sign', 'fan', 'faucet', 'cabinet', 'alarm', 'extinguisher', 'fireplace', 'flag', 'flops', 'floor', 'stand', 'flowerpot', 'chair', 'chairs', 'ladder', 'table', 'folder', 'bag', 'container', 'display', 'table', 'footrest', 'footstool', 'frame', 'pan', 'furnace', 'furniture', 'box', 'futon', 'door', 'bag', 'doors', 'globe', 'bag', 'bar', 'bag', 'guitar', 'case', 'brush', 'dryer', 'hamper', 'dryer', 'rail', 'dispenser', 'towel', 'bar', 'handrail', 'hanging', 'hat', 'hatrack', 'headboard', 'headphones', 'heater', 'helmet', 'hose', 'hoverboard', 'humidifier', 'bag', 'case', 'ipad', 'iron', 'board', 'jacket', 'jar', 'kettle', 'keyboard', 'piano', 'apron', 'cabinet', 'cabinets', 'counter', 'island', 'mixer', 'block', 'ladder', 'lamp', 'base', 'laptop', 'bag', 'basket', 'detergent', 'hamper', 'ledge', 'legs', 'light', 'switch', 'bed', 'loofa', 'luggage', 'rack', 'stand', 'box', 'machine', 'magazine', 'rack', 'mail', 'tray', 'mailbox', 'mailboxes', 'map', 'chair', 'mat', 'mattress', 'medal', 'bag', 'metronome', 'microwave', 'fridge', 'mirror', 'doors', 'monitor', 'mouse', 'bottle', 'mug', 'book', 'stand', 'gun', 'lamp', 'nightstand', 'notepad', 'object', 'chair', 'cabinet', 'organizer', 'shelf', 'ottoman', 'oven', 'mitt', 'painting', 'shelf', 'wall', 'walls', 'pants', 'paper', 'bag', 'cutter', 'organizer', 'towel', 'dispenser', 'roll', 'tray', 'papers', 'person', 'photo', 'piano', 'bench', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'table', 'pipe', 'pipes', 'pitcher', 'boxes', 'plant', 'bin', 'container', 'containers', 'bin', 'plate', 'plates', 'plunger', 'podium', 'table', 'poster', 'cutter', 'printer', 'tube', 'pot', 'plant', 'outlet', 'strip', 'printer', 'projector', 'screen', 'purse', 'quadcopter', 'rack', 'stand', 'radiator', 'rail', 'railing', 'hood', 'chair', 'bin', 'refrigerator', 'remote', 'cooker', 'rod', 'poster', 'roomba', 'rope', 'table', 'rug', 'salt', 'santa', 'scale', 'scanner', 'screen', 'seat', 'seating', 'machine', 'shampoo', 'bottle', 'shelf', 'shirt', 'shoe', 'rack', 'shoes', 'bag', 'shorts', 'shower', 'valve', 'curtain', 'rod', 'door', 'doors', 'floor', 'head', 'wall', 'walls', 'shredder', 'sign', 'sink', 'door', 'slippers', 'detector', 'soap', 'bottle', 'dish', 'dispenser', 'sock', 'stream', 'bed', 'chair', 'speaker', 'sponge', 'bottle', 'chairs', 'cups', 'chairs', 'stair', 'rail', 'staircase', 'stairs', 'stand', 'stapler', 'cup', 'statue', 'step', 'stool', 'sticker', 'stool', 'bin', 'box', 'container', 'organizer', 'shelf', 'stove', 'structure', 'light', 'animal', 'suitcase', 'suitcases', 'sweater', 'swiffer', 'switch', 'table', 'tank', 'tap', 'tape', 'kettle', 'teapot', 'bear', 'telephone', 'telescope', 'thermostat', 'tire', 'box', 'toaster', 'oven', 'toilet', 'brush', 'button', 'paper', 'dispenser', 'holder', 'package', 'rolls', 'dispenser', 'toiletry', 'toolbox', 'toothbrush', 'toothpaste', 'towel', 'rack', 'towels', 'dinosaur', 'piano', 'cone', 'bag', 'bin', 'cabinet', 'can', 'tray', 'rack', 'treadmill', 'tripod', 'trolley', 'trunk', 'tube', 'tupperware', 'tv', 'stand', 'umbrella', 'urinal', 'cleaner', 'vase', 'machine', 'vent', 'wall', 'hanging', 'lamp', 'rack', 'wardrobe', 'cabinet', 'closet', 'washcloth', 'machine', 'machines', 'bottle', 'cooler', 'fountain', 'heater', 'pitcher', 'sign', 'wheel', 'whiteboard', 'eraser', 'window', 'windowsill', 'wood', 'beam', 'workbench', 'mat', 'pad']
my_function1 = ['air', 'pad', 'pad', 'alarm', 'pad', 'baby', 'pad', 'pad', 'bag', 'pad', 'banana', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'bath', 'bath', 'pad', 'bathroom', 'bathroom', 'bathroom', 'bathroom', 'bathroom', 'pad', 'battery', 'pad', 'beanbag', 'pad', 'pad', 'beer', 'pad', 'pad', 'bike', 'bike', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'book', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'boxes', 'breakfast', 'pad', 'pad', 'pad', 'bulletin', 'bunk', 'pad', 'cabinet', 'cabinet', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'case', 'cat', 'cd', 'pad', 'ceiling', 'ceiling', 'pad', 'pad', 'pad', 'changing', 'pad', 'pad', 'pad', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'pad', 'pad', 'clothes', 'clothes', 'clothes', 'clothes', 'pad', 'clothing', 'pad', 'coat', 'pad', 'coffee', 'coffee', 'coffee', 'coffee', 'pad', 'compost', 'computer', 'conditioner', 'pad', 'pad', 'cooking', 'cooking', 'pad', 'pad', 'pad', 'couch', 'pad', 'covered', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'cutting', 'dart', 'pad', 'pad', 'desk', 'diaper', 'dining', 'dish', 'pad', 'dishwashing', 'pad', 'pad', 'display', 'display', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'dress', 'pad', 'drum', 'dryer', 'drying', 'duffel', 'pad', 'pad', 'pad', 'electric', 'pad', 'elevator', 'elliptical', 'end', 'pad', 'exercise', 'exercise', 'exit', 'pad', 'pad', 'file', 'fire', 'fire', 'pad', 'pad', 'flip', 'pad', 'flower', 'pad', 'folded', 'folded', 'folded', 'folded', 'pad', 'food', 'food', 'food', 'foosball', 'pad', 'pad', 'pad', 'frying', 'pad', 'pad', 'fuse', 'pad', 'garage', 'garbage', 'glass', 'pad', 'golf', 'grab', 'grocery', 'pad', 'guitar', 'hair', 'hair', 'pad', 'hand', 'hand', 'hand', 'hand', 'handicap', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'ikea', 'instrument', 'pad', 'pad', 'ironing', 'pad', 'pad', 'pad', 'pad', 'keyboard', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchenaid', 'knife', 'pad', 'pad', 'lamp', 'pad', 'laundry', 'laundry', 'laundry', 'laundry', 'pad', 'pad', 'pad', 'light', 'loft', 'pad', 'pad', 'luggage', 'luggage', 'lunch', 'pad', 'pad', 'magazine', 'pad', 'mail', 'pad', 'pad', 'pad', 'massage', 'pad', 'pad', 'pad', 'messenger', 'pad', 'pad', 'mini', 'pad', 'mirror', 'pad', 'pad', 'mouthwash', 'pad', 'music', 'music', 'nerf', 'night', 'pad', 'pad', 'pad', 'office', 'open', 'pad', 'organizer', 'pad', 'pad', 'oven', 'pad', 'pantry', 'pantry', 'pantry', 'pad', 'pad', 'paper', 'paper', 'paper', 'paper', 'paper', 'paper', 'paper', 'pad', 'pad', 'pad', 'pad', 'piano', 'pad', 'pad', 'pad', 'pad', 'pad', 'ping', 'pad', 'pad', 'pad', 'pizza', 'pad', 'plastic', 'plastic', 'plastic', 'plastic', 'pad', 'pad', 'pad', 'pad', 'pool', 'pad', 'poster', 'poster', 'poster', 'pad', 'potted', 'power', 'power', 'pad', 'pad', 'projector', 'pad', 'pad', 'pad', 'rack', 'pad', 'pad', 'pad', 'range', 'recliner', 'recycling', 'pad', 'pad', 'rice', 'pad', 'rolled', 'pad', 'pad', 'round', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'sewing', 'pad', 'shampoo', 'pad', 'pad', 'pad', 'shoe', 'pad', 'shopping', 'pad', 'pad', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'pad', 'pad', 'pad', 'sliding', 'pad', 'smoke', 'pad', 'soap', 'soap', 'soap', 'pad', 'soda', 'sofa', 'sofa', 'pad', 'pad', 'spray', 'stack', 'stack', 'stack', 'pad', 'stair', 'pad', 'pad', 'pad', 'pad', 'starbucks', 'pad', 'pad', 'step', 'pad', 'pad', 'storage', 'storage', 'storage', 'storage', 'storage', 'pad', 'pad', 'studio', 'stuffed', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'tea', 'pad', 'teddy', 'pad', 'pad', 'pad', 'pad', 'tissue', 'pad', 'toaster', 'pad', 'toilet', 'toilet', 'toilet', 'toilet', 'toilet', 'toilet', 'toilet', 'toilet', 'pad', 'pad', 'pad', 'pad', 'pad', 'towel', 'pad', 'toy', 'toy', 'traffic', 'trash', 'trash', 'trash', 'trash', 'pad', 'tray', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'tv', 'pad', 'pad', 'vacuum', 'pad', 'vending', 'pad', 'pad', 'wall', 'wall', 'wall', 'pad', 'wardrobe', 'wardrobe', 'pad', 'washing', 'washing', 'water', 'water', 'water', 'water', 'water', 'wet', 'pad', 'pad', 'whiteboard', 'pad', 'pad', 'pad', 'wood', 'pad', 'yoga', 'pad']
nr3d_vocab_set = ['above', 'air', 'airplane', 'alarm', 'allocentric', 'animal', 'apron', 'armchair', 'baby', 'back', 'backpack', 'bag', 'ball', 'banana', 'bananas', 'banister', 'banner', 'bar', 'barricade', 'base', 'basket', 'bath', 'bathrobe', 'bathroom', 'bathtub', 'battery', 'beachball', 'beam', 'beanbag', 'beans', 'bear', 'bed', 'beer', 'below', 'bench', 'between', 'bicycle', 'bike', 'bin', 'black', 'blackboard', 'blanket', 'blinds', 'block', 'blue', 'board', 'boards', 'boat', 'boiler', 'book', 'books', 'bookshelf', 'bookshelves', 'boots', 'bottle', 'bottles', 'bottom', 'bowl', 'box', 'boxes', 'breakfast', 'briefcase', 'broom', 'brown', 'brush', 'bucket', 'bulletin', 'bunk', 'button', 'cabinet', 'cabinets', 'cable', 'calendar', 'camera', 'can', 'candle', 'canopy', 'car', 'card', 'cardboard', 'carpet', 'carseat', 'cart', 'carton', 'case', 'cat', 'cd', 'ceiling', 'chain', 'chair', 'chairs', 'chandelier', 'changing', 'chest', 'cleaner', 'clock', 'closest', 'closet', 'cloth', 'clothes', 'clothing', 'coat', 'coatrack', 'coffee', 'color', 'column', 'compost', 'computer', 'conditioner', 'cone', 'container', 'containers', 'controller', 'cooker', 'cooking', 'cooler', 'copier', 'corner', 'costume', 'couch', 'counter', 'covered', 'crate', 'crib', 'cup', 'cups', 'curtain', 'curtains', 'curve', 'cushion', 'cushions', 'cutter', 'cutting', 'dart', 'decoration', 'desk', 'detector', 'detergent', 'diaper', 'dining', 'dinosaur', 'dish', 'dishwasher', 'dishwashing', 'dispenser', 'display', 'divider', 'doll', 'dollhouse', 'dolly', 'door', 'doorframe', 'doors', 'drawer', 'dress', 'dresser', 'drum', 'dryer', 'dryers', 'drying', 'duffel', 'dumbbell', 'dustpan', 'easel', 'electric', 'elevator', 'elliptical', 'end', 'envelope', 'eraser', 'exercise', 'exit', 'extinguisher', 'fan', 'farthest', 'faucet', 'file', 'fire', 'fireplace', 'flag', 'flip', 'floor', 'flops', 'flower', 'flowerpot', 'folded', 'folder', 'food', 'foosball', 'footrest', 'footstool', 'fountain', 'frame', 'fridge', 'front', 'frying', 'function', 'furnace', 'furniture', 'fuse', 'futon', 'garage', 'garbage', 'glass', 'globe', 'gold', 'golf', 'grab', 'green', 'grey', 'grocery', 'guitar', 'gun', 'hair', 'hamper', 'hand', 'handicap', 'handrail', 'hanger', 'hangers', 'hanging', 'hat', 'hatrack', 'head', 'headboard', 'headphones', 'heater', 'height', 'helmet', 'holder', 'hood', 'hose', 'hoverboard', 'humidifier', 'identity', 'ikea', 'instrument', 'ipad', 'iron', 'ironing', 'island', 'jacket', 'jar', 'kettle', 'keyboard', 'kitchen', 'kitchenaid', 'knife', 'ladder', 'lamp', 'laptop', 'large', 'laundry', 'ledge', 'leftmost', 'legs', 'length', 'light', 'lock', 'loft', 'long', 'loofa', 'lower', 'luggage', 'lunch', 'machine', 'machines', 'magazine', 'mail', 'mailbox', 'mailboxes', 'maker', 'map', 'massage', 'mat', 'mattress', 'medal', 'messenger', 'metronome', 'microwave', 'middle', 'mini', 'mirror', 'mitt', 'mixer', 'mobile', 'monitor', 'mouse', 'mouthwash', 'mug', 'music', 'nerf', 'night', 'nightstand', 'notepad', 'object', 'office', 'open', 'orange', 'organizer', 'orientation', 'ottoman', 'outlet', 'oven', 'package', 'pad', 'painting', 'pan', 'panel', 'pantry', 'pants', 'paper', 'papers', 'person', 'photo', 'piano', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'ping', 'pink', 'pipe', 'pipes', 'pitcher', 'pizza', 'plant', 'plastic', 'plate', 'plates', 'plunger', 'podium', 'pool', 'position', 'poster', 'pot', 'potted', 'power', 'printer', 'products', 'projector', 'pump', 'purple', 'purse', 'quadcopter', 'rack', 'radiator', 'rail', 'railing', 'range', 'recliner', 'recycling', 'red', 'refrigerator', 'relations', 'remote', 'rice', 'rightmost', 'rod', 'roll', 'rolled', 'rolls', 'roomba', 'rope', 'round', 'rug', 'salt', 'santa', 'scale', 'scanner', 'screen', 'seat', 'seating', 'set', 'sewing', 'shampoo', 'sheets', 'shelf', 'shirt', 'shoe', 'shoes', 'shopping', 'short', 'shorts', 'shower', 'shredder', 'sign', 'sink', 'size', 'sliding', 'slippers', 'sliver', 'small', 'smoke', 'soap', 'sock', 'soda', 'sofa', 'speaker', 'sponge', 'spray', 'stack', 'stair', 'staircase', 'stairs', 'stall', 'stand', 'stapler', 'starbucks', 'station', 'statue', 'step', 'sticker', 'stool', 'storage', 'stove', 'stream', 'strip', 'structure', 'studio', 'stuffed', 'suitcase', 'suitcases', 'support', 'supported', 'sweater', 'swiffer', 'switch', 'table', 'tall', 'tank', 'tap', 'tape', 'tea', 'teapot', 'teddy', 'telephone', 'telescope', 'thermostat', 'tire', 'tissue', 'toaster', 'toilet', 'toiletry', 'toolbox', 'toothbrush', 'toothpaste', 'top', 'towel', 'towels', 'tower', 'toy', 'traffic', 'trash', 'tray', 'treadmill', 'tripod', 'trolley', 'trunk', 'tube', 'tupperware', 'tv', 'umbrella', 'urinal', 'vacuum', 'valve', 'vanity', 'vase', 'vending', 'vent', 'wall', 'walls', 'wardrobe', 'washcloth', 'washing', 'water', 'wet', 'wheel', 'white', 'whiteboard', 'window', 'windowsill', 'wood', 'workbench', 'yellow', 'yoga']




# sr3d
object_class2 = ['table', 'mattress', 'airplane', 'alarm', 'clock', 'armchair', 'station', 'mobile', 'backpack', 'bag', 'beans', 'ball', 'holder', 'bananas', 'banister', 'banner', 'bar', 'barricade', 'barrier', 'cap', 'basket', 'products', 'walls', 'bathrobe', 'cabinet', 'counter', 'stall', 'door', 'vanity', 'bathtub', 'jar', 'beachball', 'chair', 'bear', 'bed', 'bedframe', 'bottles', 'bench', 'bicycle', 'lock', 'pump', 'bin', 'binder', 'binders', 'blackboard', 'eraser', 'blanket', 'blinds', 'block', 'board', 'boards', 'boat', 'boiler', 'book', 'rack', 'books', 'bookshelf', 'bookshelves', 'boots', 'bottle', 'bottles', 'bowl', 'box', 'boxes', 'paper', 'bar', 'briefcase', 'broom', 'bucket', 'buddha', 'board', 'bed', 'bycicle', 'cabinet', 'door', 'doors', 'cabinets', 'cable', 'calendar', 'camera', 'can', 'candle', 'canopy', 'cap', 'car', 'card', 'cardboard', 'carpet', 'carseat', 'cart', 'carton', 'case', 'bottles', 'box', 'case', 'cases', 'ceiling', 'fan', 'lamp', 'light', 'centerpiece', 'chain', 'chair', 'chandelier', 'station', 'chest', 'clip', 'clock', 'closet', 'ceiling', 'door', 'doorframe', 'doors', 'floor', 'rod', 'shelf', 'wall', 'walls', 'wardrobe', 'cloth', 'clothes', 'dryer', 'dryers', 'hanger', 'hangers', 'clothing', 'rack', 'coat', 'rack', 'coatrack', 'bag', 'box', 'kettle', 'maker', 'mug', 'table', 'column', 'bin', 'tower', 'bottle', 'cone', 'bottle', 'container', 'controller', 'pan', 'pot', 'cooler', 'copier', 'bag', 'costume', 'couch', 'cushions', 'counter', 'cover', 'box', 'crate', 'crib', 'crutches', 'cup', 'cups', 'curtain', 'rod', 'curtains', 'cushion', 'board', 'board', 'decoration', 'desk', 'lamp', 'bin', 'table', 'rack', 'dishwasher', 'bottle', 'dispenser', 'display', 'case', 'rack', 'sign', 'divider', 'doll', 'dollhouse', 'dolly', 'door', 'wall', 'doorframe', 'doors', 'drawer', 'rack', 'dresser', 'set', 'sheets', 'rack', 'bag', 'dumbbell', 'plates', 'dumbell', 'dustpan', 'easel', 'panel', 'elevator', 'button', 'machine', 'table', 'envelope', 'ball', 'bike', 'machine', 'sign', 'fan', 'faucet', 'cabinet', 'cabinets', 'organizer', 'light', 'alarm', 'extinguisher', 'hose', 'sprinkler', 'fireplace', 'fish', 'flag', 'flops', 'floor', 'stand', 'flowerpot', 'boxes', 'chair', 'chairs', 'ladder', 'table', 'folder', 'bag', 'container', 'display', 'table', 'footrest', 'footstool', 'frame', 'pan', 'furnace', 'furniture', 'box', 'futon', 'wheel', 'door', 'bag', 'glass', 'doors', 'globe', 'bag', 'bar', 'bag', 'guitar', 'case', 'brush', 'dryer', 'hamper', 'dryer', 'rail', 'dispenser', 'towel', 'bar', 'handrail', 'hanging', 'hat', 'hatrack', 'headboard', 'headphones', 'heater', 'helmet', 'hose', 'hoverboard', 'humidifier', 'bag', 'case', 'ipad', 'iron', 'board', 'jacket', 'jar', 'box', 'kettle', 'keyboard', 'piano', 'kinect', 'apron', 'cabinet', 'cabinets', 'counter', 'island', 'mixer', 'mixer', 'block', 'ladder', 'lamp', 'base', 'laptop', 'bag', 'basket', 'detergent', 'hamper', 'ledge', 'legs', 'light', 'switch', 'bed', 'loofa', 'lotion', 'bottle', 'luggage', 'rack', 'stand', 'box', 'machine', 'magazine', 'rack', 'mail', 'bin', 'tray', 'trays', 'mailbox', 'mailboxes', 'map', 'chair', 'mat', 'mattress', 'medal', 'center', 'bag', 'metronome', 'microwave', 'fridge', 'mirror', 'doors', 'monitor', 'mop', 'mouse', 'bottle', 'mug', 'book', 'stand', 'gun', 'lamp', 'light', 'nightstand', 'notepad', 'object', 'chair', 'cabinet', 'organizer', 'shelf', 'ottoman', 'oven', 'mitt', 'painting', 'shelf', 'wall', 'walls', 'pants', 'paper', 'bag', 'cutter', 'organizer', 'shredder', 'towel', 'dispenser', 'roll', 'rolls', 'tray', 'papers', 'holder', 'person', 'photo', 'piano', 'bench', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'paddle', 'table', 'pipe', 'pipes', 'pitcher', 'box', 'boxes', 'plant', 'bin', 'container', 'containers', 'bin', 'plate', 'plates', 'platform', 'plunger', 'podium', 'table', 'postcard', 'poster', 'cutter', 'printer', 'tube', 'pot', 'plant', 'outlet', 'strip', 'printer', 'projector', 'screen', 'purse', 'quadcopter', 'rack', 'stand', 'radiator', 'rail', 'railing', 'hood', 'chair', 'bin', 'refrigerator', 'remote', 'cooker', 'chair', 'rod', 'poster', 'roomba', 'rope', 'table', 'rug', 'salt', 'santa', 'scale', 'scanner', 'screen', 'seat', 'seating', 'machine', 'shampoo', 'bottle', 'cream', 'shelf', 'shirt', 'shoe', 'rack', 'shoes', 'bag', 'shorts', 'shower', 'valve', 'curtain', 'rod', 'door', 'doors', 'handle', 'floor', 'head', 'wall', 'walls', 'shredder', 'table', 'sign', 'sink', 'bag', 'door', 'door', 'slipper', 'slippers', 'detector', 'soap', 'bar', 'bottle', 'dish', 'dispenser', 'sock', 'socks', 'can', 'stream', 'sofa', 'bed', 'chair', 'speaker', 'sponge', 'bottle', 'chairs', 'cups', 'chairs', 'cups', 'stair', 'rail', 'staircase', 'stairs', 'stand', 'stapler', 'star', 'cup', 'statue', 'step', 'stool', 'stepladder', 'stepstool', 'stick', 'sticker', 'stool', 'stools', 'bin', 'box', 'container', 'organizer', 'shelf', 'stove', 'stovetop', 'structure', 'light', 'animal', 'subwoofer', 'suitcase', 'suitcases', 'sweater', 'swiffer', 'switch', 'table', 'lamp', 'tank', 'tap', 'tape', 'kettle', 'teapot', 'bear', 'telephone', 'telescope', 'racket', 'bag', 'thermos', 'thermostat', 'tire', 'box', 'toaster', 'oven', 'toilet', 'brush', 'button', 'paper', 'dispenser', 'holder', 'package', 'rolls', 'dispenser', 'toiletry', 'toolbox', 'toothbrush', 'toothpaste', 'towel', 'rack', 'towels', 'dinosaur', 'piano', 'cone', 'bag', 'bin', 'cabinet', 'can', 'tray', 'rack', 'treadmill', 'tripod', 'trolley', 'trunk', 'tube', 'tupperware', 'tv', 'stand', 'umbrella', 'urinal', 'cleaner', 'vase', 'machine', 'vent', 'wall', 'hanging', 'lamp', 'rack', 'wardrobe', 'cabinet', 'closet', 'washcloth', 'machine', 'machines', 'bottle', 'cooler', 'fountain', 'heater', 'pitcher', 'softener', 'sign', 'wheel', 'whiteboard', 'eraser', 'wig', 'window', 'windowsill', 'wood', 'beam', 'workbench', 'controller', 'mat', 'pad']
my_function2 = ['air', 'air', 'pad', 'pad', 'alarm', 'pad', 'baby', 'baby', 'pad', 'pad', 'bag', 'pad', 'banana', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'baseball', 'pad', 'bath', 'bath', 'pad', 'bathroom', 'bathroom', 'bathroom', 'bathroom', 'bathroom', 'pad', 'battery', 'pad', 'beanbag', 'pad', 'pad', 'pad', 'beer', 'pad', 'pad', 'bike', 'bike', 'pad', 'pad', 'pad', 'pad', 'blackboard', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'book', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'boxes', 'breakfast', 'pad', 'pad', 'pad', 'pad', 'bulletin', 'bunk', 'pad', 'pad', 'cabinet', 'cabinet', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'case', 'cat', 'cd', 'cd', 'pad', 'ceiling', 'ceiling', 'ceiling', 'pad', 'pad', 'pad', 'pad', 'changing', 'pad', 'pad', 'pad', 'pad', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'closet', 'pad', 'pad', 'clothes', 'clothes', 'clothes', 'clothes', 'pad', 'clothing', 'pad', 'coat', 'pad', 'coffee', 'coffee', 'coffee', 'coffee', 'coffee', 'coffee', 'pad', 'compost', 'computer', 'conditioner', 'pad', 'contact', 'pad', 'pad', 'cooking', 'cooking', 'pad', 'pad', 'cosmetic', 'pad', 'pad', 'couch', 'pad', 'pad', 'covered', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'curtain', 'pad', 'pad', 'cutting', 'dart', 'pad', 'pad', 'desk', 'diaper', 'dining', 'dish', 'pad', 'dishwashing', 'pad', 'pad', 'display', 'display', 'display', 'pad', 'pad', 'pad', 'pad', 'pad', 'door', 'pad', 'pad', 'pad', 'dress', 'pad', 'drum', 'dryer', 'drying', 'duffel', 'pad', 'dumbbell', 'pad', 'pad', 'pad', 'electric', 'pad', 'elevator', 'elliptical', 'end', 'pad', 'exercise', 'exercise', 'exercise', 'exit', 'pad', 'pad', 'file', 'file', 'file', 'film', 'fire', 'fire', 'fire', 'fire', 'pad', 'pad', 'pad', 'flip', 'pad', 'flower', 'pad', 'folded', 'folded', 'folded', 'folded', 'folded', 'pad', 'food', 'food', 'food', 'foosball', 'pad', 'pad', 'pad', 'frying', 'pad', 'pad', 'fuse', 'pad', 'gaming', 'garage', 'garbage', 'pad', 'glass', 'pad', 'golf', 'grab', 'grocery', 'pad', 'guitar', 'hair', 'hair', 'pad', 'hand', 'hand', 'hand', 'hand', 'handicap', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'ikea', 'instrument', 'pad', 'pad', 'ironing', 'pad', 'pad', 'jewelry', 'pad', 'pad', 'keyboard', 'pad', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchenaid', 'knife', 'pad', 'pad', 'lamp', 'pad', 'laundry', 'laundry', 'laundry', 'laundry', 'pad', 'pad', 'pad', 'light', 'loft', 'pad', 'pad', 'lotion', 'pad', 'luggage', 'luggage', 'lunch', 'pad', 'pad', 'magazine', 'pad', 'mail', 'mail', 'mail', 'pad', 'pad', 'pad', 'massage', 'pad', 'pad', 'pad', 'media', 'messenger', 'pad', 'pad', 'mini', 'pad', 'mirror', 'pad', 'pad', 'pad', 'mouthwash', 'pad', 'music', 'music', 'nerf', 'night', 'night', 'pad', 'pad', 'pad', 'office', 'open', 'pad', 'organizer', 'pad', 'pad', 'oven', 'pad', 'pantry', 'pantry', 'pantry', 'pad', 'pad', 'paper', 'paper', 'paper', 'paper', 'paper', 'paper', 'paper', 'papertowel', 'paper', 'pad', 'pen', 'pad', 'pad', 'pad', 'piano', 'pad', 'pad', 'pad', 'pad', 'pad', 'ping', 'ping', 'pad', 'pad', 'pad', 'pizza', 'pizza', 'pad', 'plastic', 'plastic', 'plastic', 'plastic', 'pad', 'pad', 'pad', 'pad', 'pad', 'pool', 'pad', 'pad', 'poster', 'poster', 'poster', 'pad', 'potted', 'power', 'power', 'pad', 'pad', 'projector', 'pad', 'pad', 'pad', 'rack', 'pad', 'pad', 'pad', 'range', 'recliner', 'recycling', 'pad', 'pad', 'rice', 'rocking', 'pad', 'rolled', 'pad', 'pad', 'round', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'sewing', 'pad', 'shampoo', 'shaving', 'pad', 'pad', 'pad', 'shoe', 'pad', 'shopping', 'pad', 'pad', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'shower', 'pad', 'side', 'pad', 'pad', 'sleeping', 'sliding', 'sliding', 'pad', 'pad', 'smoke', 'pad', 'soap', 'soap', 'soap', 'soap', 'pad', 'pad', 'soda', 'soda', 'pad', 'sofa', 'sofa', 'pad', 'pad', 'spray', 'stack', 'stack', 'stack', 'stacks', 'pad', 'stair', 'pad', 'pad', 'pad', 'pad', 'pad', 'starbucks', 'pad', 'pad', 'step', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'storage', 'storage', 'storage', 'storage', 'storage', 'pad', 'pad', 'pad', 'studio', 'stuffed', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'table', 'pad', 'pad', 'pad', 'tea', 'pad', 'teddy', 'pad', 'pad', 'tennis', 'tennis', 'pad', 'pad', 'pad', 'tissue', 'pad', 'toaster', 'pad', 'toilet', 'toilet', 'toilet', 'toilet', 'toilet', 'toilet', 'toilet', 'toilet', 'pad', 'pad', 'pad', 'pad', 'pad', 'towel', 'pad', 'toy', 'toy', 'traffic', 'trash', 'trash', 'trash', 'trash', 'pad', 'tray', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'tv', 'pad', 'pad', 'vacuum', 'pad', 'vending', 'pad', 'pad', 'wall', 'wall', 'wall', 'pad', 'wardrobe', 'wardrobe', 'pad', 'washing', 'washing', 'water', 'water', 'water', 'water', 'water', 'water', 'wet', 'pad', 'pad', 'whiteboard', 'pad', 'pad', 'pad', 'pad', 'wood', 'pad', 'xbox', 'yoga', 'pad']
sr3d_vocab_set = ['above', 'air', 'airplane', 'alarm', 'allocentric', 'animal', 'apron', 'armchair', 'baby', 'back', 'backpack', 'bag', 'ball', 'banana', 'bananas', 'banister', 'banner', 'bar', 'barricade', 'barrier', 'base', 'baseball', 'basket', 'bath', 'bathrobe', 'bathroom', 'bathtub', 'battery', 'beachball', 'beam', 'beanbag', 'beans', 'bear', 'bed', 'bedframe', 'beer', 'below', 'bench', 'between', 'bicycle', 'bike', 'bin', 'binder', 'binders', 'black', 'blackboard', 'blanket', 'blinds', 'block', 'blue', 'board', 'boards', 'boat', 'boiler', 'book', 'books', 'bookshelf', 'bookshelves', 'boots', 'bottle', 'bottles', 'bottom', 'bowl', 'box', 'boxes', 'breakfast', 'briefcase', 'broom', 'brown', 'brush', 'bucket', 'buddha', 'bulletin', 'bunk', 'button', 'bycicle', 'cabinet', 'cabinets', 'cable', 'calendar', 'camera', 'can', 'candle', 'canopy', 'cap', 'car', 'card', 'cardboard', 'carpet', 'carseat', 'cart', 'carton', 'case', 'cases', 'cat', 'cd', 'ceiling', 'center', 'centerpiece', 'chain', 'chair', 'chairs', 'chandelier', 'changing', 'chest', 'cleaner', 'clip', 'clock', 'closest', 'closet', 'cloth', 'clothes', 'clothing', 'coat', 'coatrack', 'coffee', 'color', 'column', 'compost', 'computer', 'conditioner', 'cone', 'contact', 'container', 'containers', 'controller', 'cooker', 'cooking', 'cooler', 'copier', 'corner', 'cosmetic', 'costume', 'couch', 'counter', 'cover', 'covered', 'crate', 'cream', 'crib', 'crutches', 'cup', 'cups', 'curtain', 'curtains', 'curve', 'cushion', 'cushions', 'cutter', 'cutting', 'dart', 'decoration', 'desk', 'detector', 'detergent', 'diaper', 'dining', 'dinosaur', 'dish', 'dishwasher', 'dishwashing', 'dispenser', 'display', 'divider', 'doll', 'dollhouse', 'dolly', 'door', 'doorframe', 'doors', 'drawer', 'dress', 'dresser', 'drum', 'dryer', 'dryers', 'drying', 'duffel', 'dumbbell', 'dumbell', 'dustpan', 'easel', 'electric', 'elevator', 'elliptical', 'end', 'envelope', 'eraser', 'exercise', 'exit', 'extinguisher', 'fan', 'farthest', 'faucet', 'file', 'film', 'fire', 'fireplace', 'fish', 'flag', 'flip', 'floor', 'flops', 'flower', 'flowerpot', 'folded', 'folder', 'food', 'foosball', 'footrest', 'footstool', 'fountain', 'frame', 'fridge', 'front', 'frying', 'function', 'furnace', 'furniture', 'fuse', 'futon', 'gaming', 'garage', 'garbage', 'glass', 'globe', 'gold', 'golf', 'grab', 'green', 'grey', 'grocery', 'guitar', 'gun', 'hair', 'hamper', 'hand', 'handicap', 'handle', 'handrail', 'hanger', 'hangers', 'hanging', 'hat', 'hatrack', 'head', 'headboard', 'headphones', 'heater', 'height', 'helmet', 'holder', 'hood', 'hose', 'hoverboard', 'humidifier', 'identity', 'ikea', 'instrument', 'ipad', 'iron', 'ironing', 'island', 'jacket', 'jar', 'jewelry', 'kettle', 'keyboard', 'kinect', 'kitchen', 'kitchenaid', 'knife', 'ladder', 'lamp', 'laptop', 'large', 'laundry', 'ledge', 'leftmost', 'legs', 'length', 'light', 'lock', 'loft', 'long', 'loofa', 'lotion', 'lower', 'luggage', 'lunch', 'machine', 'machines', 'magazine', 'mail', 'mailbox', 'mailboxes', 'maker', 'map', 'massage', 'mat', 'mattress', 'medal', 'media', 'messenger', 'metronome', 'microwave', 'middle', 'mini', 'mirror', 'mitt', 'mixer', 'mobile', 'monitor', 'mop', 'mouse', 'mouthwash', 'mug', 'music', 'nerf', 'night', 'nightstand', 'notepad', 'object', 'office', 'open', 'orange', 'organizer', 'orientation', 'ottoman', 'outlet', 'oven', 'package', 'pad', 'paddle', 'painting', 'pan', 'panel', 'pantry', 'pants', 'paper', 'papers', 'papertowel', 'pen', 'person', 'photo', 'piano', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'ping', 'pink', 'pipe', 'pipes', 'pitcher', 'pizza', 'plant', 'plastic', 'plate', 'plates', 'platform', 'plunger', 'podium', 'pool', 'position', 'postcard', 'poster', 'pot', 'potted', 'power', 'printer', 'products', 'projector', 'pump', 'purple', 'purse', 'quadcopter', 'rack', 'racket', 'radiator', 'rail', 'railing', 'range', 'recliner', 'recycling', 'red', 'refrigerator', 'relations', 'remote', 'rice', 'rightmost', 'rocking', 'rod', 'roll', 'rolled', 'rolls', 'roomba', 'rope', 'round', 'rug', 'salt', 'santa', 'scale', 'scanner', 'screen', 'seat', 'seating', 'set', 'sewing', 'shampoo', 'shaving', 'sheets', 'shelf', 'shirt', 'shoe', 'shoes', 'shopping', 'short', 'shorts', 'shower', 'shredder', 'side', 'sign', 'sink', 'size', 'sleeping', 'sliding', 'slipper', 'slippers', 'sliver', 'small', 'smoke', 'soap', 'sock', 'socks', 'soda', 'sofa', 'softener', 'speaker', 'sponge', 'spray', 'sprinkler', 'stack', 'stacks', 'stair', 'staircase', 'stairs', 'stall', 'stand', 'stapler', 'star', 'starbucks', 'station', 'statue', 'step', 'stepladder', 'stepstool', 'stick', 'sticker', 'stool', 'stools', 'storage', 'stove', 'stovetop', 'stream', 'strip', 'structure', 'studio', 'stuffed', 'subwoofer', 'suitcase', 'suitcases', 'support', 'supported', 'sweater', 'swiffer', 'switch', 'table', 'tall', 'tank', 'tap', 'tape', 'tea', 'teapot', 'teddy', 'telephone', 'telescope', 'tennis', 'thermos', 'thermostat', 'tire', 'tissue', 'toaster', 'toilet', 'toiletry', 'toolbox', 'toothbrush', 'toothpaste', 'top', 'towel', 'towels', 'tower', 'toy', 'traffic', 'trash', 'tray', 'trays', 'treadmill', 'tripod', 'trolley', 'trunk', 'tube', 'tupperware', 'tv', 'umbrella', 'urinal', 'vacuum', 'valve', 'vanity', 'vase', 'vending', 'vent', 'wall', 'walls', 'wardrobe', 'washcloth', 'washing', 'water', 'wet', 'wheel', 'white', 'whiteboard', 'wig', 'window', 'windowsill', 'wood', 'workbench', 'xbox', 'yellow', 'yoga']
class R2G(nn.Module):
    """
    A neural listener for segmented 3D scans based on graph-convolutions.
    """

    def __init__(self,
                 args,
                 object_encoder,
                 token_embed,
                 object_language_clf = None,
                 object_clf=None,
                 language_clf=None,
                 instruction_clf = None,
                 graph_encoder = None,
                 property_tokenid = None,
                 concept_vocab = None,
                 relation_num = None,
                 object_semantic_filter_index = None,
                 language_relation_clf = None,
                 language_encoder = None,
                 concept_vocab_seg = None,
                 num_node_properties = None,
                 concept_vocab_set = None):
        """
        Parameters have same meaning as in Base3DListener.

        @param args: the parsed arguments
        @param object_encoder: encoder for each segmented object ([point-cloud, color]) of a scan
        @param language_encoder: encoder for the referential utterance
        @param graph_encoder: the graph net encoder (DGCNN is the used graph encoder)
        given geometry is the referred one (typically this is an MLP).
        @param object_clf: classifies the object class of the segmented (raw) object (e.g., is it a chair? or a bed?)
        @param language_clf: classifies the target-class type referred in an utterance.
        @param object_language_clf: given a fused feature of language and geometry, captures how likely it is that the
        """

        super().__init__()
        self.args = args

        self.property_embedding = property_tokenid
        self.concept_vocab = concept_vocab      # object_semantic_token + color_semantic_token + relation_semantic_tokenid
        self.num_property = len(property_tokenid)
        self.object_semantic_filter_index = object_semantic_filter_index
        self.concept_vocab_seg = concept_vocab_seg
        self.concept_vocab_set = concept_vocab_set

        # Encoders
        self.object_encoder = object_encoder

        ## token embedder
        self.token_embed = token_embed

        # Classifier heads
        self.object_clf = object_clf
        self.language_clf = language_clf
        # self.instruction_clf = instruction_clf
        
        # self.obj_ref = object_language_clf
        
        ## relation pred heads
        if args.relation_pred:
            print("----------add relation_pred head")
            self.relation_pred = Relation_Estimate(n_class=relation_num , d_model = args.object_latent_dim * 2, n_head = 2)

        self.attribute_pred = Attr_Estimate(n_class=13 , obj_feat = args.object_latent_dim, n_head = 2)
        
        ## relation retrieval heads
        if args.relation_retrieval:
            print("--------add relation retrieval head")
            if 'sr3d' in args.referit3D_file:
                self.mode = 'sr3d'
            elif 'nr3d' in args.referit3D_file:
                self.mode = 'nr3d'

        # ## NSM 
        if args.language_relation_alpha + args.lang_cls_alpha + args.instruction_cls_alpha > 0:    
            # if args.language_relation_alpha == 0:
            #     language_relation_clf = None
            # if args.lang_cls_alpha == 0:
            #     language_clf = None
            if args.instruction_cls_alpha == 0:
                instruction_clf = None
            self.nsm = NSM(input_size = args.word_embedding_dim, 
                            num_node_properties = num_node_properties, 
                            num_instructions = 5, 
                            description_hidden_size = 512
                            # language_clf = language_clf,
                            # lang_relation_classify = language_relation_clf,
                            # instruction_clf = instruction_clf
                            )
        else:
            self.nsm = NSM(input_size = args.word_embedding_dim, 
                            num_node_properties = num_node_properties, 
                            num_instructions = 5, 
                            description_hidden_size = 512,
                            )



    def __call__(self, batch: dict) -> dict:
        result = defaultdict(lambda: None)
        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim
        scene_feature =  get_siamese_features(self.object_encoder, batch['scene_pc'].cuda().unsqueeze(1),
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim
        # print(objects_features)
        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        # Get feature for utterance
        # language_embedding = self.token_embed(batch['tokens']).float()## B X n_token X embedding
        language_embedding = batch['token_embedding'].cuda()


        # # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(lang_features)

        # set the NSM attribute
        property_embedding =  self.token_embed(torch.LongTensor(self.property_embedding).cuda()).float()
        concept_vocab = self.token_embed(torch.LongTensor(self.concept_vocab).cuda()).float()
        concept_vocab_set = self.token_embed(torch.LongTensor(self.concept_vocab_set).cuda()).float()
        #   B x N x 524    *    524 x embedding        ->          B X N X embedding, (B * 52 * 300), class semantic
        if self.object_semantic_filter_index is not None:
            object_semantic_prob = F.softmax(result['class_logits'][:, :, self.object_semantic_filter_index], dim =-1) @ concept_vocab[:self.concept_vocab_seg[0]]
        else:
            # object_semantic_prob = F.softmax(result['class_logits'], dim =-1) @ concept_vocab[:self.concept_vocab_seg[0]]
            # function_semantic_prob = F.softmax(result['class_logits'], dim =-1) @ concept_vocab[self.concept_vocab_seg[1]:self.concept_vocab_seg[2]]
            object_semantic_prob = batch['gt_class'][:, :, :-1].cuda() @ concept_vocab[:self.concept_vocab_seg[0]]
            function_semantic_prob = batch['gt_class'][:, :, :-1].cuda() @ concept_vocab[self.concept_vocab_seg[1]:self.concept_vocab_seg[2]]
        color_onehot = torch.Tensor(batch['color_onehot']).cuda()   # B X N X one-hot, (B * 52 * 13)
        #  B x N x 13       * 13 x embedding      ->      B X N X embedding, (B * 52 * 300)
        color_semantic_prob = color_onehot @ concept_vocab[self.concept_vocab_seg[0]:self.concept_vocab_seg[1]]

        ls_logits, tl_logits, losh_logits = Attr_Compute(self.mode, batch, batch['gt_class'], batch['object_mask'], batch['context_size'])

        # ls_logits, tl_logits, mc_logits, tb_logits, lr_logits, losh_logits = self.attribute_pred(obj_feature = torch.cat([scene_feature, objects_features], dim = 1), \
        # obj_center = torch.cat([batch['scene_center'].cuda().unsqueeze(1), batch['position_features'].cuda()], dim = 1), \
        # obj_size = torch.cat([batch['scene_size'].cuda().unsqueeze(1), batch['size_feature'].cuda()], dim =1), object_mask = batch['object_mask'].cuda()) # Bx N x N xk

        lr_logits, curve_logits = self.attribute_pred(obj_feature = torch.cat([scene_feature, objects_features], dim = 1), \
        obj_center = torch.cat([batch['scene_center'].cuda().unsqueeze(1), batch['obj_position'].cuda()], dim = 1), \
        obj_size = torch.cat([batch['scene_size'].cuda().unsqueeze(1), batch['obj_size'].cuda()], dim =1), object_mask = batch['object_mask'].cuda()) # Bx N x N xk

        # obj_attr = torch.cat([F.softmax(ls_logits, dim = -1), F.softmax(tl_logits, dim = -1), F.softmax(mc_logits, dim = -1), F.softmax(tb_logits, dim = -1), F.softmax(lr_logits, dim = -1), F.softmax(losh_logits, dim = -1)], dim = -1)
        #          B x N x k       K x embedding 
        ls_attr = ls_logits.float() @ concept_vocab[self.concept_vocab_seg[2]:self.concept_vocab_seg[3]].unsqueeze(0)
        tl_attr = tl_logits.float() @ concept_vocab[self.concept_vocab_seg[3]:self.concept_vocab_seg[4]].unsqueeze(0)
        mc_attr = batch['mc_attr'] @ concept_vocab[self.concept_vocab_seg[4]:self.concept_vocab_seg[5]].unsqueeze(0)
        tb_attr = batch['tb_attr'] @ concept_vocab[self.concept_vocab_seg[5]:self.concept_vocab_seg[6]].unsqueeze(0)
        lr_attr = F.softmax(lr_logits, dim = -1) @ concept_vocab[self.concept_vocab_seg[6]:self.concept_vocab_seg[7]].unsqueeze(0)
        losh_attr =losh_logits.float() @ concept_vocab[self.concept_vocab_seg[7]:self.concept_vocab_seg[8]].unsqueeze(0)
        curve_attr = F.softmax(curve_logits, dim = -1) @ concept_vocab[self.concept_vocab_seg[8]:self.concept_vocab_seg[9]].unsqueeze(0)

        obj_attr = torch.cat([ls_logits, tl_logits, batch['mc_attr'], batch['tb_attr'], lr_logits, losh_logits], dim = -1)


        node_attr = torch.cat([object_semantic_prob.unsqueeze(2), color_semantic_prob.unsqueeze(2), function_semantic_prob.unsqueeze(2), ls_attr.unsqueeze(2), tl_attr.unsqueeze(2), mc_attr.unsqueeze(2), tb_attr.unsqueeze(2), lr_attr.unsqueeze(2), losh_attr.unsqueeze(2), curve_attr.unsqueeze(2)], 2) # B X N X embedding -> B X N X L+1 X embedding, (B * 52 * 2 * 300) 
        edge_prob = edge_prob_logits = None


        # B x N x N x prob_softmax   * probmax x embedding     ->     B X N X N X onehot_dim -> B X N X N X embedding, (B * n * n * 300)
        if self.args.relation_pred:
            edge_prob, edge_prob_logits = self.relation_pred(dis_vec = batch['edge_vector'].cuda(), obj_feature = objects_features, object_mask = batch['object_mask'].cuda()) # Bx N x N xk
            #          B x N x N x k       K x embedding 
            edge_attr = edge_prob_logits @ concept_vocab[self.concept_vocab_seg[9]:]
        elif self.args.relation_retrieval:
            edge_prob_logits = SR_Retrieval(self.mode, batch['gt_class'], batch['edge_attr'],  torch.Tensor(batch['edge_distance']), batch['object_mask'], batch['context_size']).cuda().double()
            # edge_prob_logits = SR_Retrieval(self.mode, result['class_logits'], batch['edge_attr'],  torch.Tensor(batch['edge_distance']), batch['object_mask'], batch['context_size']).cuda().float()
            #          B x N x N x k       K x embedding 
            # edge_attr = F.softmax(edge_prob_logits, dim =-1) @ concept_vocab[self.concept_vocab_seg[2]:]
            edge_attr = edge_prob_logits.cuda().float() @ concept_vocab[self.concept_vocab_seg[9]:]
            result['edge_prob_logits'] = edge_prob_logits
        else:
            # edge_attr = F.softmax(batch['edge_attr'].cuda().float(), dim =-1) @ concept_vocab[self.concept_vocab_seg[2]:]
            edge_attr = batch['edge_attr'].cuda().float() @ concept_vocab[self.concept_vocab_seg[9]:]
            edge_prob_logits = batch['edge_attr'].cuda().float()
        
        if not self.args.language_relation_alpha > 0:
            # batch['lang_mask'][np.isinf(batch['lang_mask'])] = 1
            # batch['lang_mask'] = 1 - batch['lang_mask']
            final_node_distribution, encoded_questions, simi, simi_index, ins_simi, ins_index, lang, attention, instruction_prop = self.nsm(node_attr = node_attr, edge_attr = edge_attr, description = language_embedding, property_embeddings = property_embedding, node_mask = batch['object_mask'].cuda(), context_size = batch['context_size'].cuda(), lang_mask = batch['lang_mask'].cuda().float(), language_len = batch['tokens_len'], concept_vocab_set = concept_vocab_set)
        else:
            final_node_distribution, encoded_questions, prob , all_instruction, anchor_logits, lang_relation_logits, target_logits = self.nsm(node_attr = node_attr, description = language_embedding, property_embeddings = property_embedding, node_mask = batch['object_mask'].cuda(), context_size = batch['context_size'].cuda(), concept_vocab_set = concept_vocab_set)
            
            
        final_node_distribution_mask = final_node_distribution + batch['object_mask'].cuda()
        final_features = final_node_distribution_mask
        # print(final_features[:2], torch.topk(final_features[:2], k =5))
        # print(batch['object_mask'][:2])
        result['obj_attr'] = obj_attr
        if self.args.language_relation_alpha > 0:
            result['lang_relation_logits'] = lang_relation_logits

        # Classify the target instance label based on the text
        if self.args.lang_cls_alpha > 0:
            result['lang_logits'] = self.language_clf(encoded_questions)
            # result['lang_logits'] = target_logits

        if self.args.instruction_cls_alpha > 0:
            # result['instruction_logits'] = self.instruction_clf(all_instruction[0])
            result['instruction_logits'] = anchor_logits

        result['logits'] = final_features
        result['edge_prob'] = edge_prob_logits
        result['attention'] = attention
        result['token'] = lang
        result['instruction_prop'] = instruction_prop

        result['simi'] = simi
        result['simi_index'] = simi_index

        
        result['ins_simi'] = ins_simi
        result['ins_simi_index'] = ins_index

        if edge_prob is not None:
            result['sr_prob'] = edge_prob

        return result

def create_r2g_net(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int, class_to_index: dict) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """
    # convenience
    geo_out_dim = args.object_latent_dim
    lang_out_dim = args.language_latent_dim

    if 'sr3d' in args.referit3D_file:
        object_class = object_class2
        my_function = my_function2
        concept_vocab = sr3d_vocab_set
    elif 'nr3d' in args.referit3D_file:
        object_class = object_class1
        my_function = my_function1
        concept_vocab = nr3d_vocab_set

    # prepare the properties and concept token ids 
    object_semantic_filtertoken = vocab.encode(object_class, add_begin_end = False)[0][:-1] # 525 object-semantic-label; -1 is the pad class
    # object_semantic_filter_index = [index for index, value in enumerate(object_semantic_token) if value != 3]#183
    # object_semantic_filtertoken =  [value for index, value in enumerate(object_semantic_token) if value != 3]#183

    concept_vocab_set = vocab.encode(concept_vocab, add_begin_end = False)[0]

    color_semantic = ['white', 'blue', 'brown', 'black', 'red', 'green', 'grey', 'yellow', 'purple', 'sliver', 'gold', 'pink', 'orange']
    color_semantic_token = vocab.encode(color_semantic, add_begin_end = False)[0] # 13 color-semantic-label
    relation_semantic = ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported', 'between', 'allocentric']
    # if args.relation_retrieval and args.language_relation_alpha == 0:
    #     relation_semantic = ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported']
    # elif args.relation_pred or args.language_relation_alpha > 0:
    #     relation_semantic = ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported', 'between', 'allocentric']#support
    # else:
    #     # relation_semantic = ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported']
    #     relation_semantic = ['above', 'below', 'front', 'back', 'farthest', 'closest', 'support', 'supported', 'between', 'allocentric']#support
    relation_semantic_tokenid = vocab.encode(relation_semantic, add_begin_end = False)[0]   # 11 relation-semantic-label

    size = ['large', 'small']
    size_token = vocab.encode(size, add_begin_end = False)[0]   # 11 relation-semantic-label
    height = ['tall', 'lower']
    height_token = vocab.encode(height, add_begin_end = False)[0]   # 11 relation-semantic-label
    position = ['middle', 'corner']
    position_token = vocab.encode(position, add_begin_end = False)[0]   # 11 relation-semantic-label
    orientation = ['top', 'bottom']
    orientation_token = vocab.encode(orientation, add_begin_end = False)[0]   # 11 relation-semantic-label
    end = ['leftmost', 'rightmost']
    end_length = vocab.encode(end, add_begin_end = False)[0]   # 11 relation-semantic-label
    length = ['long', 'short']
    length_token = vocab.encode(length, add_begin_end = False)[0]   # 11 relation-semantic-label
    curve = ['curve', 'pad']
    curve_token = vocab.encode(curve, add_begin_end = False)[0]   # 11 relation-semantic-label

    # attribute = ['large', 'small', 'tall', 'lower', 'end', 'middle', 'top', 'bottom', 'leftmost', 'rightmost', 'corner', 'long', 'short']
    # attribute_token =  vocab.encode(attribute, add_begin_end = False)[0]

    # property_semantic = ['identity', 'color', 'function', 'attribute', 'relations'] # 4 properties, NSM: L + 2, L =1
    property_semantic = ['identity', 'color', 'function', 'size', 'height', 'position', 'orientation', 'end', 'length', 'curve', 'relations'] # 4 properties, NSM: L + 2, L =1
    property_tokenid = vocab.encode(property_semantic, add_begin_end = False)[0]
    function_semantic_token = vocab.encode(my_function, add_begin_end = False)[0]
    concept_vocab = object_semantic_filtertoken + color_semantic_token + function_semantic_token + size_token + height_token + position_token + orientation_token + end_length + length_token + curve_token + relation_semantic_tokenid
    concept_vocab_seg = [len(object_semantic_filtertoken), \
                         len(object_semantic_filtertoken) + len(color_semantic_token), \
                            len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token), \
                                len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token) +len(size_token), \
                                    len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token) +len(size_token) + len(height_token), \
                                        len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token) +len(size_token) + len(height_token) + len(position_token), \
                                            len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token) +len(size_token) + len(height_token) + len(position_token) + len(orientation_token), \
                                                len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token) +len(size_token) + len(height_token) + len(position_token) + len(orientation_token) + len(end_length), \
                                                    len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token) +len(size_token) + len(height_token) + len(position_token) + len(orientation_token) + len(end_length) + len(length_token), \
                                                        len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token) +len(size_token) + len(height_token) + len(position_token) + len(orientation_token) + len(end_length) + len(length_token) + len(curve_token), \
                                                            len(object_semantic_filtertoken) + len(color_semantic_token) + len(function_semantic_token) +len(size_token) + len(height_token) + len(position_token) + len(orientation_token) + len(end_length) + len(length_token) + len(curve_token) + len(relation_semantic_tokenid)]
    # make an object (segment) encoder for point-clouds with color
    if args.obj_cls_alpha > 0:
        if args.object_encoder == 'pnet_pp':
            object_encoder = single_object_encoder(geo_out_dim)
        elif args.object_encoder == 'pointnext':
            from .default_blocks import pointnext_object_encoder, object_cls_for_next
            print("------------using pointnext as object encoder--------------")
            object_encoder = pointnext_object_encoder(geo_out_dim)
        else:
            raise ValueError('Unknown object point cloud encoder!')
    else:
        object_encoder = None 

    # Optional, make a bbox encoder
    object_clf = None
    if args.obj_cls_alpha > 0:
        print('Adding an object-classification loss.')
        object_clf = object_decoder_for_clf(geo_out_dim, n_obj_classes)
    # object_clf = object_decoder_for_clf(geo_out_dim, n_obj_classes)

    language_clf = None
    if args.lang_cls_alpha > 0:
        print('Adding a text-classification loss.')
        language_clf = text_decoder_for_clf(512, n_obj_classes)#lang_out_dim
        # typically there are less active classes for text, but it does not affect the attained text-clf accuracy.


    instruction_clf = None
    if args.instruction_cls_alpha > 0:
        print('Adding a instruction-classification loss.')
        instruction_clf = text_decoder_for_clf(args.word_embedding_dim, n_obj_classes)#lang_out_dim
        # typically there are less active classes for text, but it does not affect the attained text-clf accuracy.
        
    language_relation_clf = None
    if args.language_relation_alpha > 0:
        language_relation_clf = text_decoder_for_clf(args.word_embedding_dim, 10)
    

    # # make a language encoder
    # lang_encoder = token_encoder(vocab=vocab,
    #                              word_embedding_dim=args.word_embedding_dim,#, glove 300,
    #                              lstm_n_hidden=lang_out_dim,
    #                              word_dropout=args.word_dropout,
    #                              random_seed=args.random_seed)

    # color token embed
    token_embed = token_embeder(vocab=vocab,
                                 word_embedding_dim=args.word_embedding_dim,# transform dim with visaul diim
                                 random_seed=args.random_seed)


    model = R2G(
        args=args,
        object_encoder=object_encoder,
        # language_encoder=lang_encoder,
        # graph_encoder=graph_encoder,
        object_clf=object_clf,
        language_clf=language_clf,
        token_embed = token_embed,
        instruction_clf = instruction_clf,
        # object_language_clf=object_language_clf,
        property_tokenid = property_tokenid,
        concept_vocab = concept_vocab,
        concept_vocab_set = concept_vocab_set,
        relation_num = len(relation_semantic),
        # object_semantic_filter_index = object_semantic_filter_index,
        language_relation_clf = language_relation_clf,
        concept_vocab_seg = concept_vocab_seg,
        num_node_properties = len(property_semantic) -1)

    return model
