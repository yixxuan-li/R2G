"""
Default choices for auxiliary classifications tasks, encoders & decoders.
"""
from turtle import forward
from .mlp import MLP
import torch.nn as nn
from .lstm_encoder import LSTMEncoder
from .word_embeddings import load_glove_pretrained_embedding, make_pretrained_embedding
from datasets.vocabulary import Vocabulary

try:
    from .point_net_pp import PointNetPP
except ImportError:
    PointNetPP = None
    msg = colored('Pnet++ is not found. Hence you cannot run all models. Install it via '
                  'external_tools (see README.txt there).', 'red')
    print(msg)


#
# Object Encoder
#

def single_object_encoder(out_dim: int) -> PointNetPP:
    """
    The default PointNet++ encoder for a 3D object.

    @param: out_dims: The dimension of each object feature
    """    
    sa_n_points=[32, 16, None]
    sa_n_samples=[32, 32, None]
    sa_radii=[0.2, 0.4, None]
    sa_mlps=[[3, 64, 64, 128],
            [128, 128, 128, 256],
            [256, 256, 512, out_dim]]
    
    tmp=PointNetPP(sa_n_points=sa_n_points,sa_n_samples=sa_n_samples,sa_mlps=sa_mlps,sa_radii=sa_radii)
    return tmp


# def pointnext_object_encoder(outdim: int = 256):
#     return PointNextEncoder(output_dim = outdim)


# def object_cls_for_next(cls_num):
#     return ClsHead(num_classes= cls_num)


#
#  Token Encoder
#
def token_encoder(vocab: Vocabulary,
                  word_embedding_dim: int,
                  lstm_n_hidden: int,
                  word_dropout: float,
                  init_c=None, init_h=None, random_seed=None,
                  feature_type='max',
                  glove_emb_file: str = '/home/yixuan/glove.6B.300d.txt') -> LSTMEncoder:#/data1/liyixuan/glove/glove.6B.300d.txt
    """
    Language Token Encoder.

    @param vocab: The vocabulary created from the dataset (nr3d or sr3d) language tokens
    @param word_embedding_dim: The dimension of each word token embedding
    @param glove_emb_file: If provided, the glove pretrained embeddings for language word tokens
    @param lstm_n_hidden: The dimension of LSTM hidden state
    @param word_dropout:
    @param init_c:
    @param init_h:
    @param random_seed:
    @param feature_type:
    """
    if len(glove_emb_file) > 0:
        print('Using glove pre-trained embeddings.')
        glove_embedding = load_glove_pretrained_embedding(glove_emb_file, verbose=True)
        word_embedding = make_pretrained_embedding(vocab, glove_embedding, random_seed=random_seed)

        # word-projection here is a bit deeper, since the glove-embedding is frozen.
        word_projection = nn.Sequential(nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU(),
                                        nn.Dropout(word_dropout),
                                        nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU())
    else:
        word_embedding = nn.Embedding(len(vocab), word_embedding_dim, padding_idx=vocab.pad)
        word_projection = nn.Sequential(nn.Dropout(word_dropout),
                                        nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU())

    assert vocab.pad == 0 and vocab.eos == 2

    model = LSTMEncoder(n_input=word_embedding_dim, n_hidden=lstm_n_hidden, word_embedding=word_embedding,
                        init_c=init_c, init_h=init_h, word_transformation=word_projection, eos_symbol=vocab.eos,
                        feature_type=feature_type)
    return model


#
# Token embed
#

def token_embeder(vocab: Vocabulary,
                  word_embedding_dim: int,
                  random_seed=None,
                  glove_emb_file: str = '/home/yixuan/glove.6B.300d.txt') -> LSTMEncoder:
    """
    Language Token Encoder.

    @param vocab: The vocabulary created from the dataset (nr3d or sr3d) language tokens
    @param word_embedding_dim: The dimension of each word token embedding
    @param glove_emb_file: If provided, the glove pretrained embeddings for language word tokens
    @param random_seed:

    """
    if len(glove_emb_file) > 0:
        print('Using glove pre-trained embeddings.')
        glove_embedding = load_glove_pretrained_embedding(glove_emb_file, verbose=True)
        word_embedding = make_pretrained_embedding(vocab, glove_embedding, random_seed=random_seed)
    else:
        word_embedding = nn.Embedding(len(vocab), word_embedding_dim, padding_idx=vocab.pad)

    assert vocab.pad == 0 and vocab.eos == 2


    return word_embedding


#
# Object Decoder
#
def object_decoder_for_clf(object_latent_dim: int, n_classes: int) -> MLP:
    """
    The default classification head for the fine-grained object classification.

    @param object_latent_dim: The dimension of each encoded object feature
    @param n_classes: The number of the fine-grained instance classes
    """
    return MLP(object_latent_dim, [128, 256, n_classes], dropout_rate=0.15)


#
#  Text Decoder
#
def text_decoder_for_clf(in_dim: int, n_classes: int) -> MLP:
    """
    Given a text encoder, decode the latent-vector into a set of clf-logits.

    @param in_dim: The dimension of each encoded text feature
    @param n_classes: The number of the fine-grained instance classes
    """
    out_channels = [128, n_classes]
    dropout_rate = [0.2]
    return MLP(in_feat_dims=in_dim, out_channels=out_channels, dropout_rate=dropout_rate)


#
# Referential Classification Decoder Head
#
def object_lang_clf(in_dim: int) -> MLP:
    """
    After the net processes the language and the geometry in the end (head) for each option (object) it
    applies this clf to create a logit.

    @param in_dim: The dimension of the fused object+language feature
    """
    return MLP(in_dim, out_channels=[128, 64, 1], dropout_rate=0.05)


## transform the softmax output to target dim
class prob_encoder(nn.Module):
    def __init__(self, in_dim = 128, out_dim = 300):
        super(prob_encoder, self).__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features = out_dim)
        self.out = nn.Softmax(dim = -1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.out(x)

        return x


