import torch.nn as nn
import torch
import torch.nn.functional as F

from .mlp import MLP 
from .utils import get_siamese_features


class Relation_Estimate(nn.Module):
    def __init__(self, n_class: int, d_model: int, n_head: int = 2, n_layer: int = 3) -> None:
        super(Relation_Estimate, self).__init__()
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model= d_model, nhead = n_head, activation= 'gelu'), num_layers = n_layer)
        self.linear = nn.Linear(3, 128)
        self.predict_head = MLP(in_feat_dims=d_model, out_channels=[128, n_class], dropout_rate=[0.2])

    def forward(self, dis_vec, obj_feature, object_mask):
        """
        input:
            dis_vec:  the vector between objects center, B x N x N x 3, N is the number of objects
            obj_feature: the object feature from Pointnet++, B x N x hidden_dim, 
        output:
            the probability of the objects relation  
        """
        num_objects = obj_feature.shape[1]
        obj_feature = obj_feature.unsqueeze(1).repeat(1, num_objects, 1, 1)
        dis_vec = self.linear(dis_vec)
        all_feature = torch.cat([dis_vec, obj_feature], dim = -1)#B x N x N x (3+hidden_dim)

        prob_dis = []

        for feature in all_feature.unbind(1):#B x N x (3+hidden_dim)
            _prob_dis = self.attention(feature.transpose(0,1), src_key_padding_mask = (object_mask == float('-inf')).cuda() )
            prob_dis.append(get_siamese_features(self.predict_head, _prob_dis.transpose(0,1), aggregator=torch.stack).unsqueeze(1))
        
        final_feature = torch.cat(prob_dis, dim = 1)

        prob_dis = F.softmax(final_feature, dim =-1)# B x N x N x k




        return prob_dis, final_feature


class Attr_Estimate(nn.Module):
    """
    estiamte the attributes about:('large', 'small'), ('tall', 'lower'), ('middle', 'corner'), ('top', 'bottom'), ('leftmost', 'rightmost'), ('long', 'short')
    """
    def __init__(self, n_class: int, obj_feat: int, n_head: int = 2, n_layer: int = 3) -> None:
        super(Attr_Estimate, self).__init__()
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model= obj_feat * 3, nhead = n_head, activation= 'gelu'), num_layers = n_layer)
        self.slinear = nn.Linear(3, 128)
        self.plinear = nn.Linear(3, 128)
        # self.ls_head = MLP(in_feat_dims=obj_feat*3, out_channels=[128, 2], dropout_rate=[0.2])
        # self.tl_head = MLP(in_feat_dims=obj_feat*3, out_channels=[128, 2], dropout_rate=[0.2])
        # self.mc_head = MLP(in_feat_dims=obj_feat*3, out_channels=[128, 2], dropout_rate=[0.2])
        # self.tb_head = MLP(in_feat_dims=obj_feat*3, out_channels=[128, 2], dropout_rate=[0.2])
        self.lr_head = MLP(in_feat_dims=obj_feat*3, out_channels=[128, 2], dropout_rate=[0.2])
        # self.losh_head = MLP(in_feat_dims=obj_feat*3, out_channels=[128, 2], dropout_rate=[0.2])
        self.curve_head = MLP(in_feat_dims=obj_feat*3, out_channels=[128, 2], dropout_rate=[0.2])

        

    def forward(self, obj_feature, obj_center, obj_size, object_mask):
        """
        input:
            obj_center : object's center , B x N x 3
            obj_size : object's size , B x N x 3
            obj_feature: the object feature from Pointnet++, B x N x hidden_dim, 
        output:
            the probability of the objects relation  
        """
        B, num_objects = obj_feature.shape[:2]
        num_mask = object_mask.shape[1]
        add_mask = torch.zeros([B, num_objects - num_mask]).cuda()
        fianl_mask = torch.cat([object_mask, add_mask], dim = -1)
        # dis_vec = self.linear(dis_vec)
        f_obj_center = self.plinear(obj_center)
        f_obj_size = self.slinear(obj_size)
        final_feature = torch.cat([obj_feature, f_obj_center, f_obj_size], dim = -1)#B x N x N x (3+hidden_dim)

        # attention_feature = self.attention(final_feature.transpose(0,1) )
        attention_feature = self.attention(final_feature.transpose(0,1), src_key_padding_mask = (fianl_mask == float('-inf')).cuda() )

        # output the logits
        # ls = get_siamese_features(self.ls_head, attention_feature.transpose(0,1), aggregator=torch.stack)
        # tl = get_siamese_features(self.tl_head, attention_feature.transpose(0,1), aggregator=torch.stack)
        # mc = get_siamese_features(self.mc_head, attention_feature.transpose(0,1), aggregator=torch.stack)
        # tb = get_siamese_features(self.tb_head, attention_feature.transpose(0,1), aggregator=torch.stack)
        lr = get_siamese_features(self.lr_head, attention_feature.transpose(0,1), aggregator=torch.stack)
        curve = get_siamese_features(self.curve_head, attention_feature.transpose(0,1), aggregator=torch.stack)
        # losh = get_siamese_features(self.losh_head, attention_feature.transpose(0,1), aggregator=torch.stack)



 

        # return ls[:, 1:, :], tl[:, 1:, :], mc[:, 1:, :], tb[:, 1:, :], lr[:, 1:, :], losh[:, 1:, :]
        return lr[:, 1:, :], curve[:, 1:, :]
