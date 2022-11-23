import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import BertModel
import math
import  numpy as np
from .Att_transformer import  Att,Crooss_attention
from .OT_torch_ import cost_matrix_batch_torch, IPOT_torch_batch_uniform, GW_distance_uniform, IPOT_distance_torch_batch_uniform

class GOT(torch.nn.Module):
    def __init__(self, args):
        super(GOT, self).__init__()
        self.args = args
    def forward(self,  image_feature,token_feature,token_mask,token_dependency_masks,image_rel_mask):
        token_mask = token_mask.unsqueeze(2)
        token_feature_mask = torch.repeat_interleave(token_mask,repeats=token_feature.shape[-1],dim=2)
        # print(token_feature_mask.shape)
        # print(token_feature.shape)
        token_feature = torch.mul(token_feature,token_feature_mask)

        cos_distance = cost_matrix_batch_torch(image_feature.transpose(2, 1), token_feature.transpose(2, 1))


        image_rel_mask = torch.where(image_rel_mask==1,1.0,1e-5)
        token_dependency_masks = torch.where(token_dependency_masks==1, 1.0, 1e-5)


        cos_distance = cos_distance.transpose(1, 2)
        # OT_sim = IPOT_torch_batch_uniform(cos_distance, v_.size(0), v_.size(1), q_.size(1), 30)
        # logits_1 = OT_sim.unsqueeze(1)

        beta = 0.1
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = torch.nn.functional.relu(cos_distance - threshold)
        # print(image_feature.shape)
        # print(token_feature.shape)


        wd ,T_wd= IPOT_distance_torch_batch_uniform(cos_dist, image_feature.size(0), image_feature.size(1), token_feature.size(1), 20)
        # print(wd)
        # twd = .5 * torch.mean(wd)
        gwd,T_gwd = GW_distance_uniform(image_feature.transpose(2, 1), token_feature.transpose(2, 1),token_dependency_masks,image_rel_mask,iteration=5, OT_iteration=20)
        # twd = .2 * torch.mean(gwd)
        twd = self.args.twd_weight * torch.mean(gwd) + self.args.twd_weight * torch.mean(wd)
        return  twd,T_wd,T_gwd



class GraphOTG(torch.nn.Module):

    def __init__(self, args,hidden_dim):
        super(GraphOTG, self).__init__()
        self.otg = GOT(args)
        self.hidden_dim = hidden_dim
        self.token_transformer = Att(args.attention_heads, self.hidden_dim, args.dependency_embed_dim)
        self.image_transformer = Att(args.attention_heads, self.hidden_dim, args.image_rel_embed_dim)
        self.imtotext_cross_attention = Crooss_attention(args.cross_attention_heads, self.hidden_dim, self.hidden_dim,
                                                self.hidden_dim)

        self.texttoim_cross_attention = Crooss_attention(args.cross_attention_heads, self.hidden_dim, self.hidden_dim,
                                                self.hidden_dim)

    def forward(self,x_image,image_rel_embed,image_rel_mask,x_token,token_edge_embed,token_dependency_masks,token_masks,cross_attention_mask):

        image_att = self.image_transformer(x_image, image_rel_embed, image_rel_mask)
        token_att = self.token_transformer(x_token, token_edge_embed, token_dependency_masks)
        # print(image_att.shape)
        # print(token_att.shape)
        otg_loss,T_wd,T_gwd= self.otg(image_att, token_att, token_masks, token_dependency_masks, image_rel_mask)

        textforimage = x_image
        imagefortext,_ = self.imtotext_cross_attention(token_att, image_att, image_att, cross_attention_mask)
        # textforimage, _ = self.imtotext_cross_attention(image_att, token_att, token_att, cross_attention_mask.transpose(1,2))



        return image_att,token_att,otg_loss,imagefortext,textforimage,T_wd,T_gwd,_

