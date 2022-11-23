import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import BertModel
import math
from torch.nn.utils.weight_norm import weight_norm
from .Att_transformer import  Att,Crooss_attention
from .Linguistic_GCN import L_GCN
from .graph_alght_network import  GraphOTG
from .OT_torch_ import cost_matrix_batch_torch, IPOT_torch_batch_uniform, GW_distance_uniform, IPOT_distance_torch_batch_uniform



def Linear(inputdim, outputdim, bias=True, uniform=True):
    linear = nn.Linear(inputdim, outputdim, bias)
    if uniform:
        nn.init.xavier_uniform_(linear.weight)
    else:
        nn.init.xavier_normal_(linear.weight)
    if bias:
        nn.init.constant_(linear.bias, 0.0)
    return linear

class AMGNetwork(torch.nn.Module):
    def __init__(self, args,dependency_embedding,
                 position_embedding,rel_image_embedding,pospeech_embedding):
        super(AMGNetwork, self).__init__()
        self.args = args
        self.hidden_dim= args.hidden_dim
        self.dependency_dim = dependency_embedding.shape[1]
        self.image_rel_dim = rel_image_embedding.shape[1]
        self.position_dim =position_embedding.shape[1]
        self.frequency_dim= args.frequency_embed_dim
        self.pospeech_dim = pospeech_embedding.shape[1]


        self.dependency_embed = \
            torch.nn.Embedding(dependency_embedding.shape[0],self.dependency_dim , padding_idx=0, )
        self.dependency_embed.weight.data.copy_(torch.from_numpy(dependency_embedding))


        self.position_embed = \
            torch.nn.Embedding(position_embedding.shape[0], position_embedding.shape[1], padding_idx=0, )
        self.position_embed.weight.data.copy_(torch.from_numpy(position_embedding))
        self.position_embed.weight.requires_grad = True

        self.image_rel_embed = \
            torch.nn.Embedding(rel_image_embedding.shape[0], rel_image_embedding.shape[1], padding_idx=0, )
        self.image_rel_embed.weight.data.copy_(torch.from_numpy(rel_image_embedding))
        self.image_rel_embed.weight.requires_grad = True

        self.frequency_embed = \
            torch.nn.Embedding(11,args.frequency_embed_dim, padding_idx=0)
        self.frequency_embed.weight.requires_grad = True

        self.pospeech_embed = torch.nn.Embedding(pospeech_embedding.shape[0],pospeech_embedding.shape[1], padding_idx=0 )
        self.pospeech_embed.weight.data.copy_(torch.from_numpy(pospeech_embedding))
        self.pospeech_embed.weight.requires_grad = True
        # self.nhops = args.nhops
        self.GraphOTG_layers = nn.ModuleList([GraphOTG(args,hidden_dim=self.hidden_dim) for i in range(args.nhops)])
        self.bert = BertModel.from_pretrained(args.bert_model_path,return_dict = False)
        self.trans_image = nn.Sequential(Linear(args.feature_image, args.bert_feature_dim), nn.ReLU(), nn.Dropout(args.trans_image_dro),
                                       Linear(args.bert_feature_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(args.trans_image_dro))
        self.trans_token = nn.Sequential(Linear(args.bert_feature_dim, self.hidden_dim), nn.ReLU(),nn.Dropout(0.2))

        self.pospeech_gcn= L_GCN(args.attention_heads,args.hidden_dim,self.pospeech_dim)
        self.freqence_gcn= L_GCN(args.attention_heads,args.hidden_dim,self.frequency_dim)
        self.distance_gcn=L_GCN(args.attention_heads,args.hidden_dim,self.position_dim)
        self.dropout_all = nn.Dropout(0.2)
        self.linear_all = nn.Linear(args.hidden_dim*3, args.hidden_dim)
        self.ot_loss = torch.tensor(0.0).to(args.device)
        self.drop_out = nn.Dropout(0.2)
        self.linear_out = nn.Linear(args.hidden_dim*2+self.pospeech_dim+self.frequency_dim+self.position_dim, args.class_num)

    def forward(self, bert_tokens,token_masks, token_dependency_masks, \
            token_syntactic_position,token_edge_data, token_frequency_graph,pospeech_tokens,
                image_rel_matrix, image_rel_mask, image_feature):
        bs,token_seq,= bert_tokens.shape
        _,image_seq,_ =  image_feature.shape

        cross_attention_mask= token_masks.unsqueeze(2)
        token_real_lenth = token_masks.norm(1)

        cross_attention_mask =torch.repeat_interleave(cross_attention_mask,dim=2,repeats=image_seq).double()
        token_edge_flatten = token_edge_data.view(bs,-1)
        token_edge_embed = self.dependency_embed(token_edge_flatten)
        token_edge_embed =token_edge_embed.view(bs, token_seq,token_seq,-1)

        image_rel_flatten = image_rel_matrix.view(bs, -1)

        image_rel_embed = self.image_rel_embed(image_rel_flatten)
        image_rel_embed =  image_rel_embed.view(bs, image_seq, image_seq, -1)


        out_bert = self.bert(bert_tokens,token_masks)
        x_token,cls_token = out_bert[0],out_bert[1]

        x_image = self.trans_image(image_feature)
        x_token = self.trans_token(x_token)
        ot_loss_all=[]
        for self.GraphOTG in  self.GraphOTG_layers:
            image_att,token_att,otg_loss,imagefortext,textforimage,T_wd,T_gwd,att= self.GraphOTG(x_image,image_rel_embed,image_rel_mask,
                                                                     x_token,token_edge_embed,token_dependency_masks,
                                                                     token_masks,cross_attention_mask)
            x_image= textforimage
            x_token = imagefortext


            # otg_loss =0.0
            ot_loss_all.append(otg_loss)
            # print(otg_loss)
            # print(self.ot_loss)
            # self.ot_loss+=otg_loss
        # imagefortext= x_token

        token_matrix_mask = torch.bmm(token_masks.unsqueeze(2), token_masks.unsqueeze(2).transpose(1, 2))
        # print(token_matrix_mask.shape)

        token_frequency_embed = self.frequency_embed(token_frequency_graph.view(bs,-1))
        token_frequency_embed = token_frequency_embed.view(bs,token_seq, token_seq,-1)
        #转换为matrix


        token_frequency_mask = torch.where(token_frequency_graph != 0, 1, 0)

        token_pospeech_embed = self.pospeech_embed(pospeech_tokens.view(bs,-1))
        token_pospeech_embed = token_pospeech_embed.view(bs,1,token_seq,-1)
        token_pospeech_embed_one = torch.repeat_interleave(token_pospeech_embed, repeats=token_seq, dim=1)
        token_pospeech_matrix = torch.add(token_pospeech_embed_one, token_pospeech_embed_one.transpose(1, 2))

        syntacx_postition_embed = self.position_embed(token_syntactic_position.view(bs, -1))
        syntacx_postition_embed = syntacx_postition_embed.view(bs, token_seq, token_seq, -1)
        #
        # print(token_frequency_embed.shape)
        # print(token_pospeech_matrix.shape)
        # print(syntacx_postition_embed.shape)

        frequency_channel = self.freqence_gcn(imagefortext,token_frequency_embed,token_frequency_mask)
        # print("frequency_channel over")

        pospeech_channel = self.pospeech_gcn(imagefortext,token_pospeech_matrix,token_matrix_mask)

        syntax_position_channel =self.distance_gcn(imagefortext,syntacx_postition_embed,token_matrix_mask)

        out_feature = torch.cat([frequency_channel,pospeech_channel,syntax_position_channel],dim=-1)
        out_feature = self.linear_all(out_feature)
        out_feature = self.dropout_all(out_feature)
        final_feature = out_feature.unsqueeze(2).expand([-1, -1, token_seq, -1])
        # print(bert_feature.shape)
        # exit()
        final_feature_T = final_feature.transpose(1, 2)
        features = torch.cat([final_feature, final_feature_T], dim=3)
        # print(features.shape)
        # print(features.shape)
        # print(token_pospeech_matrix.shape)
        # print(token_frequency_embed.shape)
        # print(syntax_position_channel.shape)


        concat_features = torch.cat([features,token_frequency_embed,token_pospeech_matrix,syntacx_postition_embed], dim=3)
        concat_features = self.drop_out(concat_features)
        # print(concat_features.shape)
        logits = self.linear_out(concat_features)
        # print(logits.shape)
        print(bert_tokens.cpu().numpy().tolist())

        return  logits, ot_loss_all,T_wd,T_gwd,att



