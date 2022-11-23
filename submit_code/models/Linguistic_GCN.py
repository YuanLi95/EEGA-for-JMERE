import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import BertModel
import math
from torch.nn.utils.weight_norm import weight_norm
import numpy as np
class L_GCN(torch.nn.Module):

    def __init__(self, attention_heads,hidden_size,dependency_embed_dim):
        super(L_GCN, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = attention_heads
        d_k = int(hidden_size/attention_heads)
        self.attention_head_size = d_k
        self.w_v = nn.Linear(hidden_size, attention_heads* d_k, bias=False)
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (hidden_size + hidden_size)))
        self.edge_v = nn.Linear(dependency_embed_dim, attention_heads* d_k, bias=False)
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (dependency_embed_dim + d_k)))
        self.rel_weight =nn.Linear(d_k, 1, bias=False)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def transpose_for_scores(self, x):
        if len(x.shape) == 3:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        else:
            # print(x.size()[:-1])
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)

            # return x.permute(0, 2, 1, 3)
            return x.permute(0, 3, 1, 2, 4)

    def transpose_for_edge(self, x):
        # print(x.shape)

        if len(x.shape) == 3:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)

            # return x.permute(0, 2, 1, 3)
            return x.permute(0, 3, 1, 2, 4)

    def forward(self, token_feature, edge_feature, dependency_masks):
        batch, seq, dim = token_feature.shape

        mixed_value_layer = self.w_v(token_feature)



        mixed_value_layer = mixed_value_layer.unsqueeze(1)
        mixed_value_layer = torch.repeat_interleave(mixed_value_layer, repeats=seq, dim=1)
        edge_mask = dependency_masks.unsqueeze(1)
        edge_mask = torch.repeat_interleave(edge_mask, repeats=self.num_attention_heads, dim=1)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # print(value_layer.shape)
        # print(edge_feature.shape)
        edge_v = self.edge_v(edge_feature)
        edge_v = self.transpose_for_edge(edge_v)
        edge_weight =  self.rel_weight(edge_v)



        attention_scores = torch.relu(edge_weight).squeeze(-1)
        attention_weights = attention_scores.masked_fill(edge_mask.expand_as(attention_scores) == 0, float(-1e6))
        attention_probs = nn.Softmax(dim=-1)(attention_weights).unsqueeze(-2)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer
        outputs = self.act(token_feature + outputs)

        return outputs
