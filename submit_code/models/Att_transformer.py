import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import BertModel
import math
import  numpy as np

class Att(torch.nn.Module):

    def __init__(self, attention_heads,hidden_size,dependency_embed_dim):
        super(Att, self).__init__()
        self.hidden_size = hidden_size
        self.text_W_q = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.text_W_K = torch.nn.Linear(self.hidden_size,
                                        self.hidden_size)
        self.text_W_V = torch.nn.Linear(self.hidden_size,
                                        self.hidden_size)

        self.edge_k = torch.nn.Linear(dependency_embed_dim,self.hidden_size)
        self.edge_v = torch.nn.Linear(dependency_embed_dim,self.hidden_size)

        self.num_attention_heads = attention_heads
        self.attention_head_size = int((self.hidden_size) / self.num_attention_heads)
        # self.edge_k = torch.nn.Linear(args.dependency_embed_dim,
        #                               args.lstm_dim * 4 + args.position_embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout_2= nn.Dropout(0.2)
        self.LayerNorm_2 = nn.LayerNorm(self.hidden_size)
        self.dense = nn.Linear(self.hidden_size,self.hidden_size,bias=False)

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
        mixed_query_layer = self.text_W_q(token_feature)
        mixed_key_layer = self.text_W_K(token_feature)
        mixed_value_layer = self.text_W_V(token_feature)

        mixed_query_layer = mixed_query_layer.unsqueeze(2)

        mixed_key_layer = mixed_key_layer.unsqueeze(1)
        mixed_key_layer = torch.repeat_interleave(mixed_key_layer, repeats=seq, dim=1)

        mixed_value_layer = mixed_value_layer.unsqueeze(1)
        mixed_value_layer = torch.repeat_interleave(mixed_value_layer, repeats=seq, dim=1)
        edge_mask = dependency_masks.unsqueeze(1)
        edge_mask = torch.repeat_interleave(edge_mask, repeats=self.num_attention_heads, dim=1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # print(value_layer.shape)
        # print(edge_feature.shape)
        edge_k = self.edge_k(edge_feature)
        edge_v = self.edge_v(edge_feature)
        edge_k = self.transpose_for_edge(edge_k)
        edge_v = self.transpose_for_edge(edge_v)
        key_layer = key_layer +  edge_k
        value_layer = value_layer +  edge_v
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)).squeeze(-2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_weights = attention_scores.masked_fill(edge_mask.expand_as(attention_scores) == 0, float(-1e6))
        attention_probs = nn.Softmax(dim=-1)(attention_weights).unsqueeze(-2)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer
        outputs = self.LayerNorm(token_feature + outputs)
        outputs_dense = self.dense(outputs)
        outputs = self.LayerNorm_2(outputs+outputs_dense)

        return outputs




class Crooss_attention(torch.nn.Module):
        def __init__(self, n_head, d_model, d_k, d_v, dropout=0.2, dropout2=False, attn_type='softmax'):
            super().__init__()

            self.n_head = n_head
            self.d_k = d_k
            self.d_v = d_v

            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
            nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

            if dropout2:
                # self.dropout2 = nn.Dropout(dropout2)
                self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                           dropout=dropout2)
            else:
                self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                           dropout=dropout)

            self.dropout = nn.Dropout(dropout)

            self.layer_norm = nn.LayerNorm(d_model)

            if n_head > 1:
                self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
                nn.init.xavier_normal_(self.fc.weight)

        def forward(self, q, k, v, attn_mask=None, dec_self=False):

            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

            sz_b, len_q, _ = q.size()
            sz_b, len_k, _ = k.size()
            sz_b, len_v, _ = v.size()

            residual = q

            if hasattr(self, 'dropout2'):
                q = self.dropout2(q)

            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

            q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

            if attn_mask is not None:
                attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

            output, attn = self.attention(q, k, v, attn_mask=attn_mask)

            output = output.view(n_head, sz_b, len_q, d_v)
            output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

            if hasattr(self, 'fc'):
                output = self.fc(output)

            if hasattr(self, 'dropout'):
                output = self.dropout(output)

            if dec_self:
                output = self.layer_norm(output + residual)
            else:
                output = self.layer_norm(output + residual)

            return output, attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.2, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            # print(attn_mask)
            attn = attn.masked_fill(attn_mask == 0, float(-1e6))
            # attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
