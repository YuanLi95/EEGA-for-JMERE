# -*- coding: utf-8 -*-

import numpy as np
import pickle
import spacy
import networkx as nx
import  re
import json

from transformers import BertTokenizer
bert_tokenizer =BertTokenizer.from_pretrained('bert-base-uncased')

import json
token_nize_for_tokenize = spacy.load('en_core_web_trf')

def tokenize(text):
    text=text.lower()
    document = text.split()
    # print(document)
    # document = token_nize_for_tokenize(text)
    # print
    return  " ".join(document)
    # return " ".join([token.text for token in document])

def aspect_short_path(G, target):
    """"
    """
    d = nx.shortest_path_length(G, target=target)
    distance_list = []
    for node in G.nodes():
        try:
            distance_list.append(d[node])
        except KeyError:
            distance_list.append(-1)
    return distance_list

def dependency_adj_matrix(text):
    # text = "Great food but the service was dreadful !"

    word_piece_id = bert_tokenizer(text,is_split_into_words=True, padding=True, add_special_tokens =False)
    # print(word_piece_id["input_ids"])
    word_pieces = bert_tokenizer.convert_ids_to_tokens(word_piece_id["input_ids"])


    # print(word_pieces)
    # for i in range
    # print(len(word_pieces))
    # print("+++++++++++++++++++++++++++begin+++++++++++++++++++")

    document = token_nize_for_tokenize(text)
    #     document = token_nize_for_tokenize (text)
    seq_len = len(word_pieces)
    Syntactic_dependence = []
    pos_sequence = []

    # 创建三元组(piece_token,old_index,new_index)
    three_list = []

    star =0
    for index,w in enumerate(text.split()):
        w_wordpice = bert_tokenizer(w, add_special_tokens =False)
        w_len = len(w_wordpice["input_ids"])
        end = star+w_len
        three_list.append([w, index, [star,end-1]])
        star = end
    # print(word_pieces)
    # print(len(word_pieces))

    # exit()


    matrix_dir = np.zeros([seq_len,seq_len]).astype('float32')

    matrix_undir = np.zeros([seq_len, seq_len]).astype('float32')
    matrix_redir = np.zeros([seq_len, seq_len]).astype('float32')
    #加上CLS和SEP
    final_matrix_dir = np.ones([seq_len+2,seq_len+2]).astype('float32')
    # final_matrix_dir = np.zeros([seq_len+2,seq_len+2]).astype('float32')


    final_matrix_undir = np.ones([seq_len+2, seq_len+2]).astype('float32')
    # final_matrix_undir = np.zeros([seq_len + 2, seq_len + 2]).astype('float32')
    #
    final_matrix_redir = np.ones([seq_len+2, seq_len+2]).astype('float32')
    # final_matrix_redir = np.zeros([seq_len+2, seq_len+2]).astype('float32')
    distance_list = []
    # print(three_list)
    pos_sequence.append("<unk>")
    for index,token in enumerate(document):
        # print(token,token.pos_)
        token_word_piece_list = []
        if (token.head.i<len(three_list)==True)&((token.i)<len(three_list)==True):
            token_index=three_list[token.i][2]
            token_head_index =three_list[token.head.i][2]
        else:
            if token.i >= len(three_list):
                token_index = three_list[len(three_list) - 1][2]
                if token.head.i >= len(three_list):
                    token_index = three_list[len(three_list) - 1][2]
                else:
                    token_head_index = three_list[token.head.i][2]
            else:
                token_index = three_list[token.i][2]
                if token.head.i >= len(three_list):
                    token_head_index = three_list[len(three_list) - 1][2]
                else:
                    token_head_index = three_list[token.head.i][2]

        token_l, token_r = token_index
        head_l, head_r = token_head_index
        for token_piece in range(token_l, token_r+1):
            pos_sequence.append(token.pos_.lower())

            for head_piece in range(head_l, head_r+1):
                matrix_dir[token_piece][head_piece] = 1
                matrix_undir[token_piece][head_piece] = 1
                matrix_undir[head_piece][token_piece] = 1
                # cls +1
                Syntactic_dependence.append([token_piece+1, token.dep_.lower(), head_piece+1])


                # Syntactic_dependence.append([token_piece+1, token.dep_.lower(), head_piece+1])

                # Syntactic_dependence.append([head_piece+1, token.dep_.lower(), token_piece+1])

            Syntactic_dependence.append([token_piece+1, "selfcycle", token_piece+1])
    pos_sequence.append("<unk>")
    for i in range(len(matrix_undir)):
        matrix_undir[i][i]=1
        matrix_dir[i][i]=1
        matrix_redir[i][i]=1

    #中间

    final_matrix_dir[1:-1,1:-1] =matrix_dir
    final_matrix_undir[1:-1,1:-1] = matrix_undir
    final_matrix_redir[1:-1,1:-1] = matrix_redir
    # final_matrix_dir = matrix_dir
    # final_matrix_undir = matrix_undir
    # final_matrix_redir = matrix_redir
    # print(final_matrix_undir.shape)



    G = nx.from_numpy_matrix(matrix_undir)
    distance_list.append([1000]*(matrix_undir.shape[0]+2))
    for i in range (matrix_undir.shape[0]):
        i_distance = np.array(aspect_short_path(G, i)).tolist()
        for index, distance in enumerate(i_distance):
            if distance == -1:
                i_distance[index] = 1000
        #加入头和末尾
        i_distance =[1000]+i_distance+[1000]
        distance_list.append(i_distance)
    distance_list.append([1000]*(matrix_undir.shape[0]+2))
    # print(len( word_pieces))
    # print(len(pos_sequence))
    if len(pos_sequence)!=(len(word_pieces)+2):
        if len(pos_sequence)>(len(word_pieces)+2):
            pos_sequence=pos_sequence[0:len(word_pieces)+2]
        else:
            for i in range(len(word_pieces),len(pos_sequence)+2):
                pos_sequence.append("<unk>")

    if len(pos_sequence) != (len(word_pieces) + 2):
        print(word_pieces)
        print(pos_sequence)
        print("11111111111111111111111111")
    return final_matrix_dir,final_matrix_undir,Syntactic_dependence,distance_list,pos_sequence


def process(filename_path):

    print(filename_path)
    f= open(filename_path,"r")

    lines = json.load(f)

    idx2graph_undir = {}
    Syntactic_dependence_all = {}
    idx2positon = {}
    idx2pos_sequence ={}

    filename = filename_path
    fout_undir = open(filename + 'undir' +"bert"+'.dependency_graph', 'wb')
    dependency_analysis = open(filename + "bert"+'.dependency', 'wb')
    fout_syntax_position = open(filename + "bert"+'.syntaxPosition', 'wb')
    pos_sequence_file = open(filename + "bert" + '.pos_sequence', 'wb')


    for i in range(0, len(lines)):
        sentence = lines[i]['token']
        # print(sentence)
        print(i)
        text = " ".join(sentence)

        # print(text)
        input_text = tokenize(text)


        # print(input_text)
        adj_matrix_dir,adj_matrix_undir,Syntactic_dependence,distance_list,pos_sequence= dependency_adj_matrix(input_text)
        # idx2graph_redir[i] = adj_matrix_redir
        idx2graph_undir[i] = adj_matrix_undir

        Syntactic_dependence_all[i] = Syntactic_dependence
        #syntax_position_distance
        idx2positon[i] = distance_list
        idx2pos_sequence[i] = pos_sequence

    # pickle.dump(idx2graph_dir, fout_dir)
    pickle.dump(idx2graph_undir, fout_undir)
    # pickle.dump(idx2graph_redir,fout_redir)
    pickle.dump(Syntactic_dependence_all,dependency_analysis)
    pickle.dump(idx2positon,fout_syntax_position)
    pickle.dump(idx2pos_sequence, pos_sequence_file)
    # fout_dir.close()
    # fout_redir.close()
    fout_undir.close()
    dependency_analysis.close()
    fout_syntax_position.close()
    pos_sequence_file.close()

if __name__ == '__main__':
    # print('################################')
    process('./no_none_unified_tags_txt/train.json')
    # print(count)
    process('./no_none_unified_tags_txt/val.json')
    # print(count)
    process('./no_none_unified_tags_txt/test.json')
    # print(count)
    # print('################################')


