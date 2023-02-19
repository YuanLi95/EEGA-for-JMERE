import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
# from nltk.corpus import stopwords
import math
# from nltk.corpus import stopwords
from transformers import BertTokenizer
bert_tokenizer =BertTokenizer.from_pretrained('bert-base-uncased')
import pickle
import collections
from collections import Counter
import  json
from nltk.corpus import stopwords
frequence_dict={1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5,11:6,12:6,13:7,14:7,15:7}
def tokenize(text):
    text=text.lower()
    document = text.split()
    # print(document)
    # document = token_nize_for_tokenize(text)
    return  " ".join(document)

def stopword():
    #得到bert分词之后的停止词
    stop_words = stopwords.words('english')
    for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's','(',')','-','[',']',
              '{','}',';',':','\'','"','1','2','3','4','5','6','7','8','9','0',
    '\\','<','>','.','/','@','+','=','#','$','%','^','&','*','~','|']:

        stop_words.append(w)
    text = " ".join(stop_words)

    word_piece_id = bert_tokenizer(text, is_split_into_words=True, padding=True, add_special_tokens=False)
    # print(word_piece_id["input_ids"])
    stop_words_for_bert = bert_tokenizer.convert_ids_to_tokens(word_piece_id["input_ids"])
    stop_words_for_bert = list(set(stop_words_for_bert))


    return stop_words_for_bert

punc = stopword()

def dataProcess(directory, window):
    total_token = 0

    print(directory)
    file_list = ["test.txt","train.txt","val.txt"]
    for filename in file_list:

        print(os.path.join(directory, filename))
        # exit()
        f = open(os.path.join(directory, filename), encoding='UTF-8')
        lines = f.readlines()
        for i in range(0, len(lines)):
            sentence = eval(lines[i])['token']
            sentence = " ".join(sentence)
            # Remove non-alphanumeric characters
            for ele in sentence:
                if ele in punc:
                    sentence = sentence.replace(ele, "")
            input_text = tokenize(sentence)
            word_piece_id = bert_tokenizer(input_text, is_split_into_words=True, padding=True, add_special_tokens=False)
            # print(word_piece_id["input_ids"])
            word_piece_for_bert = bert_tokenizer.convert_ids_to_tokens(word_piece_id["input_ids"])

            # Create Dictionary of co-occurance words
            # print(sentence)
            # exit()
            co_occurrence(word_piece_for_bert, window)
            # exit()
            # Count token number
            for word in word_piece_for_bert:
                total_token = total_token + 1

    print("There are {} number of tokens.".format(total_token))
    print("Number of unique token: ", len(vocab))


# Create Dictionary of co-occurance words
def co_occurrence(my_string, window):
    # For every sentence creating and adding word pairs with window size
    for i in range(len(my_string)):
        token = my_string[i]
        vocab.add(token)  # add to vocab
        next_token = my_string[i + 1: i + window]
        for t in next_token:
            key = tuple([token, t])
            word_dic[key] += 1
        # print(word_dic)
        # print(vocab)


# Create co-occurance matrix (cell value frequency)
def create_frame():
    # create dataframe of n*n
    frame = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                         index=sorted(vocab),
                         columns=sorted(vocab))

    # Adding values to cell
    # print(word_dic)
    # print(frame)
    # exit()
    for key, value in word_dic.items():
        frame.at[key[0], key[1]] = value
        frame.at[key[1], key[0]] = value
    return frame


# Create co-occurance matrix (cell value PMI)
def PMI(df):
    # Used formula  pmi = log2 P(x;y)/P(x)*P(y)

    # total count of each row and column
    col_totals = frame.sum(axis=0)
    row_totals = frame.sum(axis=1)

    # total occurance
    total = col_totals.sum()
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected

    # To ignore divide error
    with np.errstate(divide='ignore'):
        df = np.log2(df)
        df[np.isinf(df)] = 0.0  # replace inf by zero

    return df


def process_single(fname,pmi_matrix):
    f = open(fname, "r", encoding='UTF-8')
    lines = json.load(f)
    f.close()
    idx2graph = {}
    rownames = pmi_matrix.index.values
    for i in range(0, len(lines)):
        sentence = lines[i]['token']
        sentence = " ".join(sentence)
        input_text = tokenize(sentence)
        word_piece_id = bert_tokenizer(input_text, is_split_into_words=True, padding=True, add_special_tokens=False)
        word_pieces = bert_tokenizer.convert_ids_to_tokens(word_piece_id["input_ids"])
        # print(word_pieces)
        seq_len = len(word_pieces)
        # print(len(word_pieces))
        adj_matrix = np.zeros((seq_len, seq_len)).astype('float32')
        new_adj_matrix_all =np.zeros((seq_len+2, seq_len+2)).astype('int')
        for j in range(seq_len - 1):

            word1 = word_pieces[j]

            if word1 in rownames:

                adj_matrix[j][j] = pmi_matrix.loc[word1, word1]
                for k in range(j + 1, seq_len):
                    word2 = word_pieces[k]
                    if word2 in rownames:

                        adj_matrix[j][k] = pmi_matrix.loc[word1,word2]

                        adj_matrix[k][j] = pmi_matrix.loc[word2,word1]
                    else:
                        adj_matrix[j][k] = 0.0
                        adj_matrix[k][j] = 0.0
            else:
                adj_matrix[:, j] = 0.0
                adj_matrix[j, :] = 0.0
                adj_matrix[j, j] = 1.0

        new_adj_matrix = np.where(adj_matrix <= 0, 0.0, adj_matrix)
        new_adj_matrix = np.divide(new_adj_matrix, 2)
        new_adj_matrix = np.ceil(new_adj_matrix)
        # print(new_adj_matrix.shape)
        # print(new_adj_matrix)
        # exit()

        new_adj_matrix_all[1:-1,1:-1]=new_adj_matrix
        idx2graph[i] = new_adj_matrix_all

    fout = open(fname + 'frequency' + '.graph', 'wb')
    pickle.dump(idx2graph, fout)
    fout.close()


def process_single_test(fname,pmi_matrix):
    f = open(fname, "r", encoding='UTF-8')
    lines = json.load(f)
    f.close()
    idx2graph = {}
    rownames = pmi_matrix.index.values
    for i in range(0, len(lines)):
        sentence = lines[i]['token']
        sentence = " ".join(sentence)
        input_text = tokenize(sentence)
        word_piece_id = bert_tokenizer(input_text, is_split_into_words=True, padding=True, add_special_tokens=False)
        word_pieces = bert_tokenizer.convert_ids_to_tokens(word_piece_id["input_ids"])
        seq_len = len(word_pieces)
        adj_matrix = np.zeros((seq_len, seq_len)).astype('float32')
        for j in range(seq_len - 1):

            word1 = word_pieces[j]

            if word1 in rownames:

                adj_matrix[j][j] = pmi_matrix.loc[word1, word1]
                for k in range(j + 1, seq_len):
                    word2 = word_pieces[k]
                    if word2 in rownames:

                        adj_matrix[j][k] = pmi_matrix.loc[word1,word2]

                        adj_matrix[k][j] = pmi_matrix.loc[word2,word1]
                    else:
                        adj_matrix[j][k] = 0.0
                        adj_matrix[k][j] = 0.0
            else:
                adj_matrix[:, j] = 0.0
                adj_matrix[j, :] = 0.0
                adj_matrix[j, j] = 1.0

        new_adj_matrix = np.where(adj_matrix<=0, 0.0, adj_matrix)
        new_adj_matrix = np.divide(new_adj_matrix,2)
        new_adj_matrix = np.ceil(new_adj_matrix)

        idx2graph[i] = new_adj_matrix

    fout = open(fname + 'frequency' + '.graph', 'wb')
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == "__main__":

    #获得全局共现矩阵
    if os.path.exists ("co-occurance.matrix"):
        fr = open("co-occurance.matrix","rb")
        pmi_matrix = pickle.load(fr)
    else:
        setting_dict = {"Window":8,"Directory":"./unified_tags_txt"}
        word_dic = defaultdict(int)
        vocab = set()
        dataProcess(setting_dict["Directory"], setting_dict["Window"])
        print("Creating Dictionary..........")
        print("Creating PMI Matrix..........")
        frame = create_frame()
        pmi_matrix = PMI(frame)

        fw = open("co-occurance.matrix", "wb")
        pickle.dump(pmi_matrix, fw)


    pmi_matrix = pmi_matrix.round()
    numpy_pmi = pmi_matrix.to_numpy()
    print(numpy_pmi.max())
    print(numpy_pmi.min())
    print(pmi_matrix)
    # exit()

    # print("-------------------beging train----------------------------")
    process_single("no_none_unified_tags_txt/train.json",pmi_matrix)
    print("-------------------beging val----------------------------")
    process_single("no_none_unified_tags_txt/val.json",pmi_matrix)
    print("-------------------beging test----------------------------")
    process_single("no_none_unified_tags_txt/test.json",pmi_matrix)





