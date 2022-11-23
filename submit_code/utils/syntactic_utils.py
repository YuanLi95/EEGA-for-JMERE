import  os
import  pickle
import  numpy as np
import json

def build_positionizer(data_dir):

    # print(data_dir)
    # exit()
    if os.path.exists(os.path.join(data_dir, 'bertpos2idx.pkl')):

        print('>>> loading {0} tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir,'bertpos2idx.pkl'), 'rb') as f:
            position2idx = pickle.load(f)
            position_tokenizer = Positionnizer(position2idx=position2idx)
    else:
        syntax_position_all = []
        filenames = ['test.json', 'val.json', 'train.json']
        for filename in filenames:
            syntax_position_file = data_dir+"{0}bert.syntaxPosition".format(filename)
            syntax_position_file = open(syntax_position_file,"rb")
            syntax_position_list = pickle.load(syntax_position_file)
            syntax_position_file.close()
            for key,syntax_position in syntax_position_list.items():
                for j in syntax_position:
                    for i in j :
                        syntax_position_all.append(str(i))
        position_tokenizer = Positionnizer()
        position_tokenizer.fit_on_position(syntax_position_all)
        print(position_tokenizer.position2idx)
        print('>>> saving {0} position_tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir, 'bertpos2idx.pkl'), 'wb') as f:
            pickle.dump(position_tokenizer.position2idx, f)

    return position_tokenizer


def build_Part_of_Speechlizer(data_dir):
    #chuang jian pos sequence
    if os.path.exists(os.path.join(data_dir, 'bertpospeech2idx.pkl')):
        print('>>> loading {0} tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir,'bertpospeech2idx.pkl'), 'rb') as f:
            dependency2idx = pickle.load(f)
            bertpospeechnizer = pospeech_tokennizer(dependency2idx=dependency2idx)
    else:
        dependency_all = " "
        filenames = ['train.json', 'val.json', 'test.json']
        for filename in filenames:
            dependency_file = data_dir+ "{0}bert.pos_sequence".format(filename)
            dependency_file = open(dependency_file,"rb")
            dependency_file_list = pickle.load(dependency_file)
            dependency_file.close()
            for key,dependency  in dependency_file_list.items():
                # print(dependency)
                for j in dependency:
                    dependency_all+= " "+j
        bertpospeechnizer = pospeech_tokennizer()
        bertpospeechnizer.fit_on_dependency(dependency_all)
        print('>>> saving {0} bertpospeech_tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir, 'bertpospeech2idx.pkl'), 'wb') as f:
            pickle.dump(bertpospeechnizer.dependency2idx, f)
    print(bertpospeechnizer.dependency2idx)
    return bertpospeechnizer


def build_dependencyizer(data_dir):

    if os.path.exists(os.path.join(data_dir, 'bertdependency2idx.pkl')):
        print('>>> loading {0} tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir,'bertdependency2idx.pkl'), 'rb') as f:
            dependency2idx = pickle.load(f)
            dependency_tokenizer = Dependecynizer(dependency2idx=dependency2idx)
    else:
        dependency_all = " "
        filenames = ['train.json', 'val.json', 'test.json']
        for filename in filenames:
            dependency_file = data_dir+ "{0}bert.dependency".format(filename)
            dependency_file = open(dependency_file,"rb")
            dependency_file_list = pickle.load(dependency_file)
            dependency_file.close()
            for key,dependency  in dependency_file_list.items():
                # print(dependency)
                for j in dependency:
                    dependency_all+= " "+j[1]
        dependency_tokenizer = Dependecynizer()
        dependency_tokenizer.fit_on_dependency(dependency_all)

        print('>>> saving {0} dependency_tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir, 'bertdependency2idx.pkl'), 'wb') as f:
            pickle.dump(dependency_tokenizer.dependency2idx, f)
    return dependency_tokenizer



def build_image_relizer(data_dir):

    if os.path.exists(os.path.join(data_dir, 'image_rel2idx.pkl')):
        print('>>> loading {0} tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir,'image_rel2idx2idx.pkl'), 'rb') as f:
            image_rel_tokenizer = pickle.load(f)
            image_rel_tokenizer = image_reltionnizer(dependency2idx=image_rel_tokenizer)
    else:
        filenames = ['train/custom_data_info.json']
        for filename in filenames:
            rel_file = data_dir+ filename
            rel_file = open(rel_file,"r")
            image_rel_info = json.load(rel_file)
            image_rel_list = image_rel_info["ind_to_predicates"]
            image_rel_list  = ','.join(image_rel_list)

        image_rel_tokenizer = image_reltionnizer()
        image_rel_tokenizer.fit_on_relation(image_rel_list,split_str=",")
        print('>>> saving {0} image_rel_tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir, 'image_rel_tokenizer2idx.pkl'), 'wb') as f:
            pickle.dump(image_rel_tokenizer.dependency2idx, f)
    return image_rel_tokenizer



def build_dependency_matrix(dependency2idx, dependency_dim,data_name,dependecy_type):
    embedding_matrix_file_name = '{0}/{1}_{2}_bertdependency_matrix.pkl'.format(data_name,dependency_dim,dependecy_type)
    print(embedding_matrix_file_name)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading edge vectors ...')
        embedding_matrix = np.zeros((len(dependency2idx), dependency_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(dependency_dim), 1 / np.sqrt(dependency_dim), (1, dependency_dim))
        # embedding_matrix[1, :] = np.zeros(), (1, dependency_dim))

        print('building edge_matrix:', embedding_matrix_file_name)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

def build_position_matrix(position2idx, position_dim, type,dependency_type):


    embedding_matrix_file_name = '{0}_{1}_{2}_bertposition_matrix.pkl'.format(type, str(position_dim),dependency_type)

    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading edge vectors ...')
        embedding_matrix = np.zeros((len(position2idx), position_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(position_dim), 1 / np.sqrt(position_dim), (1, position_dim))
        # embedding_matrix[1, :] = np.random.uniform(-0.25, 0.25, (1, position_dim))
        print('building position_matrix:', embedding_matrix_file_name)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

class Positionnizer(object):
    def __init__(self, position2idx=None):
        if position2idx is None:
            self.position2idx = {}
            self.idx2position = {}
            self.idx = 0
            self.position2idx['<pad>'] = self.idx
            self.idx2position[self.idx] = '<pad>'
            self.idx += 1
            self.position2idx['<unk>'] = self.idx
            self.idx2position[self.idx] = '<unk>'
            self.idx += 1
            self.position2idx['<CLS>'] = self.idx
            self.idx2position[self.idx] = '<CLS>'
            self.idx += 1
            self.position2idx['<SEP>'] = self.idx
            self.idx2position[self.idx] = '<SEP>'
            self.idx += 1
        else:
            self.position2idx = position2idx
            self.idx2position = {v: k for k, v in position2idx.items()}

    def fit_on_position(self, syntax_positions):

        for syntax_position in syntax_positions:
            if syntax_position not in self.position2idx:
                self.position2idx[syntax_position] = self.idx
                self.position2idx[syntax_position] = self.idx
                self.idx2position[self.idx] = syntax_position
                self.idx += 1

    def position_to_index(self,position_sequence):

        # position_sequence = position_sequence.astype(np.str)
        # print(position_sequence)
        position_matrix=np.zeros_like(position_sequence)
        unknownidx = 1
        for index,clow in enumerate(position_sequence):

            position_matrix[index]= [self.position2idx[str(w)] if str(w) in self.position2idx else unknownidx for w in clow]

        # print(self.position2idx)
        # position_matrix = [self.position2idx["<CLS>"]] + position_matrix+[self.position2idx["<SEP>"]]
        return position_matrix




class Dependecynizer(object):
    def __init__(self, dependency2idx=None):
        if dependency2idx is None:
            self.dependency2idx = {}
            self.idx2dependency = {}
            self.idx2dependency_number={}
            self.idx = 0
            self.dependency2idx['<pad>'] = self.idx
            self.idx2dependency[self.idx] = '<pad>'
            self.idx2dependency_number['<pad>']=1
            self.idx += 1
            self.dependency2idx['<unk>'] = self.idx
            self.idx2dependency[self.idx] = '<unk>'
            self.idx2dependency_number['<unk>'] = 1
            self.idx += 1
            self.dependency2idx['sptype'] = self.idx
            self.idx2dependency[self.idx] = 'sptype'
            self.idx2dependency_number['sptype'] = 1
            self.idx += 1

        else:
            self.dependency2idx = dependency2idx
            self.idx2dependency = {v: k for k, v in dependency2idx.items()}
            self.idx2dependency_number = {v: k for k, v in dependency2idx.items()}
        self.idx2dependency_number = {}
    def fit_on_dependency(self, dependency_edge,split_str=' '):
        dependency_edges = dependency_edge.lower()
        dependency_edges = dependency_edges.split(split_str)
        for dependency_edge in dependency_edges:
            if dependency_edge not in self.dependency2idx:
                self.dependency2idx[dependency_edge] = self.idx
                self.idx2dependency[self.idx] = dependency_edge
                self.idx2dependency_number[dependency_edge]=1
                self.idx += 1
            else:
                self.idx2dependency_number[dependency_edge] += 1
    def dependency_to_index(self,dependency_edge,idx2gragh_dir):
        # print(dependency_edge)

        edge_matrix_undir = np.zeros_like(idx2gragh_dir, dtype= int)
        matrix_len = (edge_matrix_undir.shape)[0]

        unknownidx = 1
        for i in dependency_edge:
            try:
                if (matrix_len>int(i[0]))&(matrix_len>int(i[2])):

                    edge_matrix_undir[i[2]][i[0]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx
                    edge_matrix_undir[i[0]][i[2]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx

            except IndexError:
                print(matrix_len)
                print(dependency_edge)
        # edge_matrix_undir[0,:] = [0]*edge_matrix_undir.shape[0]
        # edge_matrix_undir[:,-1] = [0] * edge_matrix_undir.shape[0]
        # edge_matrix_undir[:,0] = [0] * edge_matrix_undir.shape[0]
        # edge_matrix_undir[-1,:] = [0] * edge_matrix_undir.shape[0]

        # return edge_matrix,edge_matrix_re,edge_matrix_undir
        # print(edge_matrix_undir)
        return edge_matrix_undir


class pospeech_tokennizer(object):
    def __init__(self, dependency2idx=None):
        if dependency2idx is None:
            self.dependency2idx = {}
            self.idx2dependency = {}
            self.idx2dependency_number={}
            self.idx = 0
            self.dependency2idx['<pad>'] = self.idx
            self.idx2dependency[self.idx] = '<pad>'
            self.idx2dependency_number['<pad>']=1
            self.idx += 1
            self.dependency2idx['<unk>'] = self.idx
            self.idx2dependency[self.idx] = '<unk>'
            self.idx2dependency_number['<unk>'] = 1
            self.idx += 1
            self.dependency2idx['sptype'] = self.idx
            self.idx2dependency[self.idx] = 'sptype'
            self.idx2dependency_number['sptype'] = 1
            self.idx += 1

        else:
            self.dependency2idx = dependency2idx
            self.idx2dependency = {v: k for k, v in dependency2idx.items()}
            self.idx2dependency_number = {v: k for k, v in dependency2idx.items()}
    def fit_on_dependency(self, dependency_edge,split_str=' '):
        dependency_edges = dependency_edge.lower()
        dependency_edges = dependency_edges.split(split_str)
        for dependency_edge in dependency_edges:
            if dependency_edge not in self.dependency2idx:
                self.dependency2idx[dependency_edge] = self.idx
                self.idx2dependency[self.idx] = dependency_edge
                self.idx2dependency_number[dependency_edge]=1
                self.idx += 1
            else:
                self.idx2dependency_number[dependency_edge] += 1
    def pospeech_to_index(self,pospeech_data_pack):
        # print(dependency_edge)

        pospeech_list = np.zeros_like(pospeech_data_pack, dtype= int)

        unknownidx = 1

        for  index,w in enumerate(pospeech_data_pack):
            if str(w) in self.dependency2idx:
                idx = self.dependency2idx[str(w)]
            else:
                idx = unknownidx

            pospeech_list[index]=idx

        return pospeech_list



class image_reltionnizer(object):
    def __init__(self, dependency2idx=None):
        if dependency2idx is None:
            self.dependency2idx = {}
            self.idx2dependency = {}
            self.idx2dependency_number={}
            self.idx = 0
            self.dependency2idx['<pad>'] = self.idx
            self.idx2dependency[self.idx] = '<pad>'
            self.idx2dependency_number['<pad>']=1
            self.idx += 1
            self.dependency2idx['<unk>'] = self.idx
            self.idx2dependency[self.idx] = '<unk>'
            self.idx2dependency_number['<unk>'] = 1
            self.idx += 1
            self.dependency2idx['sptype'] = self.idx
            self.idx2dependency[self.idx] = 'sptype'
            self.idx2dependency_number['sptype'] = 1
            self.idx += 1

        else:
            self.dependency2idx = dependency2idx
            self.idx2dependency = {v: k for k, v in dependency2idx.items()}
            self.idx2dependency_number = {v: k for k, v in dependency2idx.items()}
        self.idx2dependency_number = {}
    def fit_on_relation(self, dependency_edge,split_str=' '):
        dependency_edges = dependency_edge.lower()
        dependency_edges = dependency_edges.split(split_str)
        for dependency_edge in dependency_edges:
            if dependency_edge not in self.dependency2idx:
                self.dependency2idx[dependency_edge] = self.idx
                self.idx2dependency[self.idx] = dependency_edge
                self.idx2dependency_number[dependency_edge]=1
                self.idx += 1
            else:
                self.idx2dependency_number[dependency_edge] += 1
    def relation_to_graph(self,image_feature_len,rel_labels):
        relation_matrix= np.zeros([image_feature_len,image_feature_len], dtype= int)
        # print(relation_matrix.shape)
        # exit()
        # print(self.dependency2idx)
        for  rel_list in rel_labels:
            relation_matrix[rel_list[0],rel_list[2]] =self.dependency2idx[rel_list[1]]
            relation_matrix[rel_list[2], rel_list[0]] = self.dependency2idx[rel_list[1]]
        # print(relation_matrix)
        image_mask = np.where(relation_matrix!=0,1,0)
        return relation_matrix,image_mask


def build_dependency_matrix(dependency2idx, dependency_dim,data_name,dependecy_type):
    embedding_matrix_file_name = '{0}/{1}_{2}_matrix.pkl'.format(data_name,dependency_dim,dependecy_type)
    print(embedding_matrix_file_name)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading edge vectors ...')
        embedding_matrix = np.zeros((len(dependency2idx), dependency_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(dependency_dim), 1 / np.sqrt(dependency_dim), (1, dependency_dim))
        # embedding_matrix[1, :] = np.zeros(), (1, dependency_dim))

        print('building edge_matrix:', embedding_matrix_file_name)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

def build_position_matrix(position2idx, position_dim, type,dependency_type):


    embedding_matrix_file_name = '{0}_{1}_{2}_bertposition_matrix.pkl'.format(type, str(position_dim),dependency_type)

    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading edge vectors ...')
        embedding_matrix = np.zeros((len(position2idx), position_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(position_dim), 1 / np.sqrt(position_dim), (1, position_dim))
        # embedding_matrix[1, :] = np.random.uniform(-0.25, 0.25, (1, position_dim))
        print('building position_matrix:', embedding_matrix_file_name)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix




