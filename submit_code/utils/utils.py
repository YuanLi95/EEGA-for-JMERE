import multiprocessing
import pickle
import numpy as np
import sklearn

id2unified = {0: 'none', 1: 'per', 2: 'org', 3: 'misc', 4: 'loc', 5: 'parent',
              6: 'siblings', 7: 'couple', 8: 'neighbor', 9: 'peer', 10: 'charges',
              11: 'alumi', 12: 'alternate_names', 13: 'place_of_residence',
              14: 'place_of_birth', 15: 'member_of', 16: 'subsidiary', 17: 'locate_at',
              18: 'contain', 19: 'present_in', 20: 'awarded', 21: 'race', 22: 'religion',
              23: 'nationality', 24: 'part_of', 25: 'held_on'}
entity2id ={"per":1,"org":2,"misc":3, "loc":4}

id2entity ={1: 'per', 2: 'org', 3: 'misc', 4: 'loc'}

id2relation={5: 'parent', 6: 'siblings', 7: 'couple', 8: 'neighbor', 9: 'peer', 10: 'charges',
              11: 'alumi', 12: 'alternate_names', 13: 'place_of_residence',
              14: 'place_of_birth', 15: 'member_of', 16: 'subsidiary', 17: 'locate_at',
              18: 'contain', 19: 'present_in', 20: 'awarded', 21: 'race', 22: 'religion',
              23: 'nationality', 24: 'part_of', 25: 'held_on'}


def get_aspects(tags, length, ignore_index=-1):
    spans = []
    start = -1
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 1:
            if start == -1:
                start = i
        elif tags[i][i] != 1:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans


class Metric():
    def __init__(self, args, predictions, goldens, bert_lengths, sen_lengths, tokens_ranges, ignore_index=-1):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = -1
        self.data_num = len(self.predictions)

    def get_spans(self, tags, length, token_range, type=dict):
        spans = []
        start = -1
        # for i in tags:
        #     print(i)
        label_tags = -1
        for i in range(length):

            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                # label_tags =-1
                continue
            elif tags[l][l] in id2entity:

                if start == -1:
                    start = i
                    label_tags = int(tags[l][l])
            elif tags[l][l] not in  id2entity:
                if start != -1:
                    assert (label_tags!=-1)
                    spans.append([start, i - 1,label_tags])
                    start = -1

        if start != -1:
            spans.append([start, length - 1, label_tags])
        return spans

    def find_triplet(self, tags, entity_spans, token_ranges):
        triplets = []
        # print(entity_spans)
        for al, ar,a_type in entity_spans:
            for pl,pr,p_type in entity_spans:
                if al==pl:
                    continue
                tag_num = [0] * len(id2unified)
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if sum(tag_num[5:]) == 0:
                    continue
                relation_type = (tag_num[5:].index(max(tag_num[5:]))+5)
                if relation_type == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    input()
                triplets.append([al, ar,a_type, pl, pr,p_type, relation_type])
        return triplets

    def score_entity(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], entity2id)
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))
            # print(golden_set)

            predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], entity2id)
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))
            # print(predicted_set)
            # exit()
        print(len(golden_set))
        print(len(predicted_set))

        # exit()
        correct_num = len(golden_set & predicted_set)
        print(correct_num)

        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1


    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_entity_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], entity2id)

            golden_tuples = self.find_triplet(self.goldens[i], golden_entity_spans, self.tokens_ranges[i])

            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_entity_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], entity2id)

            predicted_tuples = self.find_triplet(self.predictions[i], predicted_entity_spans, self.tokens_ranges[i])
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))
        print(len(golden_set))
        print(len(predicted_set))
        correct_num = len(golden_set & predicted_set)
        print(correct_num)

        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1