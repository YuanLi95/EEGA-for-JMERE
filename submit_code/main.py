#coding utf-8
import copy
import os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange

from utils.data import load_data_instances, DataIterator
from models.AMGN import AMGNetwork
from baseline_models.UMGF import MMNerModel
from baseline_models.AGBANRe import AGBAN
from utils.utils import Metric
from utils.syntactic_utils import build_dependency_matrix,build_position_matrix,build_positionizer,build_dependencyizer,\
    build_image_relizer,image_reltionnizer,build_Part_of_Speechlizer
import  numpy as np
import  codecs
import  time
import pickle
model_list = {"AMGNetwork":AMGNetwork,"UMGF":MMNerModel, "AGBAN":AGBAN}
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args,position_tokenizer,dependency_tokenizer,rel_tokenizer,pospeech_tokenizer,dependency_embedding,position_embedding,rel_image_embedding,pospeech_embedding):

    # load dataset
    text_train_path = args.prefix  + 'train.json'
    text_val_path = args.prefix + '/val.json'
    text_test_path = args.prefix +  '/test.json'
    image_train_path = args.image_path+"/train/"
    image_val_path = args.image_path + "/val"
    image_test_path = args.image_path + "/test/"


    instances_train = load_data_instances(text_train_path,image_train_path, position_tokenizer, dependency_tokenizer,rel_tokenizer,pospeech_tokenizer, args)
    instances_val = load_data_instances(text_val_path,image_val_path, position_tokenizer, dependency_tokenizer,rel_tokenizer,pospeech_tokenizer, args)
    instances_test = load_data_instances(text_test_path,image_test_path,  position_tokenizer, dependency_tokenizer,rel_tokenizer,pospeech_tokenizer, args)
    # exit()
    # random.shuffle(instances_train)


    trainset = DataIterator(instances_train, args)
    valset = DataIterator(instances_val, args)
    testset = DataIterator(instances_test, args)
    f_out = codecs.open('log/'  + ' with_no_image_{0}_val.txt'.format(args.model), 'a+', encoding="utf-8")


    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if args.model!="AMGNetwork":
        model = model_list[args.model](args).to(args.device)
        # model = model_list[args.model](args)

    else:
        model =model_list[args.model](args,dependency_embedding,position_embedding,
                          rel_image_embedding,pospeech_embedding).to(args.device)
        # model = model_list[args.model](args, dependency_embedding, position_embedding,
        #                                rel_image_embedding, pospeech_embedding)
    # print(model)
    # parameters = list(model.parameters())
    # parameters = filter(lambda x: x.requires_grad, parameters)
    # for i in model.state_dict():
    #     print(i)
    # # print(model.parameters())
    # exit()
    bert_params = list(map(id, model.bert.parameters()))
    my_self_params = filter(lambda p: id(p) not in bert_params,
                            model.parameters())
    # print(my_self_params)

    optimizer = torch.optim.Adam([{"params":my_self_params,"lr":args.lr},
                                  {"params": model.bert.parameters(), "lr": 2e-5}
                                  ])


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decline, gamma=0.5, last_epoch=-1)

    best_joint_f1 = 0
    best_joint_epoch = 0
    test_f1 = 0
    test_p = 0
    test_r = 0
    best_test_model = None
    dev_test_model = None
    for indx in range(args.epochs):

        print('Epoch:{}'.format(indx))
        train_all_loss = 0.0
        train_ot_loss=0.0
        for j in trange(trainset.batch_count):
            bert_tokens, lengths, token_masks, sens_lens, token_ranges, token_dependency_masks, \
            token_syntactic_position, token_edge_data, token_frequency_graph,pospeech_tokens, image_rel_matrix, image_rel_mask, image_feature, entity_tags, tags\
                = trainset.get_batch(j)
            #
            # print(model)
            # exit()

            if args.model=="AMGNetwork":
                preds, ot_loss_all = model(bert_tokens, token_masks, token_dependency_masks, \
                                           token_syntactic_position, token_edge_data, token_frequency_graph,
                                           pospeech_tokens, image_rel_matrix, image_rel_mask, image_feature,
                                           )
            elif args.model=="AGBAN":
                #增加v_modal_label和t_modal_label
                batch_size = bert_tokens.shape[0]
                v_modal_label = torch.zeros((batch_size * 10)).long().cuda()
                t_modal_label = torch.ones((batch_size * args.max_sequence_len)).long().cuda()
                # Pad modal label with -1
                v_mask = torch.ones((batch_size, 10)).bool()
                v_pad_mask = (1 - v_mask.int()).bool().view(-1).cuda()
                v_modal_label = v_modal_label.masked_fill(v_pad_mask, -1).cuda() # pad position should with zero attion score
                mask =token_masks
                t_pad_mask = (1 - mask.int()).bool().view(-1).cuda()
                t_modal_label = t_modal_label.masked_fill(t_pad_mask, -1).cuda()
                preds, ot_loss_all = model(bert_tokens, token_masks, token_dependency_masks, \
                                           token_syntactic_position, token_edge_data, token_frequency_graph,
                                           pospeech_tokens, image_rel_matrix, image_rel_mask, image_feature,t_modal_label,v_modal_label
                                           )

            else:
                ot_loss_all = 0.00
                preds = model(bert_tokens, token_masks, token_dependency_masks, \
                              token_syntactic_position, token_edge_data, token_frequency_graph, pospeech_tokens,
                              image_rel_matrix, image_rel_mask, image_feature,
                              )



            batch_max_lengths = torch.max(lengths)
            # preds =
            preds = preds[:, :batch_max_lengths, :batch_max_lengths]
            tags = tags[:,:batch_max_lengths,:batch_max_lengths]
            # exit()
            preds_flatten = preds.reshape([-1, preds.shape[3]])
            # print(tags.shape)
            # exit()
            tags_flatten = tags.reshape([-1])

            loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)
            if isinstance(ot_loss_all,list):
                for i in ot_loss_all:
                    train_ot_loss+=i
                    loss+=i
            else:
                train_ot_loss += ot_loss_all
                loss += ot_loss_all
            train_all_loss += loss
            optimizer.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)

            optimizer.step()
        scheduler.step()
        print('this epoch train loss :{0}  ot_loss:{1}'.format(train_all_loss,train_ot_loss))
        # print("------------------this is train result-------------------------------------")
        # # _, _, _, _ = eval(model, trainset, args)
        #
        print("------------------this is dev result-------------------------------------")
        joint_precision, joint_recall, joint_f1,dev_loss,dev_entity_result = eval(model, valset, args)
        print("------------------this is test result-------------------------------------")
        test_joint_precision, test_joint_recall, test_joint_f1, _,test_entity_result = eval(model, testset, args)
        if joint_f1 > best_joint_f1:
            best_joint_f1 = joint_f1
            best_joint_epoch = indx
            dev_test_model = copy.deepcopy(model)

        if test_joint_f1 > test_f1:
            test_f1 = test_joint_f1
            test_p = test_joint_precision
            test_r = test_joint_recall
            print("best test")
            best_test_model = copy.deepcopy(model)
        print("11111111111111")

        print('this poch:\t dev {} loss: {:.5f}\n\n'.format(args.task, dev_loss))
    model_path = args.model_dir + args.model + args.task + "dev_for_test_f1"+str(best_joint_f1) + "dev" + '.pt'

    torch.save(dev_test_model, model_path)


    best_test_model_path = args.model_dir + args.model + args.task  + "best_test_f1" + str(
        test_f1) + '.pt'
    torch.save(best_test_model, best_test_model_path)

    arguments = " "
    for arg in vars(args):
        if arg== "dependency_embedding":
            continue
        elif arg == "position_embedding":
            continue
        else:
            arguments += '{0}: {1} '.format(arg, getattr(args, arg))


    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    f_out.write('time:{0}\n'.format(time_str))
    test_model = torch.load(model_path).to(args.device)



    dev_for_test_precision, dev_for_test_recall, dev_for_test_f1, _,dev_for_test_entity = eval(test_model, testset, args)

    best_dev_precision, best_dev_recall, best_dev_f1, _, best_dev_entity = eval(test_model, valset,
                                                                                                args)


    best_test_model_path = args.model_dir + args.model + args.task  + "best_test_f1" + str(test_f1) + '.pt'
    best_test_model =torch.load(best_test_model_path).to(args.device)
    best_test_precision, best_test_recall, best_test_f1, _, best_test_entity = eval(best_test_model, testset,
                                                                                                args)
    f_out.write(arguments)
    f_out.write("\n")
    f_out.write('dev_max_test_acc: {0}, dev_max_test_recall:{1}, dev_max_f1: {2}\n'.format(dev_for_test_precision,
                                                                                           dev_for_test_recall,
                                                                                             dev_for_test_f1))

    f_out.write('dev_max_test_acc_entity: {0}, dev_max_test_recall_entity:{1}, dev_max_f1_entity: {2}\n'.format(dev_for_test_entity[0],
                                                                                           dev_for_test_entity[1],
                                                                                           dev_for_test_entity[2],))

    f_out.write('best_dev_acc: {0}, best_dev_recall:{1}, best_dev_f1: {2}\n'.format(best_dev_precision,
                                                                                           best_dev_recall,
                                                                                           best_dev_f1))

    f_out.write('best_dev_acc_entity: {0}, best_dev_recall_entity:{1}, best_dev_entity: {2}\n'.format(
        best_dev_entity[0],
        best_dev_entity[1],
        best_dev_entity[2], ))

    print(('dev_max_test_acc: {0}, dev_max_test_recall:{1}, dev_max_f1: {2}\n'.format(dev_for_test_precision,
                                                                                           dev_for_test_recall,
                                                                                           dev_for_test_f1)))




    f_out.write('best_test_precision: {0}, best_test_recall:{1}, best_test_f1: {2}\n'.format(best_test_precision,
                                                                                           best_test_recall,
                                                                                           best_test_f1))

    f_out.write('best_test_precision_entity: {0}, best_test_recall_entity:{1}, best_test_f1_entity: {2}\n'.format(
        best_test_entity[0],
        best_test_entity[1],
        best_test_entity[2], ))
    f_out.write("\n")

    f_out.close()
    print('best_test_precision: {0}, best_test_recall:{1}, best_test_f1: {2}\n'.format(best_test_precision,
                                                                                           best_test_recall,
                                                                                           best_test_f1))

    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_test_f1))
    print('max test precision:{} ----- recall:{}-------- f1:{}'.format(str(test_p), str(test_r), str(test_f1)))



def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        dev_loss =0.0
        for i in range(dataset.batch_count):
            bert_tokens, lengths, token_masks, sens_lens, token_ranges, token_dependency_masks, \
            token_syntactic_position, token_edge_data, token_frequency_graph, pospeech_tokens, image_rel_matrix, image_rel_mask, image_feature, entity_tags, tags \
                = dataset.get_batch(i)


            if args.model=="AMGNetwork":
                prediction, ot_loss,T_wd,T_gwd,att = model(bert_tokens, token_masks, token_dependency_masks, \
                                           token_syntactic_position, token_edge_data, token_frequency_graph,
                                           pospeech_tokens, image_rel_matrix, image_rel_mask, image_feature,
                                           )
            elif args.model=="AGBAN":
                #增加v_modal_label和t_modal_label
                batch_size = bert_tokens.shape[0]
                v_modal_label = torch.zeros((batch_size * 10)).long().cuda()
                t_modal_label = torch.ones((batch_size * args.max_sequence_len)).long().cuda()
                # Pad modal label with -1
                v_mask = torch.ones((batch_size, 10)).bool()
                v_pad_mask = (1 - v_mask.int()).bool().view(-1).cuda()
                v_modal_label = v_modal_label.masked_fill(v_pad_mask, -1).cuda() # pad position should with zero attion score
                mask =token_masks
                t_pad_mask = (1 - mask.int()).bool().view(-1).cuda()
                t_modal_label = t_modal_label.masked_fill(t_pad_mask, -1).cuda()
                prediction, ot_loss = model(bert_tokens, token_masks, token_dependency_masks, \
                                           token_syntactic_position, token_edge_data, token_frequency_graph,
                                           pospeech_tokens, image_rel_matrix, image_rel_mask, image_feature,t_modal_label,v_modal_label
                                           )

            else:
                ot_loss = 0.00
                prediction = model(bert_tokens, token_masks, token_dependency_masks, \
                              token_syntactic_position, token_edge_data, token_frequency_graph, pospeech_tokens,
                              image_rel_matrix, image_rel_mask, image_feature,
                              )
            if i==57:
                print(T_wd)
                print(T_wd.shape)
                print(T_gwd)
                print(T_gwd.shape)
                print(att)
                print(att.shape)
                file = open('attention.pickle', 'wb')
                pickle.dump(T_wd.cpu().numpy(), file)
                exit()


            # prediction ,ot_loss = model(bert_tokens,  token_masks, token_dependency_masks, \
            # token_syntactic_position, token_edge_data, token_frequency_graph,pospeech_tokens, image_rel_matrix, image_rel_mask, image_feature,
            #               )

            prediction_argmax = torch.argmax(prediction, dim=3)
            tags_flatten = tags[:, :prediction.shape[1], :prediction.shape[1]].reshape([-1])
            prediction_flatten = prediction.reshape([-1, prediction.shape[3]])
            dev_loss = dev_loss + F.cross_entropy(prediction_flatten, tags_flatten, ignore_index=-1)
            prediction_padded = torch.zeros(prediction.shape[0], args.max_sequence_len, args.max_sequence_len)
            prediction_padded[:, :prediction_argmax .shape[1], :prediction_argmax .shape[1]] =prediction_argmax

            all_preds.append(prediction_padded)
            all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)


        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags()
        entity_results = metric.score_entity()
        print('entity_results\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(entity_results[0], entity_results[1],
                                                                  entity_results[2]))
        print("unified_results"+ '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1, dev_loss,entity_results


def test(args):
    print("Evaluation on testset:")
    text_test_path = args.prefix +  '/test.json'
    image_test_path = args.image_path + "/test/"
    # model_path = args.model_dir + args.model + args.task+"dev" + '.pt'
    model_path = args.model_dir+"AMGNetworktripletbest_test_f10.5500747384155455.pt"
    print(model_path)

    model = torch.load(model_path).to(args.device)



    print(model)
    model.eval()

    instances_test = load_data_instances(text_test_path,image_test_path,  position_tokenizer, dependency_tokenizer,rel_tokenizer,pospeech_tokenizer, args)
    testset = DataIterator(instances_test , args)
    eval(model, testset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="./unified_tags_datasets/no_none_unified_tags_txt/",
                        help='dataset and embedding path prefix')

    parser.add_argument('--image_path', type=str, default="./unified_tags_datasets/no_none_image_path/",
                        help='dataset and embedding path prefix')

    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="test", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--max_sequence_len', type=int, default=70,
                        help='max length of a sentence')
    parser.add_argument('--max_image_feature_len', type=int, default=10,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu mps')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=70,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=26,
                        help='label number')
    parser.add_argument('--dependency_embed_dim', type=int, default=100,
                        )
    parser.add_argument('--frequency_embed_dim', type=int, default=100,
                        )

    parser.add_argument('--image_rel_embed_dim', type=int, default=100,
                        )

    parser.add_argument('--position_embed_dim', type=int, default=100,
                        )

    parser.add_argument('--pospeech_embed_dim', type=int, default=100,
                        )
    parser.add_argument('--lr', type=float, default=2e-5,
                        )

    parser.add_argument('--feature_image', type=int, default=4096,
                        help='dimension of pretrained image feature')

    parser.add_argument('--trans_image_dro', type=float, default=0.4,
                        help='')
    parser.add_argument('--twd_weight', type=float, default=0.2,
                        help='')
    parser.add_argument('--attention_heads', type=int, default=12,
                        help='attribute transformer attention')

    parser.add_argument('--cross_attention_heads', type=int, default=12,
                        help='attribute transformer attention')



    parser.add_argument('--seed', type=int, default=19)

    parser.add_argument('--alpha_adjacent', type=float, default=0.4)
    parser.add_argument('--nhops', type=int, default=1,
                        help='inference times')

    parser.add_argument('--decline', type=int, default=30, help="number of epochs to decline")

    parser.add_argument('--model', type=str, default="AMGNetwork", help="number of epochs to decline")



    args = parser.parse_args()
    setup_seed(args.seed)

    position_tokenizer = build_positionizer(args.prefix)
    dependency_tokenizer = build_dependencyizer(args.prefix)
    rel_tokenizer = build_image_relizer(args.image_path)
    pospeech_tokenizer = build_Part_of_Speechlizer(args.prefix)


    dependency_embedding = build_dependency_matrix(dependency_tokenizer.dependency2idx,
                                                   args.dependency_embed_dim, args.prefix,
                                                   "dependency")
    position_embedding = build_position_matrix(position_tokenizer.position2idx,
                                               args.position_embed_dim, args.prefix,
                                               "position")

    rel_image_embedding = build_dependency_matrix(rel_tokenizer.dependency2idx,
                                               args.image_rel_embed_dim, args.image_path,
                                               "iamge_rel")
    pospeech_embedding = build_dependency_matrix(pospeech_tokenizer.dependency2idx,
                                                  args.pospeech_embed_dim, args.prefix,
                                                  "pospeech")

    if args.mode == 'train':
        train(args,position_tokenizer,dependency_tokenizer,rel_tokenizer,pospeech_tokenizer,dependency_embedding,position_embedding,rel_image_embedding,pospeech_embedding)
        # test(args)
    else:
        test(args)
