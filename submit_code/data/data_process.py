import json
# text_list =["./txt/ours_train.txt"]
# text_list = ["./txt/ours_test.txt"]
text_list = ["./txt/ours_val.txt"]
#./txt/ours_test.txt","./txt/ours_val.txt"
# out_list = ".//no_none_unified_tags_txt/train.json"
# out_list = "./no_none_unified_tags_txt/test.json"
out_list = "./no_none_unified_tags_txt/val.json"
out_file = open(out_list,"w")
out_put_file = []
for text_path in text_list:
    f = open(text_path, encoding='UTF-8')
    f_lines = f.readlines()
    my_dict={}
    i = 0
    for index in range(len(f_lines)):
        if (i>=len(f_lines))==True:
            break
        print(index)
        label=[]
        data_line = eval(f_lines[i])
        data_token = data_line["token"]
        relation = data_line["relation"]
        relation_list = relation.split("/")
        tagging_len = len(relation_list)
        begin_index, sec_index = data_line["h"], data_line["t"]
        img_id = data_line['img_id']
        if tagging_len<=3:
            i+=1
            continue
        else:
            beg_tag, sec_tag, rel = relation_list[1:]
            begin_index["tags"] = beg_tag
            sec_index["tags"] = sec_tag
        label.append([{"beg_ent": begin_index, "sec_ent": sec_index, "relation": rel}])
        token_str =  ' '.join(data_token)


        data_next_number=1
        #下一行的token_str
        if i <=(len(f_lines)-2):
            # print("111111111111111111")
            next_token_data = eval(f_lines[i+data_next_number])
            while (' '.join(next_token_data["token"]) ==token_str)==True:

                next_relation = next_token_data["relation"]

                next_relation_list = next_relation.split("/")
                begin_index_next, sec_index_next = next_token_data["h"], next_token_data["t"]
                if len(next_relation_list)<=3:
                    data_next_number += 1
                    break
                else:
                    beg_tag_next, sec_tag_next, rel_next = next_relation_list[1:]
                    begin_index_next["tags"] = beg_tag_next
                    sec_index_next["tags"] = sec_tag_next
                label.append([{"beg_ent": begin_index_next, "sec_ent": sec_index_next, "relation": rel_next}])
                data_next_number+=1
                #更新位置
                if (i + data_next_number)<=(len(f_lines)-1):
                    next_token_data = eval(f_lines[i + data_next_number])
                else:
                    break

        #跳跃index
        i+=data_next_number
        output_line = {"token": data_line["token"], "img_id":img_id, "label_list":label}
        out_put_file.append(output_line)
print(len(out_put_file))
json.dump(out_put_file,out_file)


