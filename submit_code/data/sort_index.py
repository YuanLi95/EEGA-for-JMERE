import  json
import pickle
import gc
def process(data_set):
    text_path = "./no_none_unified_tags_txt/{0}.json".format(data_set)
    prediction_path = "./image_path/{0}/custom_prediction.json".format(data_set)
    feature_path = "./image_path/{0}/output_feature.pickle".format(data_set)
    prediction_data = open(prediction_path, "r")
    load_prediction = json.load(prediction_data)

    # exit()
    new_prediction = []
    new_output_feature = []
    f = open(text_path, encoding='UTF-8')
    f_lines = json.load(f)
    # for key,value in load_prediction.items():
    #     # print(key)
    #     # print(value)
    #     # exit()

    with open(feature_path,"rb") as file:
        load_feature = pickle.load(file)

        for index,line in enumerate(f_lines):
            print(line)
            # exit()

            print(line["img_id"])

            # print(len(load_feature[line["img_id"]]))
            new_prediction.append(load_prediction[line["img_id"]])
            new_output_feature.append(load_feature[line["img_id"]])
    print(len(new_prediction))
    print(len(new_output_feature))
    new_file = open("./no_none_image_path/{0}/new_prediction.pickle".format(data_set), "wb")
    pickle.dump(new_prediction,new_file)
    new_file.close()
    new_file =  open("./no_none_image_path/{0}/new_output_feature.pickle".format(data_set),"wb")
    pickle.dump(new_output_feature,new_file)
    new_file.close()



if __name__ == '__main__':
    process("test")
    print("this is test over")
    # process("val")
    # print("this is val over")
    # process("train")
    # print("this is train over")

