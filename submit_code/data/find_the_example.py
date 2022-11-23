import  json
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import  os
import pickle
import numpy as np
text_file = "no_none_unified_tags_txt/val.json"
text_path  = open(text_file,encoding="UTF-8")
text_list = json.load(text_path)


#val
# my_text = ["RT", "@Stuart_PhotoAFC", ":", "Arsenal", "manager", "Arsene", "Wenger", "says", "goodbye", "to", "the", "fans", "at", "Emirates", "Stadium", "."]

# my_text = ["RT", "@g______eeel", ":", "Anthony", "Kiedis", "kissing", "Eddie", "Vedder", ",", "1992"]
# #来自test数据集
# my_text = ["ENewsVideo", ":", "Meghan", "Markle", "and", "Prince", "Harry", "announce", "their", "first", "official", "royal", "tour", ":"]
#备选需要改
# my_text=["RT", "@TeenVogue", ":", "PARDON", "ME", ",", "EXCUSE", "ME", "LILI", "AND", "COLE", "CAME", "TO", "THE", "#", "METGALA", "TOGETHER", "!", "!", "!"]

# my_text = ["RT", "@TheLaunchMag", ":", "10", "years", "ago", "today", "Kobe", "Bryant", "wins", "his", "first", "and", "only", "NBA", "MVP", "award", "for", "regular", "season"]
# my_text = ["RT", "@GloballyCurry30", ":", "Stephen", "Curry", "and", "Michael", "Jordan", "are", "both", "players", "who", "have", "/", "had", "3", "NBA", "Championships", "by", "age", "30"]

# my_text= ["RT", "@TeenVogue", ":", "PARDON", "ME", ",", "EXCUSE", "ME", "LILI", "AND", "COLE", "CAME", "TO", "THE", "#", "METGALA", "TOGETHER", "!", "!", "!"]
my_text =["Madagascar", "/", "African", "South", "East", "Coast", "/", "JFW"]
my_str = " ".join(my_text)
print(my_str)
for index,example in enumerate(text_list):
    print(index)

    sentence= " ".join(example["token"])

    if sentence ==my_str:
        print(example)
        img_id = example["img_id"]
        path = "../img_org/val/"
        img_path = os.path.join(path,img_id)
        lena = mpimg.imread(img_path)
        plt.imshow(lena)
        plt.show()
        break
file_path = "./no_none_image_path/val/new_prediction.pickle"
# print(index)
# df = open(file_path,"rb")
# image_prediction = pickle.load(df)
# print(image_prediction)


print(my_str)