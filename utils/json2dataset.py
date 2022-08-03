import os
import shutil
from tqdm import tqdm

"""
该脚本结合labelme python gui keypoints/Segmentation 打标签工具生成binaryscatter格式数据集
"""


def trans(json_dir):
    for json_name in tqdm(os.listdir(json_dir)):
        path = r"{}/{}".format(json_dir,json_name)
        os.system("labelme_json_to_dataset {}".format(path))


def extract_rename_save(src_dir,img_dir,label_dir):
    idx = 0
    for name in tqdm(os.listdir(src_dir)):
        path = src_dir + '/{}'.format(name)
        if os.path.isdir(path):
            img_path = os.path.join(path,'img.png')
            label_path = os.path.join(path,'label.png')
            try:
                shutil.move(img_path, os.path.join(img_dir,"{}.png".format(idx)))
                shutil.move(label_path, os.path.join(label_dir, "{}.png".format(idx)))
                idx += 1
            except:
                continue


json_dir = r'F:\AI智慧育种数据库\origin\Annotations'
img_dir = r'D:\PythonCode\Pytorch-UNet-master\data\imgs'
label_dir = r'D:\PythonCode\Pytorch-UNet-master\data\masks'
# 先后运行两个函数
trans(json_dir)
extract_rename_save(
    json_dir,
    img_dir,
    label_dir
)

