import os
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def get_confusion_matrix(y_pred, y_label, cls_num, fig_save_dir=None):
    #############
    # TP    FN  #
    # FP    TN  #
    #############
    """
    :param y_pred: a prediction numpy array just like [0,1,3,5,7]
    :param y_label:a label numpy array just like [0,1,3,5,7]
    :return:None
    """
    sns.set_style("whitegrid")
    cls_list = range(cls_num)
    cmatrix = confusion_matrix(y_label, y_pred,labels=cls_list)
    df = pd.DataFrame(data=cmatrix,index=cls_list,columns=cls_list)
    plt.figure(dpi=300,figsize=(4,3))
    plt.rc('font', family='Times New Roman', size=14)
    sns.heatmap(data=df)
    if fig_save_dir:
        plt.savefig(fig_save_dir + '/heatmap.svg')
    return cmatrix


def dataset_distribution(refer_path, cls_path, save_dir=None):
    """
    tips:make sure your refer.csv have "label" key.
    """
    sns.set_style("whitegrid")
    df = pd.read_csv(refer_path,encoding='utf-8')
    cls_list = open(cls_path, 'r').read().split('\n')
    plt.figure(dpi=300,figsize=(4,3))
    plt.rc('font', family='Times New Roman', size=14)
    count_df = pd.DataFrame(df['label'].value_counts()).T
    cols = list(count_df.columns)
    cols = [cls_list[i] for i in cols]
    count_df.columns = cols
    sns.barplot(data=count_df)
    if save_dir:
        plt.savefig(save_dir + '/dataset_distribution.svg')


def log_plot(df, save_fig_dir):
    """
    :param df: indicators的dataframe——通过pandas以utf-8编码格式读入
    :param save_fig_dir: 存储路径
    """
    sns.set_style("whitegrid")
    indicators = list(df.columns[2:])
    data = df.to_numpy()[:, 2:]
    index = list(df.index)
    df_ = pd.DataFrame(data=data,index=index,columns=indicators)
    plt.rc('font', family='Times New Roman', size=14)
    plt.figure(dpi=300,figsize=(4,3))
    plt.xlabel("Epoch")
    plt.ylabel("Indicator")
    sns.lineplot(data=df_, markers=True)
    plt.savefig(save_fig_dir + '/indicators.svg')

    plt.figure(dpi=300,figsize=(4,3))
    sns.lineplot(data=df,y='Loss',x="Epoch")
    plt.savefig(save_fig_dir + '/loss_epoch.svg')


def generate_feature(test_img_path, resize, model_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    layers = list(model.named_children()) #这儿需要针对每一种model情况而定
    transformer = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])
    with torch.no_grad():
        pil_img = Image.open(test_img_path)
        img_input = transformer(pil_img)
        img_input = img_input.reshape((1, *img_input.shape))
        for i in tqdm(range(len(layers))):
            try:
                layer = layers[:i + 1]
                layer_ = torch.nn.Sequential(*[i[1] for i in layer])
                name = layers[i][0]
                outputs = layer_(img_input.to(device))  # 1,c,h,w
                outputs = outputs.permute(1, 0, 2, 3)  # c,1,h,w
                shape = outputs.shape
                grid_tensor = make_grid(outputs, nrow=int(sqrt(shape[0])))
                save_image(grid_tensor, '{}/{}.jpg'.format(save_dir, name))
            except:
                print("{} can not be plot normally.".format(name))