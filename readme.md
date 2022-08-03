# Pytorch-SemanticSegmentation-master使用方法


## 一.配置环境

### 

先创建好环境并安装好pytorch，激活环境(至于要安装什么pytorch版本，可以参考[这篇文章](https://blog.csdn.net/Killer_kali/article/details/123173414?spm=1001.2014.3001.5501)：

```

conda create -n torchclassify python=3.8
conda activate torchclassify
conda install pytorch=1.8 torchvision cudatoolkit=10.2

```

进入项目目录：

```
cd Pyotrch-ImageClassification-master
```

安装相关包库：

```
pip install -r requirements.txt
```

tips:

如果prettytable库无法安装，可以尝试如下命令：

```
python -m pip install -U prettytable
```

## 二.运行predict测试文件

## 三.制作并训练自己的数据
