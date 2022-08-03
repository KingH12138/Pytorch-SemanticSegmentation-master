"""
"binaryscatter" dataset format:
-src_dir
    -0.png
    -1.png
    ....
-mask_dir
    -0.png
    -1.png
"""
import os
import pandas


class BinaryScatterData():
    def __init__(self, src_dir, mask_dir, cls_path, refer_path):
        """
        :param src_dir:经过json2dataset.py处理后的文件夹中的img.png
        :param raw_mask_dir: 经过json2dataset.py处理后的、只有存储label.png等图片的文件夹的目录
        :param mask_dir:
        :param cls_path:
        :param refer_path:
        """
        super(BinaryScatterData, self).__init__()
        self.src_dir = src_dir
        if not os.path.exists(self.src_dir):
            raise RuntimeError("No src file in your 'src_dir'.Please check your path!")
        self.mask_dir = mask_dir
        if not os.path.exists(self.mask_dir):
            raise RuntimeError("No mask file in your 'mask_dir'.Please check your path!")
        self.cls_path = cls_path
        self.refer_path = refer_path

    def __len__(self) -> int:
        return len(os.listdir(self.src_dir)) + len(os.listdir(self.mask_dir))

    def cls2txt(self):
        # tips:当然你也可以直接把labelme的类别文件重命名为classes.txt后使用
        pass

    def txt2cls(self):
        with open(self.cls_path,'r') as f:
            return f.read().split('\n')

    def generate(self):
        # 生成classes.txt
        if not os.path.exists(self.cls_path):
            self.cls2txt()
        print(f"Attention!You are using f{self.cls_path}.Please make sure you don't have any upgrading action "
              f"for 'classes.txt'")
        # 生成DIF
        data = {'filename':[],'img_path':[],'mask_path':[]}
        for filename in os.listdir(self.src_dir):
            img_path = os.path.join(self.src_dir, filename)
            mask_path = os.path.join(self.mask_dir, filename)
            if not os.path.isfile(img_path):
                RuntimeError("Please make sure 'src_img' and 'mask_img' have same name or there is no 'src_img' in"
                             f"f{img_path}")
            data['filename'].append(filename)
            data['img_path'].append(img_path)
            data['mask_path'].append(mask_path)
        df = pandas.DataFrame(data)
        df.to_csv(self.refer_path,encoding='utf-8')












