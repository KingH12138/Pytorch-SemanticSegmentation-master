import pandas as pd
import torch


def torch2onnx(model_path,outputs_path, inputs_shape, device='cuda'):
    """
    tips:你保存的模型一定是直接通过torch.save(model)保存的——也就是保存整个模型
    :param model_path:.pth文件路径
    :param outputs_path:输出的.onnx文件路径
    :param inputs_shape:(bs,x,x,h,w)
    :param device:cuda or cpu
    :return:
    """
    dummy_input = torch.randn(*inputs_shape, device=device)
    input_names = ["input"]
    output_names = ["output"]
    model = torch.load(model_path)
    torch.onnx.export(model, dummy_input, outputs_path, verbose=True, input_names=input_names, output_names=output_names)


def df_generator(epoches, tags, save_path=None):
    """
    2022/5/24:update Auc and mAP
    """
    keys = ['Epoch', 'dice_score']
    values = [range(1, epoches + 1)] + tags
    data = dict(zip(keys, values))
    df = pd.DataFrame(data=data)
    if save_path:
        df.to_csv(save_path, encoding='utf-8')
    return df




