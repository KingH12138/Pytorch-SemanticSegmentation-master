import argparse
import shutil
from datetime import datetime

import torchsummary
from prettytable import PrettyTable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


from data.binaryscatter import BinaryScatterData
from datasets.dataload import get_dataloader
from utils.create_log import logger_get
from utils.draw_utils import *
from utils.report_utils import df_generator, torch2onnx
from model.backbone.UNet import UNet


# 导入参数
def get_arg():
    parser = argparse.ArgumentParser(description='classification parameter configuration(train)')
    parser.add_argument(
        '-t',
        type=str,
        default='Pytorch-SemanticSegmentation-master',
        help='This is your task theme name'
    )
    parser.add_argument(
        '-srcd',
        type=str,
        default=r'F:\PedCut2013_SegmentationDataset\data\completeData\left_images',
        help="src's directory"
    )
    parser.add_argument(
        '-maskd',
        type=str,
        default=r'F:\PedCut2013_SegmentationDataset\data\completeData\left_groundTruth',
        help="mask's directory"
    )
    parser.add_argument(
        '-csvp',
        type=str,
        default=r'D:\PythonCode\Pytorch-SemanticSegmentation-master\data\refer.csv',
        help="DIF(dataset information file)'s path"
    )
    parser.add_argument(
        '-clsp',
        type=str,
        default=r'D:\PythonCode\Pytorch-SemanticSegmentation-master\data\classes.txt',
        help="classes.txt's path"
    )
    parser.add_argument(
        '-tp',
        type=float,
        default=0.9,
        help="train dataset's percent"
    )
    parser.add_argument(
        '-bs',
        type=int,
        default=32,
        help="train dataset's batch size"
    )
    parser.add_argument(
        '-rs',
        type=tuple,
        default=(224, 224),
        help='resized shape of input tensor'
    )
    parser.add_argument(
        '-cn',
        type=int,
        default=1,
        help='the number of classes(no background->at least 1)'
    )
    parser.add_argument(
        '-e',
        type=int,
        default=10,
        help='epoch'
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=0.001,
        help='learning rate'
    )
    parser.add_argument(
        '-nw',
        type=int,
        default=6,
        help='number of workers'
    )
    parser.add_argument(
        '-ld',
        type=str,
        default=r'D:\PythonCode\Pytorch-SemanticSegmentation-master\workdir',
        help="the training log's save directory"
    )

    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # 任务开始前的准备工作
    args = get_arg()  # 得到参数Namespace
    nowtime = datetime.now()  # 获取任务开始时间
    log_dir = "{}/exp_{}_{}_{}_{}_{}".format(args.ld, nowtime.month, nowtime.day, nowtime.hour, nowtime.minute, args.t)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,
                            "exp_{}_{}_{}_{}_{}.log".format(nowtime.month, nowtime.day, nowtime.hour, nowtime.minute,
                                                            args.t))
    file_logger = logger_get(log_path)  # 获取logger
    source_data = BinaryScatterData(args.srcd, args.maskd, args.clsp, args.csvp)
    file_logger.info("Generating classes.txt and DIF file.......")
    try:
        source_data.generate()
        file_logger.info("Done.")
    except:
        file_logger.error("Generate failure!")
    # 训练设备信息
    device_table = ""
    if torch.cuda.is_available():
        device_table = PrettyTable(['number of gpu', 'applied gpu index', 'applied gpu name'], min_table_width=80)
        gpu_num = torch.cuda.device_count()
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name()
        device_table.add_row([str(gpu_num), str(gpu_index), str(gpu_name)])
        file_logger.info('Training device information:\n{}\n'.format(device_table))
    else:
        file_logger.warning("Using cpu......")
        device_table = 'CPU'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------------------------------------------------------------------------------------
    # 数据集信息
    file_logger.info(
        "Use dataset information file:{}\nLoading dataset from path: {}......".format(args.csvp, args.srcd))
    train_dl, valid_dl, samples_num, train_num, valid_num = get_dataloader(args.srcd, args.csvp, args.rs, args.bs
                                                                           , args.nw, args.tp)
    dataset_table = PrettyTable(['number of samples', 'train number', 'valid number', 'percent'], min_table_width=80)
    dataset_table.add_row([samples_num, train_num, valid_num, args.tp])
    file_logger.info("dataset information:\n{}\n".format(dataset_table))
    # ----------------------------------------------------------------------------------------------------------------------
    # 类别信息
    classes = source_data.txt2cls()
    classes_table = PrettyTable(classes, min_table_width=80)
    classes_table.add_row(range(len(classes)))
    file_logger.info("Classes information:\n{}\n".format(classes_table))
    # ----------------------------------------------------------------------------------------------------------------------
    # 训练组件配置
    model = UNet(3, args.cn).to(device)  ##################################################
    optimizer = Adam(params=model.parameters(), lr=args.lr)  ##################################################
    loss_fn = CrossEntropyLoss()  ##################################################
    train_table = PrettyTable(['theme', 'resize', 'batch size', 'epoch', 'learning rate', 'directory of log'],
                              min_table_width=120)
    torchsummary.summary(model,(3, *args.rs), args.bs)
    train_table.add_row([args.t, args.rs, args.bs, args.e, args.lr, args.ld])
    file_logger.info('Train information:\n{}\n'.format(train_table))
    # ----------------------------------------------------------------------------------------------------------------------
    # 开始训练
    file_logger.info("Train begins......")
    losses = []
    valid_losses = []
    shapeuse = ()
    valid_loss = 0.
    best_checkpoint = 0.

    st = datetime.now()
    for epoch in range(args.e):
        model.train()
        train_bar = tqdm(iter(train_dl), ncols=150, colour='blue')
        train_loss = 0.
        i = 0
        for train_data in train_bar:
            x_train, y_train = train_data
            shapeuse = x_train.shape
            x_train = x_train.to(device)    # (bs,3,224,224)
            y_train = y_train.to(device).squeeze(1).long()  # eg:(bs,224,224)
            output = model(x_train)  # eg:(bs,cn,224,224)
            loss = loss_fn(output, y_train)
            optimizer.zero_grad()
            # clone().detach()：可以仅仅复制一个tensor的数值而不影响tensor# 原内存和计算图
            train_loss += loss.clone().detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            # 显示每一批次的loss
            train_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(train_dl)))
            train_bar.set_postfix({"train loss": "%.3f" % loss.data})
            i += 1
        train_loss = train_loss / i
        file_logger.info("Epoch loss:{}".format(train_loss))
        # 最后得到的i是一次迭代中的样本数批数
        losses.append(train_loss)

        model.eval()
        valid_bar = tqdm(iter(valid_dl), ncols=150, colour='blue')
        i = 0
        with torch.no_grad():
            for valid_data in valid_bar:
                x_valid, y_valid = valid_data
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device).squeeze(1).long()  # eg:(bs,224,224)
                output = model(x_valid)  # eg:(bs,cn,224,224)
                loss = loss_fn(output, y_valid)
                valid_loss+=loss.data
                # 显示每一批次的acc/precision/recall/f1
                valid_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(valid_dl)))
                y_valid_ = y_valid.clone().detach()  # y_valid就不必放到gpu上训练了
                i += 1
        # 最后得到的i是一次迭代中的样本数批数,每一次epoch计算一次indicators

        # 验证阶段信息输出
        indicator_table = PrettyTable(['valid_loss'], )
        indicator_table.add_row([valid_loss])
        file_logger.info('\n{}\n'.format(indicator_table))
        # indicator保存
        valid_losses.append(valid_loss)
        # 保存最好的指标的checkpoint
        if valid_loss <= max(valid_losses):  # 如果本次epoch的f1大于了存储f1列表的最大值，那么最好的checkpoint赋值为model
            best_checkpoint = model
        # 保存每次的checkpoint，从而实现断点继训
        os.makedirs("../checkpoints", exist_ok=True)  # 项目根路径下的checkpoints目录下保存临时checkpoint
        if not os.path.exists("../checkpoints/train_info.txt"):
            with open("../checkpoints/info.txt", 'w') as f:
                content = "{}\n{}\n{}\n{}y\n{}\n".format(dataset_table, classes_table, device_table, train_table,
                                                         optimizer)
                f.write(content)
        torch.save(model, "../checkpoints/{}.pth".format(epoch))
    et = datetime.now()
    # 训练完，记得把model和优化器也加入到日志中（训练完加入以防训练前对model或者优化器产生影响）
    file_logger.info("optimizer:\n{}\nmodel:\n{}\n".format(optimizer, model))
    # ----------------------------------------------------------------------------------------------------------------------
    # 完成训练后的断电续训的临时文件的删除、日志保存(程序结束后自动保存)以及绘图等后续工作
    # 删除临时checkpoints文件以及临时信息文件
    cmd = input("是否删除临时文件和临时信息文件？[y/n]")
    if cmd == 'y':
        shutil.rmtree("../checkpoints")
    # indicators和loss的df记录文件生成
    df = df_generator(args.e, [losses, valid_losses], os.path.join(log_dir, 'indicators.csv'))
    # 权重生成(onnx/pth + bestf1/last)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model, os.path.join(checkpoint_dir, 'last.pth'))  # 最后一次的checkpoint
    try:
        torch2onnx(os.path.join(checkpoint_dir, 'last.pth'), os.path.join(checkpoint_dir, 'last.onnx'), shapeuse)
        file_logger.info(
            "Last model transforms successfully and path is {}.".format(os.path.join(checkpoint_dir, 'last.onnx')))
    except:
        file_logger.warning("Last model transforms failed.")

    torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'best_f1.pth'))  # 最好的f1的checkpoint
    try:
        torch2onnx(os.path.join(checkpoint_dir, 'best_f1.pth'), os.path.join(checkpoint_dir, 'best_f1.onnx'), shapeuse)
        file_logger.info(
            "Best model transforms successfully and path is {}.".format(os.path.join(checkpoint_dir, 'best_f1.onnx')))
    except:
        file_logger.warning("Best model transforms failed.")
    # 绘图（当然也可以选择使用提供的函数在训练后绘制，一些参数可以在宝库函数中自行调整）
    # 1.绘制loss和indicators变化曲线
    log_plot(df, log_dir)
    # 2.绘制数据分布图
    dataset_distribution(args.csvp, args.clsp, log_dir)
    # 3.用最好的f1的checkpoints绘制中间特征图(测试图片可以自己选)
    generate_feature('../demo.jpg', args.rs, os.path.join(checkpoint_dir, 'best_f1.pth'),
                     os.path.join(log_dir, 'feature_maps'))
