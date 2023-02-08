import yaml
from easydict import EasyDict as edict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os

from util.datasets import flower17
from util.utils import create_log_folder, get_network, get_optimizer, model_info
from util.trainer import train, validate


def parse_arg():
    config_path = 'config/netConfig.yaml'
    with open(config_path, "rb") as f:
        config = yaml.safe_load(f)
        config = edict(config)

    print(config)
    return config

if __name__ == "__main__":

    cfg = parse_arg() #加载配置参数

    output_dict = create_log_folder(cfg, phase='train')

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    # 创建模型
    model = get_network(cfg)


    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(cfg.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    last_epoch = cfg.TRAIN.BEGIN_EPOCH

    optimizer = get_optimizer(cfg, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_STEP,
                                                   cfg.TRAIN.LR_FACTOR, last_epoch - 1)


    if cfg.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = cfg.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT  # 微调的文件目录
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file)
        model.load_state_dict(checkpoint['state_dict'])

    elif cfg.TRAIN.RESUME.IS_RESUME:
        model_state_file = cfg.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file)
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)
    # 加载数据集
    train_dataset = flower17(cfg, is_train=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=cfg.TRAIN.SHUFFLE,
                              num_workers=cfg.WORKERS,
                              pin_memory=cfg.PIN_MEMORY)
    val_dataset = flower17(cfg, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=cfg.TEST.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY)

    best_acc = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(last_epoch, cfg.TRAIN.END_EPOCH):
        train(cfg, train_loader, train_dataset, model, criterion, optimizer, device, epoch, writer_dict, output_dict)

        acc = validate(cfg, val_loader, val_dataset, model, criterion, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("best acc is:", best_acc)
        if is_best:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    # "epoch": epoch + 1,
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                    # "best_acc": best_acc,
                }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
            )

    writer_dict['writer'].close()
