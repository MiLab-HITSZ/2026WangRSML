import warnings

warnings.filterwarnings("ignore")
import os
import warnings
from sklearn import metrics
import  yaml


import sys
import os
sys.path.append("/home/wxp/25week/2026WangRSML")

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")

import time
import sys
import os

with open('/home/wxp/25week/2026WangRSML/efficientnet.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
    model_cfg = cfg['MODEL']
# import logging
import warnings
import numpy
import torch
import torch.nn as nn
from rsml import RSML
# from .rsml_allinone_detector import *
from datasets.dataset import DeepfakeDataset

import cv2
# from utils2 import dist_average, ACC, compute_metrics, compute_metrics2
from config0 import train_config

# from datasets.cli_utils import get_params
assert torch.cuda.is_available()

# args = get_params()

from torch.utils.data import RandomSampler, SubsetRandomSampler
import random

def main_worker_23():
    local_rank =1
    rank =3
    torch.manual_seed(49)
    np.random.seed(49)
    random.seed(49)
    # 若使用CUDA
    torch.cuda.manual_seed_all(49)  # y  tensor([1, 0, 0, 0, 1, 1, 1, 1, 1, 1], device='cuda:3')
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(local_rank)
    Config1 = train_config('ff-b-c23', batch_size=20, resize=(224, 224))
    config=Config1
    train_dataset = DeepfakeDataset(phase='train', **Config1.dataset)

    Config11 = train_config('ff-b-c23', batch_size=20, resize=(224, 224))
    test_dataset1 = DeepfakeDataset(phase='test', **Config11.dataset)

    Config6 = train_config('celebv1', batch_size=20, resize=(224, 224))
    test_dataset6 = DeepfakeDataset(phase='test', **Config6.dataset)

    Config8 = train_config('faceshifter', batch_size=20, resize=(224, 224))
    test_dataset8 = DeepfakeDataset(phase='test', **Config8.dataset)
    #

    Config2 = train_config('celebv2', batch_size=20, resize=(224, 224))
    test_dataset2 = DeepfakeDataset(phase='test', **Config2.dataset)

    Config3 = train_config('dfdc', batch_size=20, resize=(224, 224))
    test_dataset3 = DeepfakeDataset(phase='test', **Config3.dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config1.batch_size,
                                               sampler=RandomSampler(train_dataset),
                                               pin_memory=True, num_workers=Config1.workers)
    #
    test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=Config11.batch_size,
                                               # sampler= RandomSampler(test_dataset1),
                                               pin_memory=True, num_workers=Config1.workers)
    #

    test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset2),
                                               pin_memory=True, num_workers=Config1.workers)

    test_loader3 = torch.utils.data.DataLoader(test_dataset3, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset3),
                                               pin_memory=True, num_workers=Config1.workers)

    test_loader6 = torch.utils.data.DataLoader(test_dataset6, batch_size=Config1.batch_size,
                                               # sampler= RandomSampler(test_dataset2),
                                               pin_memory=True, num_workers=Config1.workers)

    test_loader8 = torch.utils.data.DataLoader(test_dataset8, batch_size=Config1.batch_size,
                                               # sampler=RandomSampler(test_dataset4),
                                               pin_memory=True, num_workers=Config1.workers)


    net = RSML(model_cfg)


    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    # from AGDA import AGDA
    # AG = AGDA(**config.AGDA_config).to(local_rank)
    AG = None
    config.learning_rate = config.learning_rate * (config.batch_size / 80)
    config.learning_rate = 0.00025
    print(f"*config.learning_rate{config.learning_rate}  ")
    # handles = register_nan_hooks_to_submat(net.sub)

    # AG = AGDA(**config.AGDA_config).to(local_rank)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, betas=config.adam_betas,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step,
                                                gamma=config.scheduler_gamma)
    torch.cuda.empty_cache()
    start_epoch = 0
    val_list = []
    train_list = []
    val_auc = 0
    best_auc = 0
    val_acc = 0
    best_acc = 0

    net, optimizer, scheduler, start_epoch, best_acc, val_real_acc, val_fake_acc = load_checkpoint(
        net, optimizer, scheduler,
        '/mnt/data/wxp/25week/2021multiple-attention-master/对比算法/MINETEST2/model_best.pth.tar'
    )

    print(f"Loaded")
    epoch = start_epoch
    print(f"**************************    Epoch {epoch} validating  in celebv1   **************************")

    val_loss_value, val_acc, val_real_acc, val_fake_acc, val_auc = run(epoch, data_loader=test_loader6, net=net,
                                                                       optimizer=optimizer, local_rank=local_rank,
                                                                       config=Config2, phase='val', AG=None)
    # print(f"**************************    Epoch {epoch} validating  in celebv2   **************************")
    #
    # val_loss_value, val_acc, val_real_acc, val_fake_acc, val_auc = run(epoch, data_loader=test_loader2, net=net,
    #                                                                    optimizer=optimizer, local_rank=local_rank,
    #                                                                    config=Config2, phase='val', AG=None)
    # print(f"**************************    Epoch {epoch} validating  in dfdc   **************************")
    # val_loss_value, val_acc, val_real_acc, val_fake_acc, val_auc = run(epoch, data_loader=test_loader3, net=net,
    #                                                                    optimizer=optimizer, local_rank=local_rank,
    #                                                                    config=Config3, phase='val', AG=None)
    # print(f"**************************    Epoch {epoch} validating  in FS   **************************")
    # val_loss_value, val_acc, val_real_acc, val_fake_acc, val_auc = run(epoch, data_loader=test_loader8, net=net,
    #                                                                    optimizer=optimizer, local_rank=local_rank,
    #                                                                    config=Config1, phase='val', AG=None)

import torch.nn.functional as F

import numpy as np
import logging
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.amp import autocast

def run(epoch, data_loader, net, optimizer, local_rank, config, phase, AG):
    if phase == 'train':
        net.train()
    else:
        net.eval()

    data_length, acc, val_loss = 0, 0, 0
    temp_total_loss, temp_total_acc = 0, 0
    data_loader_length = len(data_loader)

    # total_step = data_loader_length / getattr(args, "batch_size")
    # 仅用于test
    out_list, label_list = [], []

    for i, datas in enumerate(data_loader):
        # data_loader_train = tqdm(data_loader, desc='Train', ncols=60)

        images, labels,_ = datas
        data_length += len(images)
        X = images.to(local_rank, non_blocking=True)
        y = labels.to(local_rank, non_blocking=True)
        if phase == 'train':
            with autocast(device_type='cuda', dtype=torch.float16):
                with torch.set_grad_enabled(phase == 'train'):
                    logits,lossR= net(X ,y)

        else:
            with torch.no_grad():
                logits,lossR= net(X,y )

        batch_loss = F.cross_entropy(logits, y)+0.1*lossR
        batch_loss_val = batch_loss.detach()
        if batch_loss_val.dim() > 0:
            batch_loss_val = batch_loss_val.mean()  # 或者 .sum() 根据你的策略
        temp_total_loss += batch_loss_val.item()
        if phase == 'train':
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        label_list.extend(y)
        out_list.extend(logits)
    outs = torch.stack(out_list)
    outs = outs.reshape(-1, 2)
    ys = torch.stack(label_list)
    ys = ys.reshape(-1)
    val_acc, val_f1, val_auc, val_real_acc, val_fake_acc, real_cnt, fake_cnt = compute_metrics_all(outs, ys)

    return round(temp_total_loss / data_length, 5), round(float(val_acc), 4), round(float(val_real_acc), 4), round(float(val_fake_acc), 4), round(float(val_auc), 4)


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, val_real_acc, val_fake_acc, val_auc, save_model_path,
                    suffix, filename='checkpoint.pth.tar'):
    os.makedirs(save_model_path, exist_ok=True)
    filename_best = 'model_best.pth.tar'
    filename_best = os.path.join(save_model_path, filename_best)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict(),
        'best_acc': best_acc,
        'val_real_acc': val_real_acc,
        'val_fake_acc': val_fake_acc,
        'auc': val_auc
    }, filename_best)
    # torch.save(model.state_dict(), filename_best)
    # shutil.copyfile(filename, filename_best)
    print(f"[INFO] Saved checkpoint to {filename_best}")

    logging.info('save model {filename}'.format(filename=filename_best))
    info_path = os.path.join(save_model_path, 'best_model_info.txt')
    with open(info_path, 'w') as f:
        f.write(f'Epoch: {suffix}\n')
        f.write(f'val_auc auc: {val_auc:.4f}\n')
        f.write(f'Val Real Acc: {val_real_acc:.4f}\n')
        f.write(f'Val Fake Acc: {val_fake_acc:.4f}\n')
    return filename_best


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    print("模型加载成功（已忽略不匹配的参数）")
    # optimizer.load_state_dict(checkpoint['optim_state'])
    # scheduler.load_state_dict(checkpoint['sched_state'])

    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    val_real_acc = checkpoint['val_real_acc']
    val_fake_acc = checkpoint['val_fake_acc']

    return model, optimizer, scheduler, epoch, best_acc, val_real_acc, val_fake_acc


def distributed_train(config, world_size=0, num_gpus=0, rank_offset=0):
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
    if not world_size:
        world_size = num_gpus
    # mp.spawn(main_worker, nprocs=num_gpus, args=(world_size, rank_offset, config))
    main_worker(config)
    torch.cuda.empty_cache()

def inspect_outputs(outputs, prefix="outputs"):
    """
    自动递归查看模型输出（支持 tensor / tuple / list / dict）
    """
    if torch.is_tensor(outputs):
        print(f"{prefix}: Tensor shape={tuple(outputs.shape)} dtype={outputs.dtype}")
    elif isinstance(outputs, (list, tuple)):
        print(f"{prefix}: {type(outputs).__name__} (len={len(outputs)})")
        for i, item in enumerate(outputs):
            inspect_outputs(item, prefix=f"{prefix}[{i}]")
    elif isinstance(outputs, dict):
        print(f"{prefix}: dict (len={len(outputs)})")
        for k, v in outputs.items():
            inspect_outputs(v, prefix=f"{prefix}['{k}']")
    else:
        print(f"{prefix}: {type(outputs)} (non-tensor value)")

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics_all(outs, ys, threshold=0.5, verbose=True):
    """
    计算分类指标，支持二分类和五分类
    返回统一格式: acc, f1, auc, real_acc, fake_acc, real_cnt, fake_cnt
    对于五分类: real_acc=类别1的准确率, fake_acc=类别0的准确率
    """
    # 自动判断任务类型
    if outs.ndim == 1:
        task_type = "binary"
        probs = torch.sigmoid(outs).detach().cpu().numpy()
        preds = (probs >= threshold).astype(int)
    else:
        if outs.shape[1] == 2:
            task_type = "binary"
            probs = torch.softmax(outs, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(outs, dim=1).detach().cpu().numpy()
        elif outs.shape[1] == 5:
            task_type = "multiclass"
            probs = torch.softmax(outs, dim=1).detach().cpu().numpy()
            preds = torch.argmax(outs, dim=1).detach().cpu().numpy()
        else:
            raise ValueError(f"不支持的输出维度: {outs.shape}")

    ys = ys.detach().cpu().numpy()

    if task_type == "binary":
        # 二分类逻辑（与原函数相同）
        acc = accuracy_score(ys, preds)
        f1 = f1_score(ys, preds, zero_division=0)
        try:
            auc = roc_auc_score(ys, probs)
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(ys, preds, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()

        real_cnt = TP + FN
        real_acc = TP / real_cnt if real_cnt > 0 else 0.0
        fake_cnt = TN + FP
        fake_acc = TN / fake_cnt if fake_cnt > 0 else 0.0

        if verbose:
            print(f"[Binary] ACC={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | "
                  f"Real_ACC={real_acc:.4f} | Fake_ACC={fake_acc:.4f}")

        return acc, f1, auc, real_acc, fake_acc, real_cnt, fake_cnt

    else:
        # 五分类逻辑
        acc = accuracy_score(ys, preds)
        f1 = f1_score(ys, preds, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(ys, probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = float("nan")

        # 计算所有五个类别的准确率
        cm = confusion_matrix(ys, preds, labels=[0, 1, 2, 3, 4])
        class_accuracies = []
        class_counts = []

        for i in range(5):
            correct = cm[i, i]  # 对角线元素，正确分类的数量
            total = cm[i, :].sum()  # 该类别总样本数
            class_acc = correct / total if total > 0 else 0.0
            class_accuracies.append(class_acc)
            class_counts.append(total)

        # 为了保持返回格式兼容，仍然使用real_acc和fake_acc
        fake_acc = class_accuracies[0]  # 类别0的准确率
        real_acc = class_accuracies[1]  # 类别1的准确率
        fake_cnt = class_counts[0]  # 类别0的样本数
        real_cnt = class_counts[1]  # 类别1的样本数

        if verbose:
            print(f"[5-Class] ACC={acc:.4f} | F1_macro={f1:.4f} | AUC={auc:.4f}")
            for i in range(5):
                print(f"                Class_{i}_ACC={class_accuracies[i]:.4f} ({class_counts[i]} samples)")

        return acc, f1, auc, real_acc, fake_acc, real_cnt, fake_cnt


if __name__ == '__main__':
    from config0 import train_config
    import linecache
    import sys
    import torch
    import sys
    import codecs
    import linecache

    torch.cuda.empty_cache()
    main_worker_23()
