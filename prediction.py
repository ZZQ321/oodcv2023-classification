# --------------------------------------------------------
# Reference from Swin-Transformer https://github.com/microsoft/Swin-Transformer
# Reference from ATDOC https://github.com/tim-learn/ATDOC
# --------------------------------------------------------

import argparse
import datetime
import time
import math
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm

from configs.config import get_config
from data import build_loader
from models import build_model
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as dists

import pdb

try:
    from apex import amp
except ImportError:
    amp = None

def inv_lr_scheduler(optimizer, iter_num, power=0.75, gamma=0.001, lr=0.001):
    lr = lr * (1 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
    return optimizer


def set_weight_decay(model, cfg, lr_mult=1):
    features_has_decay = []
    features_no_decay = []
    classifier_has_decay = []
    classifier_no_decay = []
    params_has_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if name.startswith("my_fc"):
            print(name)
            if len(param.shape) == 1 or name.endswith(".bias"):
                classifier_no_decay.append(param)
            else:
                classifier_has_decay.append(param)
        elif name.startswith("s_dist") or name.endswith('_ratio'):
            params_has_decay.append(param)
        else:
            if len(param.shape) == 1 or name.endswith(".bias"):
                features_no_decay.append(param)
                # print(f"{name} has no weight decay")
            else:
                features_has_decay.append(param)
    if len(classifier_has_decay) > 0:
        res = [{'params': classifier_has_decay, 'lr_mult': cfg.head_lr_ratio * lr_mult},
               {'params': params_has_decay, 'lr_mult': cfg.head_lr_ratio * lr_mult},
               {'params': classifier_no_decay, 'lr_mult': cfg.head_lr_ratio * lr_mult, 'weight_decay': 0.},
               {'params': features_has_decay, 'lr_mult': lr_mult},
               {'params': features_no_decay, 'lr_mult': lr_mult, 'weight_decay': 0.}]
    else:
        res = [{'params': features_has_decay, 'lr_mult': lr_mult},
               {'params': features_no_decay, 'lr_mult': lr_mult, 'weight_decay': 0.}]
    return res


def parse_option():
    parser = argparse.ArgumentParser('PM training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str,
                        default="configs/swin_base.yaml",
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=64, help="batch size for single GPU")
    parser.add_argument('--devices', type=str, default='0', help="device IDs")
    parser.add_argument('--dataset', type=str, default='office_home',
                        choices=['office31', 'office_home', 'VisDA', 'domainnet','robin'], help='dataset used')
    parser.add_argument('--data-root-path', type=str, default='./dataset/', help='path to dataset txt files')
    parser.add_argument('--source', type=str, default='Clipart', help='source name')
    parser.add_argument('--target', type=str, default='Art', help='target name')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--results_name', type=str, default='results')
    parser.add_argument('--output', default='results', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--alpha', type=float, default=0.8, help='hyper-parameters alpha')
    parser.add_argument('--beta', type=float, default=3, help='hyper-parameters beta')
    parser.add_argument('--log', default='log/', help='log path')
    parser.add_argument('--head_lr_ratio', type=float, default=3, help='hyper-parameters head lr')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--sourceOnly', action='store_true', default=False, help='Perform source training only')
    # distributed training
    parser.add_argument("--local_rank", nargs="+", type=int, required=True,
                        help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def mixup_ce_loss_soft(preds, targets_a, targets_b, lam):
    """ mixed categorical cross-entropy loss for soft labels
    """
    mixup_loss_a = -torch.sum(targets_a * F.log_softmax(preds), dim=1)
    mixup_loss_b = -torch.sum(targets_b * F.log_softmax(preds), dim=1)
    mixup_loss = torch.sum(torch.mul(mixup_loss_a, lam)) + torch.sum(torch.mul(mixup_loss_b, (1 - lam)))
    return mixup_loss / lam.shape[0]


def main(config):
    dsets, dset_loaders = build_loader(config)
    config.defrost()
    config.TRAIN.MAX_ITER = config.TRAIN.EPOCHS * max(len(dset_loaders['source_train']),
                                                      len(dset_loaders['target_train']))
    if config.MODEL.TYPE == "swin":
        config.MODEL.NUM_FEATURES = int(config.MODEL.SWIN.EMBED_DIM * 2 ** (len(config.MODEL.SWIN.DEPTHS) - 1))
    elif config.MODEL.TYPE == "vit":
        config.MODEL.NUM_FEATURES = config.MODEL.VIT.EMBED_DIM
    elif config.MODEL.TYPE == "deit":
        config.MODEL.NUM_FEATURES = config.MODEL.DEIT.EMBED_DIM
    config.freeze()
    logger.info(f"Creating base_model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    # model_t, model_s = build_model(config, logger)
    model = build_model(config, logger)

    # model_t.cuda()
    model.cuda()
    writer = SummaryWriter(log_dir=config.log)

    logger.info("======================================")
    logger.info("source: " + config.DATA.SOURCE)
    logger.info("target: " + config.DATA.TARGET)
    logger.info("======================================")

    parameters = set_weight_decay(model, lr_mult=1, cfg=config)
    optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                            lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=config.AMP_OPT_LEVEL)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()]
                                                      ,broadcast_buffers = False, find_unused_parameters = True)  # ]

    model_without_ddp = model.module
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    mem_fea = torch.rand(len(dset_loaders["target_train"].dataset), config.MODEL.NUM_FEATURES).cuda()
    mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
    mem_cls = torch.ones(len(dset_loaders["target_train"].dataset),
                         config.MODEL.NUM_CLASSES).cuda() / config.MODEL.NUM_CLASSES     #1/7

    logger.info("==============>Start training....................")
    start_time = time.time()
    max_accuracy = 0
    acc1 = validate(dset_loaders['target_val'], model, config)


    writer.close()

def convert_to_onehot(s_label, class_num):
    s_sca_label = s_label.cpu().data.numpy()
    return np.eye(class_num)[s_sca_label]
def softplus(x):
    return  torch.log(1+torch.exp(x))
# def train_one_epoch(config, model, dset_loaders, optimizer, epoch, writer, mem_fea, mem_cls):
def train_one_epoch(config, model, dset_loaders, optimizer, epoch, writer,mem_fea,mem_cls):
    class_weight_src = torch.ones(config.MODEL.NUM_CLASSES, ).cuda()
    model.train()
    optimizer.zero_grad()

    num_steps = max(len(dset_loaders['source_train']), len(dset_loaders['target_train']))
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()

    iter_source = iter(dset_loaders['source_train'])
    iter_target = iter(dset_loaders['target_train'])

    for idx in tqdm(range(num_steps)):
        try:
            samples_source, label_source = next(iter_source)

        except:

            iter_source = iter(dset_loaders['source_train'])
            samples_source, label_source = next(iter_source)

        try:
            samples_target, _, img_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders['target_train'])
            samples_target, _, img_idx = next(iter_target)

        idx_step = epoch * num_steps + idx
        optimizer = inv_lr_scheduler(optimizer, idx_step, lr=config.TRAIN.BASE_LR)

        samples_source = (samples_source).cuda(non_blocking=True)
        label_source = (label_source).cuda(non_blocking=True)

        samples_target = (samples_target).cuda(non_blocking=True)
        total_loss = model(samples_target, label_source, source_only=config.sourceOnly, source=samples_source, mem_fea=mem_fea, \
                           mem_cls=mem_cls, class_weight_src=class_weight_src, img_idx=img_idx, )

        # eff = (epoch + 1) / config.TRAIN.EPOCHS


        
        #print("mmd_loss", mmd_loss)
        # total_loss = classifier_loss + mxiup_ratio*loss_pseudo + pseudo_ratio* mixup_loss #+ 0.1 * eff*mmd_loss


        optimizer.zero_grad()

        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
        else:
            total_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)

        optimizer.step()

        # update memory bank,
        model.eval()
        with torch.no_grad():
            t_cls, t_logits= model(samples_target, label_source,samples_target)
            feature_t = t_logits / torch.norm(t_logits, p=2, dim=1, keepdim=True)
            outputs_target = t_cls ** 2 / ((t_cls ** 2).sum(dim=0))
            del t_cls, t_logits

        model.train()
        mem_fea[img_idx] = feature_t.clone()
        mem_cls[img_idx] = outputs_target.clone()

        torch.cuda.synchronize()
        loss_meter.update(total_loss.item(), samples_target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if dist.get_rank() == 0:
            writer.add_scalar('Loss/train', total_loss.item(), epoch)
            writer.add_scalar('Parameters_distribution/alpha', softplus(model.module.s_dist_alpha), idx_step)
            writer.add_scalar('Parameters_distribution/dist_beta',  softplus(model.module.s_dist_beta), idx_step)
            writer.add_scalar('Parameters_others/super_ratio', softplus(model.module.super_ratio), idx_step)
            writer.add_scalar('Parameters_others/unsuper_ratio ', softplus(model.module.unsuper_ratio), idx_step)
            writer.add_scalar('Parameters_others/lr', optimizer.param_groups[0]['lr'], idx_step)

    if dist.get_rank() == 0:
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'lr: {lr:.7f}\t'
                    f'loss: {loss_meter.avg:.3f}\t')
        return loss_meter.avg


@torch.no_grad()
def validate(data_loader, model, config):
    catogories= ['diningtable', 'train', 'bus', 'chair' ,'aeroplane', 'sofa', 'car', 'bicycle', 'boat', 'motorbike']
    unsort_csv_file = 'unsort_results.csv'
    model.eval()
    import csv
    with open(unsort_csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # 写入表头
        writer.writerow(["imgs", "pred"])
        
        # 写入每一行数据
        iter_val = iter(data_loader)
        for idx in range(len(data_loader)):
            batch_val, target,path = next(iter_val)
            image_name = path[0].split('/')[-1]
            images = (batch_val).cuda(non_blocking=True)
            target = (target).cuda(non_blocking=True)

            # compute output
            t_logits, _ = model(images, target,images)
            _, t_cls = torch.max(t_logits[:,:10], dim=1)
            # if t_cls > 9:
                
            class_name  = catogories[t_cls]
            writer.writerow([image_name, class_name])
    sort_csv_by_image_number(unsort_csv_file, '{}.csv'.format(config.results_name))




def sort_csv_by_image_number(input_csv, output_csv):
    """
    将CSV文件按照图像名字中的数字部分由小到大排序，并将结果保存到新的CSV文件中。
    
    参数:
    input_csv (str): 输入CSV文件的路径。
    output_csv (str): 输出排序后的CSV文件的路径。
    """
    # 读取CSV文件
    import pandas as pd
    df = pd.read_csv(input_csv)

    # 提取图像名字中的数字并作为新的一列
    df['Image Number'] = df['imgs'].str.extract('(\d+)').astype(int)

    # 根据图像名字的数字列进行排序
    df.sort_values(by='Image Number', inplace=True)

    # 删除临时的数字列
    df.drop(columns=['Image Number'], inplace=True)

    # 将排序后的数据重新写入CSV文件
    df.to_csv(output_csv, index=False)

    print(f'CSV文件已按图像名字数字排序并保存为 {output_csv}')

# 使用示例


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

if __name__ == '__main__':
    args, config = parse_option()
    import torch.multiprocessing as mp
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"
        
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
        
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend='nccl',rank=rank, world_size=num_gpus)
    dist.barrier()
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
    logger.info(config.dump())
    main(config)
