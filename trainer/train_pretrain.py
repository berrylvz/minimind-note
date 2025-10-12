import os
import sys
__package__ = "trainer"
# 将当前文件所在目录的上一级目录添加到sys.path中，这样就可以导入当前文件所在目录的上一级目录中的模块了
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

# 忽略警告
warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    # 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))是余弦退火公式
    # 它的作用是在训练过程中，动态调整学习率，使学习率在训练过程中从大到小变化，最终收敛到一个较小的值
    # 这里的 lr 是初始学习率，current_step 是当前训练 step，total_steps 是总训练 step
    # 后者从 current_step 到 total_steps 线性变化，使学习率从 lr 线性减少到 0
    # 整体的lr从1.1 lr逐渐衰减到0.1 lr
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    # 使用 nn.CrossEntropyLoss 计算损失
    # reduction='none' 表示不进行损失的平均或求和，而是返回每个样本的损失值
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # with ctx: 启用自动混合精度
        # 对数值精度不敏感的操作，如矩阵乘法，使用float16加速计算并减少显存占用
        # 对数值精度敏感的操作，如softmax，使用float32计算
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        with ctx:
            # 前向传播
            res = model(X)
            loss = loss_fct(
                # 将原始logits（size为[batch_size, seq_len, vocab_size]）
                # 转换为2D张量[batch_size*(seq_len), vocab_size]，用于计算损失
                res.logits.view(-1, res.logits.size(-1)),
                # 将目标标签Y（size为[batch_size, seq_len]）
                # 转换为1D张量[batch_size*(seq_len)]，用于计算损失
                Y.view(-1)
            ).view(Y.size()) # 将损失转换为[batch_size, seq_len]的张量，与loss_mask对齐
            # 应用损失掩码，将padding位置的损失设为0
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 加上辅助损失
            loss += res.aux_loss
            # 平均累积的梯度
            # 在进行accumulation_steps次反向传播后，才进行一次优化器更新
            # 因此这里需要将损失除以accumulation_steps，等效于使用accumulation_steps倍大批次进行一次更新
            loss = loss / args.accumulation_steps
        # 反向传播
        scaler.scale(loss).backward()
        # 当完成了args.accumulation_steps次反向传播后，才进行一次优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 将之前用scaler.scale(loss)放大的梯度恢复为正常大小
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            # 将梯度范数限制在args.grad_clip以内
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 执行优化器更新
            scaler.step(optimizer)
            # 更新缩放因子
            scaler.update()
            # 清空梯度，准备下一次迭代
            optimizer.zero_grad(set_to_none=True)
        
        # 记录损失和学习率
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    # 剩余训练时间的估计值，单位为分钟
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})
        
        # 保存模型
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    # AutoTokenizer 从预训练模型目录加载分词器, 能够自动识别模型类型并加载对应的分词器
    # from_pretrained 加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE
    # 初始化分布式训练环境
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    # 是否使用分布式数据并行训练，Distributed Data Parallel
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    # 隐藏层数量
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    # 最大序列长度
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    # 语言模型配置
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    # 保存模型的目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    # 每个迭代的tokens数量：batch_size * max_seq_len
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # wandb 运行名称
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 自动混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        # 获取当前进程在分布式训练组(process group)中的全局排名
        # 在ddp训练中，通常会启动多个进程，每个进程都有一个唯一的排名(rank)
        # rank是当前进程在分布式训练组中的唯一标识符，范围从0到world_size-1
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    # 预训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果是分布式训练，需要为数据集添加分布式采样器
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True, # 数据加载器是否将数据加载到 CUDA 固定内存中，以加速数据传输到 GPU
        drop_last=False, # 是否在最后一个批次丢弃不足一个批次的数据
        shuffle=False, # 是否在每个 epoch 开始时对数据进行洗牌
        num_workers=args.num_workers, # 数据加载器使用的子进程数量，用于并行加载数据
        sampler=train_sampler # 分布式训练时，需要为数据加载器指定分布式采样器
    )

    # 自动混合精度(Automatic Mixed Precision, AMP)训练缩放器
    # 防止半精度训练中出现梯度下溢问题
    # 在反向传播之前，将loss乘以一个大的缩放因子，以确保梯度在半精度训练中不会下溢
    # 在更新参数之前，将放大的梯度除以缩放因子，以确保参数更新的准确性
    # 整个过程对用户透明，无需额外操作
    # 仅在 float16 或 bfloat16  dtype 下启用自动混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        # 忽略 pos_cis 缓冲区，避免在分布式训练中同步该缓冲区
        # pos_cis 是位置编码缓冲区，每个进程都有一个副本，不需要同步
        # 避免在分布式训练中同步该缓冲区，以减少通信开销
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 将模型包装在 DistributedDataParallel 中，以实现分布式训练
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
