# wujian@2018
"""
训练框架与 SI-SNR/PIT 损失实现。

职责：
1. 提供通用 Trainer，负责训练、验证、学习率调度和 checkpoint；
2. 定义 `SiSnrTrainer`，实现 Conv-TasNet 训练所需的 PIT + SI-SNR loss；
3. 约定训练 batch 的输入格式，并将其送入模型和损失函数。

本文件是训练链路中最靠近“优化目标”的一层。
"""

import os
import sys
import time

from itertools import permutations
from collections import defaultdict

import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .utils import get_logger


def load_obj(obj, device):
    """
    递归地把 batch 中的 Tensor 移动到目标设备。

    输入：
    - obj: Tensor / list / dict 的嵌套结构
    - device: 目标设备

    输出：
    - 与输入同结构、但 Tensor 已搬到设备上的对象
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    """
    训练日志使用的简易计时器。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    批级损失统计器。

    输出：
    - `report()` 返回 `{"loss": float, "batches": int, "cost": float}`
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []  # batch数为索引
        self.timer = SimpleTimer()

    def add(self, loss):  # 返回每个周期批次内的批次损失均值
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches with period{:d}"
                             "(loss = {:+.2f})...".format(N, self.period, avg))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }


class Trainer(object):
    """
    通用训练器基类。

    职责：
    - 管理模型、优化器、学习率调度器和 checkpoint；
    - 约定 `compute_loss()` 由子类实现；
    - 提供 train/eval/run 三个主要训练阶段接口。
    """

    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=6):
        """
        输入：
        - nnet: 待训练模型
        - checkpoint: 模型和日志输出目录
        - gpuid: 单卡或多卡设备 id
        - optimizer_kwargs: 优化器参数字典

        输出：
        - 初始化完成的 Trainer 实例
        """
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)
        self.tensorboard_dir = os.path.join(checkpoint, "tensorboard")
        self.tb_writer = SummaryWriter(
            log_dir=self.tensorboard_dir) if SummaryWriter else None

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.train_step = 0

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.train_step = cpt.get("train_step", 0)
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if self.tb_writer:
            self.logger.info("TensorBoard logs will be written to {}".format(
                self.tensorboard_dir))
        else:
            self.logger.info("TensorBoard is unavailable; skipping curve logs")
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        """
        保存当前 epoch 的模型和优化器状态。

        输出：
        - `best.pt.tar` 或 `last.pt.tar`
        """
        cpt = {
            "epoch": self.cur_epoch,
            "train_step": self.train_step,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        """
        根据名称创建优化器，可选恢复历史状态。

        输出：
        - `torch.optim.Optimizer` 实例
        """
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        """
        由子类实现的损失函数接口。

        输入：
        - egs: 一个 batch，约定结构为
          `{"mix": Tensor[N, S], "ref": List[Tensor[N, S]]}`

        输出：
        - 标量 loss Tensor
        """
        raise NotImplementedError

    def train(self, data_loader):
        """
        执行一个训练 epoch。

        输入：
        - data_loader: 迭代返回 batch 的数据加载器

        输出：
        - `{"loss": float, "batches": int, "cost": float}`
        """
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

            loss_value = loss.item()
            reporter.add(loss_value)
            self.train_step += 1
            if self.tb_writer:
                self.tb_writer.add_scalar("loss/train_batch", loss_value,
                                          self.train_step)
                if not self.train_step % self.logging_period:
                    self.tb_writer.flush()
        return reporter.report()

    def eval(self, data_loader):
        """
        执行一个验证 epoch，不做反向传播。

        输出：
        - `{"loss": float, "batches": int, "cost": float}`
        """
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=50):
        """
        驱动完整训练流程。

        输入：
        - train_loader: 训练集数据加载器
        - dev_loader: 验证集数据加载器
        - num_epochs: 训练轮数上限

        输出：
        - 无显式返回值
        - 训练过程中持续刷新日志、学习率与 checkpoint
        """
        # avoid alloc memory from gpu0
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            start_epoch = self.cur_epoch
            # check if save is OK
            self.save_checkpoint(best=False)
            cv = self.eval(dev_loader)
            best_loss = cv["loss"]
            if self.tb_writer:
                self.tb_writer.add_scalar("loss/dev_epoch", best_loss,
                                          self.cur_epoch)
                self.tb_writer.add_scalar("lr/epoch",
                                          self.optimizer.param_groups[0]["lr"],
                                          self.cur_epoch)
                self.tb_writer.flush()
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_impr = 0
            # make sure not inf
            self.scheduler.best = best_loss
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        cur_lr, self.cur_epoch)
                tr = self.train(train_loader)
                stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                    tr["loss"], tr["cost"], tr["batches"])
                cv = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(
                    cv["loss"], cv["cost"], cv["batches"])
                if self.tb_writer:
                    self.tb_writer.add_scalar("loss/train_epoch", tr["loss"],
                                              self.cur_epoch)
                    self.tb_writer.add_scalar("loss/dev_epoch", cv["loss"],
                                              self.cur_epoch)
                    self.tb_writer.add_scalar("lr/epoch", cur_lr,
                                              self.cur_epoch)
                    self.tb_writer.flush()
                stats["scheduler"] = ""
                if cv["loss"] > best_loss:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv["loss"])
                # flush scheduler info
                sys.stdout.flush()
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            ran_epochs = self.cur_epoch - start_epoch
            if self.cur_epoch < num_epochs:
                self.logger.info(
                    "Training stopped early after {:d} epochs in this run (current epoch: {:d})."
                    .format(ran_epochs, self.cur_epoch))
            else:
                self.logger.info(
                    "Training completed after {:d} epochs in this run (current epoch: {:d})."
                    .format(ran_epochs, self.cur_epoch))
            if self.tb_writer:
                self.tb_writer.close()


class SiSnrTrainer(Trainer):
    """
    使用 SI-SNR 作为目标函数的训练器。

    该类在 `compute_loss()` 中实现了 permutation invariant training:
    会枚举说话人排列，并选择当前 batch 中每条样本的最佳匹配。
    """

    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def sisnr(self, x, s, eps=1e-8):
        """
        计算 batch 内每条样本的 SI-SNR。

        输入：
        - x: `Tensor[N, S]`，模型输出的一路分离语音
        - s: `Tensor[N, S]`，对应参考语音

        输出：
        - `Tensor[N]`，batch 内每条样本各自的 SI-SNR
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, egs):
        """
        计算 PIT + SI-SNR 损失。

        输入：
        - `egs["mix"]`: `Tensor[N, S]`
        - `egs["ref"]`: `List[Tensor[N, S]]`

        中间结果：
        - `ests`: `List[Tensor[N, S]]`，模型输出的多路分离结果
        - `sisnr_mat`: `Tensor[num_permutations, N]`

        输出：
        - 标量 loss Tensor，数值等于最佳排列下平均 SI-SNR 的相反数
        """
        # spks x n x S
        ests = th.nn.parallel.data_parallel(
            self.nnet, egs["mix"], device_ids=self.gpuid)
        # spks x n x S
        refs = egs["ref"]
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute所以在你这段代码里，如果 permute=(0,1)，那么 for s, 
            # t in enumerate(permute) 就会依次取到：s=0, t=0s=1t=1
            return sum(
                [self.sisnr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)

        # P x N
        N = egs["mix"].size(0)
        sisnr_mat = th.stack(
            [sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = th.max(sisnr_mat, dim=0)
        # si-snr
        return -th.sum(max_perutt) / N
