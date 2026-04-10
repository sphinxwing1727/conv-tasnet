# wujian@2018
"""
数据加载与切块逻辑。

职责：
1. 从 Kaldi 风格的 `mix.scp` / `spk*.scp` 读取整句语音；
2. 将整句样本在线切成固定长度的 chunk；
3. 重新聚合成训练所需的 mini-batch。

整体数据流：
`scp -> utterance -> chunk -> batch`
"""

import random
import torch as th
import numpy as np

from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat

from .audio import WaveReader
from .utils import get_logger


logger = get_logger(__name__)


def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16):
    """
    构造本项目自定义的 DataLoader。

    输入：
    - train: 是否为训练模式，影响 shuffle 和切块起点
    - data_kwargs: 数据路径与采样率配置，通常来自 `conf.py`
    - chunk_size: 每个语音块的长度，单位为采样点
    - batch_size: 最终送入模型的 chunk 数量

    输出：
    - `DataLoader` 实例，迭代时返回：
      `{"mix": Tensor[N, S], "ref": List[Tensor[N, S]]}`
    """
    dataset = Dataset(**data_kwargs)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


class Dataset(object):
    """
    逐条 utterance 读取数据。

    输入：
    - mix_scp: 混合语音 scp
    - ref_scp: 参考语音 scp 列表，例如 `[spk1.scp, spk2.scp]`

    输出：
    - `__getitem__` 返回单条样本字典：
      `{"mix": np.ndarray[S], "ref": List[np.ndarray[S]]}`
    """
    def __init__(self, mix_scp="", ref_scp=None, sample_rate=8000):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = [
            WaveReader(ref, sample_rate=sample_rate) for ref in ref_scp
        ]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        """
        读取一条完整语音样本。

        输入：
        - index: 数据集索引

        输出：
        - dict，包含一条混合语音和多条参考语音
        """
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = [reader[key] for reader in self.ref]
        return {
            "key": key,
            "mix": mix.astype(np.float32),
            "ref": [r.astype(np.float32) for r in ref]
        }


class ChunkSplitter(object):
    """
    将整句语音切成固定长度 chunk。

    输入：
    - `eg["mix"]`: `np.ndarray[S]`
    - `eg["ref"]`: `List[np.ndarray[S]]`

    输出：
    - `List[chunk]`
    - 每个 chunk 的结构与输入样本一致，但长度被裁成 `chunk_size`
    """
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        从起点 `s` 截取一个 chunk。

        输入：
        - eg: 单条完整样本
        - s: 起始采样点

        输出：
        - `{"mix": np.ndarray[chunk_size], "ref": List[np.ndarray[chunk_size]]}`
        """
        chunk = dict()
        if "key" in eg:
            chunk["key"] = eg["key"]
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = [ref[s:s + self.chunk_size] for ref in eg["ref"]]
        return chunk

    def split(self, eg):
        """
        将单条 utterance 切成一个或多个 chunk。

        规则：
        - 太短的语音直接丢弃；
        - 不足 `chunk_size` 的语音补零；
        - 更长的语音按 `least` 步长滑窗切分。

        输出：
        - `List[dict]`，列表中的每个元素都是一个训练 chunk
        """
        N = eg["mix"].size
        key = eg.get("key", "<unknown>")
        # too short, throw away
        if N < self.least:
            logger.info(
                "ChunkSplitter produced 0 chunks (too short): key=%s, length=%d, least=%d, chunk_size=%d",
                key, N, self.least, self.chunk_size)
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            if "key" in eg:
                chunk["key"] = key
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = [
                np.pad(ref, (0, P), "constant") for ref in eg["ref"]
            ]
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        if len(chunks) != 1:
            logger.info(
                "ChunkSplitter produced %d chunks: key=%s, length=%d, chunk_size=%d, least=%d, train=%s",
                len(chunks), key, N, self.chunk_size, self.least, self.train)
        return chunks


class DataLoader(object):
    """
    面向 chunk 级 PIT 训练的在线数据加载器。

    输入：
    - `Dataset` 提供的整句样本

    输出：
    - 迭代返回训练 batch：
      `{"mix": Tensor[N, S], "ref": List[Tensor[N, S]]}`
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True):
        if batch_size < 2:
            raise ValueError(
                "batch_size must be >= 2 because the utterance loader uses "
                "batch_size // 2 internally, got {}".format(batch_size))
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        在 PyTorch DataLoader 的 collate 阶段完成在线切块。

        输入：
        - batch: 若干条完整 utterance 组成的列表

        输出：
        - `List[chunk]`
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list):
        """
        将累计的 chunk 列表重新拼成 mini-batch。

        输入：
        - chunk_list: 尚未组成 batch 的 chunk 列表

        输出：
        - blist: `List[batch]`
        - remain: 无法凑满 batch 的剩余 chunk
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        """
        逐个产出最终训练 batch。

        输出：
        - `{"mix": Tensor[N, S], "ref": List[Tensor[N, S]]}`
        """
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj
