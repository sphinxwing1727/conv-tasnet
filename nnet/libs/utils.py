# wujian@2018
"""
通用工具函数。

职责：
1. 构造统一格式的日志对象；
2. 把训练配置写成 JSON；
3. 从 checkpoint 目录恢复保存的 JSON 配置。

这些函数被训练、推理和评估脚本共同使用。
"""

import os
import json
import logging


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False,
        console=None):
    """
    创建一个 Python logger。

    输入：
    - name: logger 名称；当 `file=True` 时也会作为日志文件路径
    - format_str / date_format: 日志格式
    - file: 是否写入文件；也可以直接传入日志文件路径
    - console: 是否输出到终端；默认在 file=False 时输出终端，在 file=True 时仅写文件

    输出：
    - `logging.Logger` 实例
    +
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if console is None:
        console = not file

    for handler in list(logger.handlers):
        if getattr(handler, "_codex_managed", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)

    if console:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        handler._codex_managed = True
        logger.addHandler(handler)

    if file:
        log_path = name if file is True else file
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        handler._codex_managed = True
        logger.addHandler(handler)

    return logger


def dump_json(obj, fdir, name):
    """
    将 Python 对象写入 JSON 文件。

    输入：
    - obj: 待序列化对象
    - fdir: 目标目录
    - name: 文件名

    输出：
    - 无显式返回值，结果写入 `fdir/name`
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir, name):
    """
    从目录中读取 JSON 文件。

    输入：
    - fdir: 文件所在目录
    - name: 文件名

    输出：
    - 反序列化后的 Python 对象
    """
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj
