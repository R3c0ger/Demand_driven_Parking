# 需求驱动的自主代客泊车
[English version](./README.md)
## 安装
依赖库:

- [python 3.8](https://www.python.org/downloads/release/python-3818/) <3
- [numpy](https://numpy.org/install/) <3
- `pip install pandas` 用作数据处理 <3
- `pip install stable-baselines3` 用于强化学习的demo <3
- `pip install gym` 用于强化学习环境搭建 <3
- `pip install transformers` 用于指令编码 <3
- `pip install opencv-python` 用于处理图像 <3
- `pip install ray` 用于分布式计算 <3
- `pip install dm_tree, typer, scipy` <3
- `pip install h5py` 用于处理 HDF5 文件 <3

[//]: # (- `pip install fastapi` for building APIs <3)

[//]: # (- `pip install ray` for parallel and distributed computing <3)

[//]: # (- `pip install requests` for making HTTP requests <3)

[//]: # (- `pip install gradio` for interactive web UIgits <3)

[//]: # (- `pip install uvicorn` for ASGI server <3)

## 数据集
本项目需要特定的数据集才能正常运行。请从以下链接下载数据集，并保存到本地文件夹：
[数据集下载地址](https://doi.org/10.57760/sciencedb.12908)

确保已下载整个数据集，并将数据文件放到项目说明中指定的正确文件夹中。

## 快速启动
请按照以下步骤，使用提供的 Python 脚本快速开始使用强化学习环境构建与测试：
1. 确保已按照 [数据集](#数据集) 中的说明下载并正确设置了数据集。
2. 将版本库克隆到本地计算机上。 
3. 安装任何必要的依赖项。

### 深度学习样例
运行脚本进入强化学习环境，使用完美泊车代理获取深度学习代理的输入与对应的完美动作标签。
```
$ python training_data_with_deep_learning.py
```
### 强化学习样例
运行脚本进入强化学习环境，快速开始训练简单的DQN代理来完成自主代客泊车任务。
```
$ python RL_demo.py
```
### 测试样例
运行脚本测试代理的运行结果，这里使用了随机代理，可任意更换代理，运行结果将保存为json和zip文件。
```
$ python test_demo.py
```
## 许可
本项目采用 MIT 许可。有关详细信息，请参阅本软件源附带的 [许可](LICENSE) 文件。