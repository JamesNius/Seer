import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import FileDatasetsIter
from model import Brain, OracleNet
from config import config

version = config['control']['version']

# 1) 数据管线: 走真实的 FileDatasetsIter + DataLoader, 只取一个 batch
ds = FileDatasetsIter(
    version = version,
    file_list = [r'E:\天凤训练数据\2021\01\2021010100gm-00a9-0000-0af99d68.mjson'],
    pts = config['env']['pts'],
    file_batch_size = 1,
    oracle = True,
)
loader = DataLoader(ds, batch_size=64, num_workers=0)
obs, actions, masks, s2d, kr, pr, o_cnt, o_bin = next(iter(loader))
print('obs  ', obs.shape, obs.dtype)       # 期望 (64, 1012, 34) float32/uint8
print('o_cnt', o_cnt.shape, o_cnt.dtype)   # 期望 (64, 3, 35) int64
print('o_bin', o_bin.shape, o_bin.dtype)   # 期望 (64, 3, 38) float32

# 2) 模型前向 + oracle loss 组装 (复制 train_batch 里的算法)
mortal = Brain(version=version, **config['resnet'])
oracle_net = OracleNet(input_size=512 if version == 1 else 1024)
ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

phi = mortal(obs.float())
if version == 1:
    phi = phi[0]
o_cnt_logit, o_aka_logit, o_shan_logit, o_wait_logit, o_furi_logit = oracle_net(phi)
print('头输出', [tuple(o.shape) for o in (o_cnt_logit, o_aka_logit, o_shan_logit, o_wait_logit, o_furi_logit)])
# 期望 (64,3,34,5) (64,3,3) (64,3,7) (64,3,34) (64,3)

loss = (ce(o_cnt_logit.reshape(-1, 5), o_cnt[..., :34].reshape(-1))
        + ce(o_shan_logit.reshape(-1, 7), o_cnt[..., 34].reshape(-1))
        + 0.5 * (bce(o_wait_logit, o_bin[..., :34])
                 + bce(o_aka_logit, o_bin[..., 34:37])
                 + bce(o_furi_logit, o_bin[..., 37])))
loss.backward()
print(f'oracle loss = {loss.item():.4f} (期望 4.0~5.0), 反传 OK')
print('冒烟测试通过')