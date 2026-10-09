def train():
    """知识蒸馏: 把 value-based 教师(BN Brain+DQN) 的策略蒸馏到 policy-based 学生(GN Brain+CategoricalPolicy)。

    为什么需要: Mortal-Policy 的 AWR 从随机初始化开训不稳(其 README 亦如此提示)。
    先用本脚本把当前最强模型(seer.1.0.收敛版)的 Q 分布蒸馏进学生, 得到好的起点,
    产出的 checkpoint 可直接被 train_offline_phase1.py(AWR) resume。

    存盘格式与 train_offline_phase1.py 完全一致:
        mortal / policy_net / optimizer / scheduler / scaler / steps / timestamp
        / best_perf / config / shared_stats
    """
    import prelude

    import logging
    import sys
    import os
    import gc
    import json
    import random
    import math
    import time
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    import torch
    import numpy as np
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import parameter_count, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, CategoricalPolicy
    from libriichi.consts import obs_shape
    from config import config
    from multiprocessing import Manager

    # ── 与 train_offline_phase1.py 相同的配置读取 ──────────────────────
    version = config['control']['version']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    # ── 蒸馏专用参数 [distill] ────────────────────────────────────────
    teacher_state_file = config['distill']['teacher_state_file']
    tau = float(config['distill']['tau'])          # 教师 Q 的 softmax 温度: 越小分布越锐利
    max_steps = int(os.environ.get('DISTILL_MAX_STEPS', config['distill']['max_steps']))  # 总步数; 环境变量可覆盖(冒烟用)
    LOG_EVERY = int(os.environ.get('DISTILL_LOG_EVERY', 20))   # 每隔多少步打一行进度(步数/速度/ETA)
    t_start = time.time()

    # ── 学生(与 AWR 完全一致的构造) ────────────────────────────────────
    mortal = Brain(version=version, **config['resnet'], Norm="GN").to(device)
    policy_net = CategoricalPolicy().to(device)
    if enable_compile:
        mortal.compile()
        policy_net.compile()

    # ── 教师(BN Brain + DQN，冻结、eval) ──────────────────────────────
    t_state = torch.load(teacher_state_file, weights_only=True, map_location='cpu')
    t_cfg = t_state['config']
    assert version == t_cfg['control']['version'], f"version mismatch: {version} vs {t_cfg['control']['version']}"
    assert config['resnet']['conv_channels'] == t_cfg['resnet']['conv_channels'], 'conv_channels mismatch'
    assert config['resnet']['num_blocks'] == t_cfg['resnet']['num_blocks'], 'num_blocks mismatch'
    teacher = Brain(version=version, **config['resnet']).to(device)          # 注意: 默认 Norm="BN"
    teacher_dqn = DQN(version=version).to(device)
    teacher.load_state_dict(t_state['mortal'])
    teacher_dqn.load_state_dict(t_state['current_dqn'])
    teacher.eval()
    teacher_dqn.eval()
    for p in chain(teacher.parameters(), teacher_dqn.parameters()):
        p.requires_grad_(False)
    logging.info(f'teacher: {teacher_state_file} (steps={t_state.get("steps"):,}, tau={tau})')

    logging.info(f'version: {version}, obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy_net):,}')

    # ── 优化器(与 AWR 相同: Linear/Conv 的 weight 才衰减) ──────────────
    decay_params = []
    no_decay_params = []
    for model in (mortal, policy_net):
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    best_perf = {'avg_rank': 4., 'avg_pt': -135.}

    steps = 0
    state_file = config['control']['state_file']

    manager = Manager()
    shared_stats = {
        'count': manager.Value('i', 0),
        'mean': manager.Value('d', 0.0),
        'M2': manager.Value('d', 0.0),
        'lock': manager.Lock()
    }

    # ── 断点续训: state_file 已存在则接着训 ────────────────────────────
    # 用途: 先跑 30k 看 teacher-match 趋势, 不够就把 max_steps 调大、重跑本脚本,
    #       它会从这里接着练, 不用从随机初始化重来。
    if path.exists(state_file):
        _st = torch.load(state_file, weights_only=False, map_location=device)
        mortal.load_state_dict(_st['mortal'])
        policy_net.load_state_dict(_st['policy_net'])
        if 'optimizer' in _st:
            optimizer.load_state_dict(_st['optimizer'])
        if 'scheduler' in _st:
            scheduler.load_state_dict(_st['scheduler'])
        if 'scaler' in _st:
            scaler.load_state_dict(_st['scaler'])
        steps = int(_st.get('steps', 0))
        logging.info(f'[DISTILL] resumed from {state_file} @ step {steps:,} (继续训到 max_steps)')
    else:
        logging.info(f'[DISTILL] 从随机初始化开始 (state_file 不存在: {state_file})')

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    # 日志同时写文件(脚本自带, 不依赖 .bat 重定向), 控制台也实时可见
    os.makedirs('logs', exist_ok=True)
    _fh = logging.FileHandler(path.join('logs', 'distill.log'), mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(_fh)

    writer = SummaryWriter(config['control']['tensorboard_dir'])

    # ── 数据(复用 AWR 同一套 FileDatasetsIter；只用 obs/masks) ─────────
    file_index = config['dataset']['file_index']
    if path.exists(file_index):
        file_list = torch.load(file_index, weights_only=True)['file_list']
    else:
        logging.info('building file index...')
        file_list = []
        for pat in config['dataset']['globs']:
            file_list.extend(glob(pat, recursive=True))
        file_list.sort(reverse=True)
        torch.save({'file_list': file_list}, file_index)
    logging.info(f'file list size: {len(file_list):,}')

    input_keys = ['obs', 'masks']   # 蒸馏只需要 obs + masks

    def train_epoch():
        nonlocal steps

        file_data = FileDatasetsIter(
            version=version,
            file_list=file_list,
            pts=pts,
            shared_stats=shared_stats,
            file_batch_size=file_batch_size,
            reserve_ratio=reserve_ratio,
            num_epochs=num_epochs,
            enable_augmentation=enable_augmentation,
            augmented_first=augmented_first,
        )
        data_loader = iter(DataLoader(
            dataset=file_data,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn,
        ))

        idx = 0

        def train_batch(obs, masks):
            nonlocal steps, idx
            obs = obs.to(dtype=torch.float32, device=device)
            masks = masks.to(dtype=torch.bool, device=device)

            with torch.autocast(device.type, enabled=enable_amp):
                with torch.no_grad():
                    q_t = teacher_dqn(teacher(obs), masks)          # 教师 Q（mask 外为 -inf）
                    p_t = torch.softmax(q_t / tau, dim=-1)          # 软目标
                p_s = policy_net(mortal(obs), masks)                # 学生概率
                # 交叉熵 = KL(p_t || p_s) + const(教师熵)，对小概率做 clamp 防 -inf
                loss = -(p_t * p_s.clamp_min(1e-8).log()).sum(-1).mean()

            scaler.scale(loss / opt_step_every).backward()
            stats['loss'] += loss.item()
            stats['n'] += 1
            with torch.no_grad():
                # 学生 argmax 与教师 argmax 的一致率 = 最直观的"学到没有"指标
                stats['match'] += (p_s.argmax(-1) == p_t.argmax(-1)).float().mean().item()

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            # ── 进度行: 每 LOG_EVERY 步一行, 含 步数/百分比/loss/一致率/速度/预计剩余 ──
            if steps % LOG_EVERY == 0:
                n = max(stats['n'], 1)
                avg_loss = stats['loss'] / n
                avg_match = stats['match'] / n
                rate = steps / max(time.time() - t_start, 1e-9)
                eta_h = (max_steps - steps) / max(rate, 1e-9) / 3600.0
                logging.info(
                    f'[DISTILL] step {steps:,}/{max_steps:,} ({100.0 * steps / max_steps:.1f}%)'
                    f' | loss={avg_loss:.4f} | teacher-match={avg_match:.4f}'
                    f' | {rate:.2f} step/s | ETA {eta_h:.1f}h'
                )
                writer.add_scalar('loss', avg_loss, steps)
                writer.add_scalar('teacher_match', avg_match, steps)
                stats['loss'] = 0.0
                stats['match'] = 0.0
                stats['n'] = 0

            if steps % save_every == 0:
                writer.flush()
                state = {
                    'mortal': mortal.state_dict(),
                    'policy_net': policy_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                    'shared_stats': None,
                }
                torch.save(state, state_file)
                logging.info(f'[DISTILL] saved: {state_file} (steps={steps:,})')

        for batch in data_loader:
            obs, _actions, masks = batch[0], batch[1], batch[2]
            if obs.shape[0] != batch_size:
                continue
            train_batch(obs, masks)
            if steps >= max_steps:
                break

    stats = {'loss': 0, 'match': 0, 'n': 0}
    while steps < max_steps:
        train_epoch()
        gc.collect()

    # 收尾落盘
    state = {
        'mortal': mortal.state_dict(),
        'policy_net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'steps': steps,
        'timestamp': datetime.now().timestamp(),
        'best_perf': best_perf,
        'config': config,
        'shared_stats': None,
    }
    torch.save(state, state_file)
    logging.info(f'[DISTILL] done. final saved: {state_file} (steps={steps:,})')


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
