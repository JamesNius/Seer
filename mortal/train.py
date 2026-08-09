def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from libriichi.consts import obs_shape
    from config import config
    from model import Brain, DQN, AuxNet, OracleNet

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    oracle_weight = config['aux'].get('oracle_weight', 0.1)
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    point_weight = config['env'].get('point_weight', 0.0)
    gamma = config['env']['gamma']
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

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    if version == 1:
        aux_net = AuxNet((4,), input_size=512).to(device)
    else:
        aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)
    oracle_net = OracleNet(input_size=512 if version == 1 else 1024).to(device)
    # ── oracle 冻结（step 1,065,600 起）──────────────────────────────
    # 架构说明：OracleNet 是挂在共享 backbone（Brain.encoder 的 phi）上的线性头，
    # 与 DQN/AuxNet 共用同一 encoder。冻结后 oracle 头参数不再更新，但 backbone
    # 仍会继续训练——因此当前 checkpoint 里的 oracle_net 会随 phi 分布漂移，
    # 其输出不再可信。推理侧请加载快照 checkpoints/oracle_frozen_1065600.pt
    # （其中 oracle_net 是冻结时刻的完整快照，与当时的 backbone 对齐），
    # 而不要用训练过程中新保存的 mortal.pth 里的 oracle_net。
    for p in oracle_net.parameters():
        p.requires_grad_(False)
    if enable_compile:
        oracle_net.compile()
    logging.info(f'oracle params: {parameter_count(oracle_net):,}')
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
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
    oracle_optimizer = optim.AdamW(oracle_net.parameters(), lr=1, weight_decay=weight_decay, betas=betas, eps=eps)
    oracle_scheduler = LinearWarmUpCosineAnnealingLR(oracle_optimizer, **config['optim']['scheduler'])
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if not online or state['config']['control']['online']:
            if 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])
            if 'scheduler' in state:
                scheduler.load_state_dict(state['scheduler'])
                # load_state_dict 会把 checkpoint 里的旧超参（max_steps/final/peak 等）
                # 和旧 lambda 一起灌回来，覆盖按当前 config 构造的新曲线。
                # 恢复 config 的超参 + 装回新 lambda，再用当前 step 重算 LR。
                sched_cfg = config['optim']['scheduler']
                for k in ('init', 'peak', 'final', 'warm_up_steps', 'max_steps', 'offset', 'epoch_size'):
                    if k in sched_cfg:
                        setattr(scheduler, k, sched_cfg[k])
                scheduler.lr_lambdas = [scheduler._step_inner for _ in scheduler.optimizer.param_groups]
                # 方案 A：旧曲线 max_steps=0 表示从 step 0 起就恒在 peak，
                # 若直接 resume，cos_steps=当前step 会让 LR 跳到余弦尾部（悬崖式跳水）。
                # 把 warm_up_steps 拨到当前 step、max_steps 拨到当前 step + 剩余衰减步数，
                # 使余弦从现在重新起坡：LR 从 peak 平滑降到 final，用满剩余步数。
                cur_step = scheduler.last_epoch
                remaining = sched_cfg['max_steps'] - cur_step
                scheduler.warm_up_steps = cur_step
                scheduler.max_steps = cur_step + remaining
                scheduler.step(cur_step)
        if 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        if 'oracle_net' in state:
            oracle_net.load_state_dict(state['oracle_net'])
            if 'oracle_optimizer' in state:
                oracle_optimizer.load_state_dict(state['oracle_optimizer'])
            if 'oracle_scheduler' in state:
                oracle_scheduler.load_state_dict(state['oracle_scheduler'])
        best_perf = state['best_perf']
        steps = state['steps']


    optimizer.zero_grad(set_to_none=True)
    oracle_optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
        'oracle_loss': 0,
        'oracle_cnt_acc': 0,
        'oracle_shan_acc': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def save_state():
        state = {
            'mortal': mortal.state_dict(),
            'current_dqn': dqn.state_dict(),
            'aux_net': aux_net.state_dict(),
            'oracle_net': oracle_net.state_dict(),
            'oracle_optimizer': oracle_optimizer.state_dict(),
            'oracle_scheduler': oracle_scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            'timestamp': datetime.now().timestamp(),
            'best_perf': best_perf,
            'config': config,
            # oracle 自 step 1,065,600 起冻结：此 state_dict 里的 oracle_net
            # 未再训练，且 backbone 继续漂移，已不再对齐，勿用于推理
            'oracle_frozen': oracle_weight <= 0,
        }
        torch.save(state, state_file)
        logging.info(f'checkpoint saved at step {steps:,}')

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            oracle = True,
            file_list = file_list,
            pts = pts,
            point_weight = point_weight,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            prefetch_factor = 8,
            worker_init_fn = worker_init_fn,
        ))

        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_o_cnt = []
        remaining_o_bin = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, o_cnt, o_bin):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            o_cnt = o_cnt.to(dtype=torch.int64, device=device)
            o_bin = o_bin.to(dtype=torch.float32, device=device)
            assert masks[range(batch_size), actions].all()

            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                if version == 1:
                    mu, logsig = phi
                    phi = mu
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                dqn_loss = 0.5 * mse(q, q_target_mc)
                cql_loss = 0
                if not online:
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                next_rank_logits, = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)

                # oracle 已冻结（config oracle_weight=0）：前向跳过，不反传；
                # 为 0 时其 optimizer 的 unscale/step/scheduler 也一并跳过
                oracle_active = oracle_weight > 0
                if oracle_active:
                    o_cnt_logit, o_aka_logit, o_shan_logit, o_wait_logit, o_furi_logit = oracle_net(phi)
                    oracle_cnt_loss = ce(o_cnt_logit.reshape(-1, 5), o_cnt[..., :34].reshape(-1))
                    oracle_shan_loss = ce(o_shan_logit.reshape(-1, 7), o_cnt[..., 34].reshape(-1))
                    oracle_wait_loss = bce(o_wait_logit, o_bin[..., :34])
                    oracle_aka_loss = bce(o_aka_logit, o_bin[..., 34:37])
                    oracle_furi_loss = bce(o_furi_logit, o_bin[..., 37])
                    oracle_loss = oracle_cnt_loss + oracle_shan_loss \
                                  + 0.5 * (oracle_wait_loss + oracle_aka_loss + oracle_furi_loss)

                loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                    next_rank_loss * next_rank_weight,
                    oracle_loss * oracle_weight if oracle_active else 0.,
                ))
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                if not online:
                    stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                if oracle_weight > 0:
                    stats['oracle_loss'] += oracle_loss
                    stats['oracle_cnt_acc'] += (o_cnt_logit.argmax(-1) == o_cnt[..., :34]).float().mean()
                    stats['oracle_shan_acc'] += (o_shan_logit.argmax(-1) == o_cnt[..., 34]).float().mean()
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                    if oracle_active:
                        # oracle 冻结时 oracle_net 无梯度，不能再 unscale_/step 它
                        scaler.unscale_(oracle_optimizer)
                        clip_grad_norm_(oracle_net.parameters(), max_grad_norm)
                scaler.step(optimizer)
                if oracle_active:
                    scaler.step(oracle_optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if oracle_active:
                    oracle_optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if oracle_active:
                oracle_scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # downsample to reduce tensorboard event size
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                if not online:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                writer.add_scalar('loss/oracle_loss', stats['oracle_loss'] / save_every, steps)
                writer.add_scalar('oracle/cnt_acc', stats['oracle_cnt_acc'] / save_every, steps)
                writer.add_scalar('oracle/shan_acc', stats['oracle_shan_acc'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    'oracle_net': oracle_net.state_dict(),
                    'oracle_optimizer': oracle_optimizer.state_dict(),
                    'oracle_scheduler': oracle_scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                    # oracle 自 step 1,065,600 起冻结：此 state_dict 里的 oracle_net
                    # 未再训练，且 backbone 继续漂移，已不再对齐，勿用于推理
                    'oracle_frozen': oracle_weight <= 0,
                }
                torch.save(state, state_file)

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, dqn, device)
                    mortal.train()
                    dqn.train()

                    avg_pt = stat.avg_pt([90, 45, 0, -135]) # for display only, never used in training
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, o_cnt, o_bin in data_loader:
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_o_cnt.append(o_cnt)
                remaining_o_bin.append(o_bin)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, o_cnt, o_bin)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            o_cnt = torch.cat(remaining_o_cnt, dim=0)
            o_bin = torch.cat(remaining_o_bin, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                    o_cnt[start:end],
                    o_bin[start:end],
                )
                start = end
                end += batch_size
        pb.close()
        save_state()   # epoch 兜底: 尾部不足 save_every 的步数也落盘

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
