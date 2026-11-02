"""双格式模型加载：兼容老(DQN + BatchNorm) / 新(policy + GroupNorm) 两种存档。

背景：1v3 / 自对弈 / test_play 要能"新模型打旧模型"——
  新 = ACH/clone（GroupNorm Brain + CategoricalPolicy，存档含 'policy_net'）
  旧 = 1.0/Mortal（BatchNorm Brain + DQN，存档含 'current_dqn'）
两者的主干归一化层不同、头也不同，硬编码任一种都会在加载另一种时报错。
检测逻辑与 mortal_review_server.py 的双模型兼容保持一致。
"""
import torch
from model import Brain, DQN, CategoricalPolicy
from engine import MortalEngine


def load_checkpoint(path, map_location=None):
    """torch.load 包装: 优先 weights_only=True, 失败则回退 weights_only=False。

    原因: ACH 的存档 (train_ach.py) 里 shared_stats 会是 np.float64 标量,
    weights_only=True 会因为 "Unsupported global: numpy._core.multiarray.scalar"
    直接拒绝反序列化。回退只影响反序列化策略, 不改变读出的内容。
    存档都是本项目自己训练产出的, 可信。
    """
    try:
        return torch.load(path, weights_only=True, map_location=map_location)
    except Exception:
        return torch.load(path, weights_only=False, map_location=map_location)


def build_head(state, version):
    """按存档 key 判定格式并构造头。返回 (norm, head, is_policy)。"""
    is_policy = "policy_net" in state
    norm = "GN" if is_policy else "BN"
    if is_policy:
        head = CategoricalPolicy().eval()
        head.load_state_dict(state["policy_net"])
    else:
        head = DQN(version=version).eval()
        head.load_state_dict(state["current_dqn"])
    return norm, head, is_policy


def build_engine_from_state(
    state_file,
    device,
    name,
    *,
    enable_compile=False,
    enable_amp=False,
    enable_rule_based_agari_guard=False,
    stochastic_latent=False,
    explore_rate=0,
    is_oracle=False,
    review_policy_tau=None,
):
    """从任意格式的 checkpoint 构造一个 MortalEngine（自动识别 BN/DQN vs GN/policy）。"""
    state = load_checkpoint(state_file, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']

    norm, head, is_policy = build_head(state, version)
    brain = Brain(
        version=version, num_blocks=num_blocks, conv_channels=conv_channels, Norm=norm
    ).eval()
    brain.load_state_dict(state['mortal'])

    if enable_compile:
        brain.compile()
        head.compile()

    return MortalEngine(
        brain,
        head,
        is_oracle=is_oracle,
        version=version,
        device=device,
        stochastic_latent=stochastic_latent,
        enable_amp=enable_amp,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        name=name,
        explore_rate=explore_rate,
        review_policy_tau=review_policy_tau,
    ), is_policy
