import torch
import numpy as np

class RewardCalculator:
    def __init__(self, grp=None, pts=None, uniform_init=False, point_weight=0.0):
        self.device = torch.device('cpu')
        self.grp = grp.to(self.device).eval()
        self.uniform_init = uniform_init
        self.point_weight = float(point_weight)          # ← 新增

        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)

    def calc_grp(self, grp_feature):
        seq = list(map(
            lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
            range(len(grp_feature)),
        ))

        with torch.inference_mode():
            logits = self.grp(seq)
        matrix = self.grp.calc_matrix(logits)
        return matrix

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        matrix = self.calc_grp(grp_feature)

        final_ranking = torch.zeros((1, 4), device=self.device)
        final_ranking[0, rank_by_player[player_id]] = 1.
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        if self.uniform_init:
            rank_prob[0, :] = 1 / 4
        return rank_prob

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        exp_pts = rank_prob @ self.pts
        reward = exp_pts[1:] - exp_pts[:-1]
        return reward.cpu().numpy()

    def calc_delta_points(self, player_id, grp_feature, final_scores):
        seq = np.concatenate((grp_feature[:, 3 + player_id] * 1e4, [final_scores[player_id]]))
        delta_points = seq[1:] - seq[:-1]
        return delta_points

    def calc_rewards(self, player_id, grp_feature, rank_by_player, final_scores):
        rewards = self.calc_delta_pt(player_id, grp_feature, rank_by_player)
        if self.point_weight > 0:
            delta_points = self.calc_delta_points(player_id, grp_feature, final_scores)
            rewards = rewards + self.point_weight * delta_points / 1e4
        return rewards