import torch
import torch.nn.functional as F
from torch.func import jvp


def rk4_step(f, z, dt):
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_flow_map(f, z, dt):
    return rk4_step(f, z, dt)


def build_cross_traj_knn_pairs(z, traj_ids, k):
    batch_size = z.shape[0]
    k = max(int(k), 0)
    if batch_size == 0 or k == 0:
        empty_index = torch.zeros((batch_size, k), dtype=torch.long, device=z.device)
        empty_mask = torch.zeros((batch_size, k), dtype=torch.bool, device=z.device)
        empty_dist = torch.zeros((batch_size, k), dtype=z.dtype, device=z.device)
        empty_kth = torch.zeros((batch_size,), dtype=z.dtype, device=z.device)
        return empty_index, empty_mask, empty_dist, empty_kth

    dist = torch.cdist(z, z)
    traj_ids = traj_ids.view(-1)
    valid = traj_ids[:, None] != traj_ids[None, :]
    valid.fill_diagonal_(False)

    dist = dist.masked_fill(~valid, float('inf'))
    max_neighbors = min(k, dist.shape[1])
    knn_dist, knn_idx = torch.topk(dist, k=max_neighbors, largest=False, dim=1)
    knn_valid = torch.isfinite(knn_dist)

    if max_neighbors < k:
        pad_shape = (batch_size, k - max_neighbors)
        knn_idx = torch.cat([knn_idx, torch.zeros(pad_shape, dtype=torch.long, device=z.device)], 1)
        knn_valid = torch.cat([knn_valid, torch.zeros(pad_shape, dtype=torch.bool, device=z.device)], 1)
        knn_dist = torch.cat([knn_dist, torch.zeros(pad_shape, dtype=z.dtype, device=z.device)], 1)

    valid_counts = knn_valid.sum(1)
    kth_idx = torch.clamp(valid_counts, min=1, max=k) - 1
    kth_dist = knn_dist.gather(1, kth_idx.unsqueeze(1)).squeeze(1)
    kth_dist = torch.where(valid_counts > 0, kth_dist, torch.zeros_like(kth_dist))
    knn_dist = torch.where(knn_valid, knn_dist, torch.zeros_like(knn_dist))

    return knn_idx, knn_valid, knn_dist, kth_dist


def rbf_pair_weights(delta0, kth_dist, eps):
    sq_dist = (delta0 * delta0).sum(-1)
    scale = kth_dist.unsqueeze(-1) ** 2 + eps
    return torch.exp(-sq_dist / scale)


def transport_secants_jvp(flow_map, anchors, deltas):
    if anchors.numel() == 0:
        empty = torch.zeros_like(anchors)
        return empty, empty
    return jvp(flow_map, (anchors,), (deltas,))


def _weighted_mean(values, weights, eps):
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    return (values * weights).sum() / weight_sum.clamp_min(eps)


def secant_transport_metrics(delta_pred, delta_true, weights, eps):
    if delta_pred.numel() == 0 or weights.numel() == 0 or torch.count_nonzero(weights) == 0:
        zero = torch.zeros((), dtype=delta_true.dtype, device=delta_true.device)
        return {'logmag_mae': zero, 'angle_err': zero, 'rel_l2': zero}

    pred_norm = torch.linalg.norm(delta_pred, dim=-1)
    true_norm = torch.linalg.norm(delta_true, dim=-1)
    logmag = torch.abs(torch.log(pred_norm + eps) - torch.log(true_norm + eps))
    angle = 1.0 - F.cosine_similarity(delta_pred, delta_true, dim=-1, eps=eps)
    rel_l2 = torch.linalg.norm(delta_pred - delta_true, dim=-1) / (true_norm + eps)

    return {
        'logmag_mae': _weighted_mean(logmag, weights, eps),
        'angle_err': _weighted_mean(angle, weights, eps),
        'rel_l2': _weighted_mean(rel_l2, weights, eps),
    }


def secant_transport_loss(delta_pred, delta_true, weights, eps, norm_weight, angle_weight):
    metrics = secant_transport_metrics(delta_pred, delta_true, weights, eps)
    return norm_weight * metrics['logmag_mae'] + angle_weight * metrics['angle_err']
