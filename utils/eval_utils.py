'''
author: optas
sources:
https://github.com/optas/latent_3d_points/blob/master/src/general_utils.py
https://github.com/optas/latent_3d_points/blob/master/src/evaluation_metrics.py

We used the evaluation metrics implemented in above urls, introduced in the paper
```Learning Representations And Generative Models For 3D Point Clouds```.
'''
import torch
import numpy as np
import warnings

from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from utils.utils import chamfer, earth_mover, f_score
from utils.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DFunction
from utils.utils import earth_mover_module
from termcolor import colored

def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    '''Computes the JSD between two sets of point-clouds, as introduced in the paper ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (torch tensor S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (torch tensor S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    '''

    # convert torch tensor to numpy
    sample_pcs = sample_pcs.detach().cpu().numpy()
    ref_pcs = ref_pcs.detach().cpu().numpy()

    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)

def emd_in_batch(xyz1, xyz2):
    # xyz1, xyz2: (B, N, 3)
    dis, _ = earth_mover_module(xyz1, xyz2, 0.002, 10000) 
    all_dist_in_batch = torch.sqrt(dis).mean(1)
    return all_dist_in_batch # (B, 1)

def cd_in_batch(xyz1, xyz2):
    # xyz1, xyz2: (B, N, 3)
    dist1, dist2, _, _ = chamfer_3DFunction.apply(xyz1, xyz2)
    all_dist_in_batch = (torch.mean(torch.sqrt(dist1), 1) + torch.mean(torch.sqrt(dist2), 1)) / 2
    return all_dist_in_batch # (B, 1)

def coverage(sample_pcs, ref_pcs, batch_size, use_EMD=False, ret_dist=False):
    '''Computes the Coverage between two sets of point-clouds.
    Args:
        sample_pcs (torch tensor SxKx3): the S point-clouds, each of K points that will be matched
            and compared to a set of "reference" point-clouds.
        ref_pcs    (torch tensor RxKx3): the R point-clouds, each of K points that constitute the
            set of "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to
            make the comparisons of the sample-vs-ref point-clouds.
        use_EMD (boolean): If true, the matchings are based on the EMD.
        ret_dist (boolean): If true, it will also return the distances between each sample_pcs and
            it's matched ground-truth.
        Returns: the coverage score (int),
                 the indices of the ref_pcs that are matched with each sample_pc
                 and optionally the matched distances of the samples_pcs.
    '''
    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    n_sam, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible Point-Clouds.')

    matched_gt = []
    matched_dist = []
    for i in range(n_sam):
        best_in_all_batches = []
        loc_in_all_batches = []

        sam_pc = sample_pcs[i]

        for ref_chunk in iterate_in_chunks(ref_pcs, batch_size):
            batch_ref_chunk = ref_chunk.shape[0]
            sam_repeat = sam_pc.repeat(batch_ref_chunk, 1, 1)

            if use_EMD:
                all_dist_in_batch = emd_in_batch(sam_repeat, ref_chunk)
            else:
                all_dist_in_batch = cd_in_batch(sam_repeat, ref_chunk)

            best_in_batch = torch.min(all_dist_in_batch).item()
            location_of_best = torch.argmin(all_dist_in_batch, axis=0).item()

            best_in_all_batches.append(best_in_batch)
            loc_in_all_batches.append(location_of_best)

        best_in_all_batches = np.array(best_in_all_batches)
        b_hit = np.argmin(best_in_all_batches)    # In which batch the minimum occurred.
        matched_dist.append(np.min(best_in_all_batches))
        hit = np.array(loc_in_all_batches)[b_hit]
        matched_gt.append(batch_size * b_hit + hit)

    cov = len(np.unique(matched_gt)) / float(n_ref)

    if ret_dist:
        return cov, matched_gt, matched_dist
    else:
        return cov, matched_gt

def minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, use_EMD=False):
    '''Computes the MMD between two sets of point-clouds.
    Args:
        sample_pcs (torch tensor SxKx3): the S point-clouds, each of K points that will be matched and
            compared to a set of "reference" point-clouds.
        ref_pcs (torch tensor RxKx3): the R point-clouds, each of K points that constitute the set of
            "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to make
            the comparisons of the sample-vs-ref point-clouds.
        test_loader: data loader from test dataset. This must extract 1 samples for each iteration.
        use_EMD (boolean: If true, the matchings are based on the EMD.
    Returns:
        A tuple containing the MMD and all the matched distances of which the MMD is their mean.
    '''

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')

    matched_dists = []
    for i in range(n_ref):
        best_in_all_batches = []
        ref_pc = ref_pcs[i]

        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            batch_sample_chunk = sample_chunk.shape[0]
            ref_repeat = ref_pc.repeat(batch_sample_chunk, 1, 1)

            if use_EMD:
                all_dist_in_batch = emd_in_batch(ref_repeat, sample_chunk)
            else:
                all_dist_in_batch = cd_in_batch(ref_repeat, sample_chunk)

            best_in_batch = torch.min(all_dist_in_batch).item()
            best_in_all_batches.append(best_in_batch)
        matched_dists.append(np.min(best_in_all_batches))
    mmd = np.mean(matched_dists)
    return mmd, matched_dists

def nearest_neighbor_accuracy(sample_pcs, ref_pcs, batch_size, use_EMD=False):
    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    n_sam, n_pc_points_s, pc_dim_s = sample_pcs.shape

    n_nearest_in_sam = 0
    n_nearest_in_ref = 0

    # calculate n_nearest_in_sam
    for i in range(n_sam):
        # X
        target_sam_pc = sample_pcs[i]

        # sample_pcs except X
        sample_pcs_ith_removed = torch.cat([sample_pcs[:i], sample_pcs[i+1:]])
        # calculate best match distance btw X and sample_pcs(X removed)
        best_in_all_batches_sample_pcs = []
        for sample_chunk in iterate_in_chunks(sample_pcs_ith_removed, batch_size):
            batch_sample_chunk = sample_chunk.shape[0]
            target_sam_repeat = target_sam_pc.repeat(batch_sample_chunk, 1, 1)

            if use_EMD:
                all_dist_in_batch = emd_in_batch(target_sam_repeat, sample_chunk)
            else:
                all_dist_in_batch = cd_in_batch(target_sam_repeat, sample_chunk)

            best_in_batch = torch.min(all_dist_in_batch).item()
            best_in_all_batches_sample_pcs.append(best_in_batch)
        best_match_dist_in_sample_pcs = np.min(best_in_all_batches_sample_pcs)

        # calculate best match distance btw X and ref_pcs
        best_in_all_batches_ref_pcs = []
        for ref_chunk in iterate_in_chunks(ref_pcs, batch_size):
            batch_ref_chunk = ref_chunk.shape[0]
            target_sam_repeat = target_sam_pc.repeat(batch_ref_chunk, 1, 1)

            if use_EMD:
                all_dist_in_batch = emd_in_batch(target_sam_repeat, ref_chunk)
            else:
                all_dist_in_batch = cd_in_batch(target_sam_repeat, ref_chunk)

            best_in_batch = torch.min(all_dist_in_batch).item()
            best_in_all_batches_ref_pcs.append(best_in_batch)
        best_match_dist_in_ref_pcs = np.min(best_in_all_batches_ref_pcs)

        # if nearest pc from X is in sample_pcs(X removed)
        if best_match_dist_in_sample_pcs < best_match_dist_in_ref_pcs:
            n_nearest_in_sam += 1

    # calculate n_nearest_in_ref
    for i in range(n_ref):
        # X
        target_ref_pc = ref_pcs[i]

        # calculate best match distance btw X and sample_pcs
        best_in_all_batches_sample_pcs = []
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            batch_sample_chunk = sample_chunk.shape[0]
            target_ref_repeat = target_ref_pc.repeat(batch_sample_chunk, 1, 1)

            if use_EMD:
                all_dist_in_batch = emd_in_batch(target_ref_repeat, sample_chunk)
            else:
                all_dist_in_batch = cd_in_batch(target_ref_repeat, sample_chunk)

            best_in_batch = torch.min(all_dist_in_batch).item()
            best_in_all_batches_sample_pcs.append(best_in_batch)
        best_match_dist_in_sample_pcs = np.min(best_in_all_batches_sample_pcs)

        # ref_pcs except X
        ref_pcs_ith_removed = torch.cat([ref_pcs[:i], ref_pcs[i+1:]])
        # calculate best match distance btw X and ref_pcs(X removed)
        best_in_all_batches_ref_pcs = []
        for ref_chunk in iterate_in_chunks(ref_pcs_ith_removed, batch_size):
            batch_ref_chunk = ref_chunk.shape[0]
            target_ref_repeat = target_ref_pc.repeat(batch_ref_chunk, 1, 1)

            if use_EMD:
                all_dist_in_batch = emd_in_batch(target_ref_repeat, ref_chunk)
            else:
                all_dist_in_batch = cd_in_batch(target_ref_repeat, ref_chunk)

            best_in_batch = torch.min(all_dist_in_batch).item()
            best_in_all_batches_ref_pcs.append(best_in_batch)
        best_match_dist_in_ref_pcs = np.min(best_in_all_batches_ref_pcs)

        # if nearest pc from X is in ref_pcs(X removed)
        if best_match_dist_in_sample_pcs > best_match_dist_in_ref_pcs:
            n_nearest_in_ref += 1
    
    nna = (n_nearest_in_sam + n_nearest_in_ref) / (n_sam + n_ref)
    return nna

def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    '''Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    '''another way of computing JSD'''

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))

def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing

def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]


# for logging during test
class MetricWriter():
    def __init__(self, csv_header, metric_name):
        self.metric_name = metric_name

        if metric_name == 'cd':
            self.metric = chamfer
        elif metric_name == 'emd':
            self.metric = earth_mover
        else:
            raise NotImplementedError

        csv_header.append(self.metric_name)
        self.total_loss = 0
        self.loss_per_cat = {}
        self.latest_loss = -1

    def run(self, row, output, gt, synset_id):
        if self.metric == 'emd':
            loss = earth_mover(output, gt, is_training=False).detach().item()
        else:
            loss = self.metric(output, gt).detach().item()
        self.total_loss += loss

        if not self.loss_per_cat.get(synset_id):
            self.loss_per_cat[synset_id] = []
        self.loss_per_cat[synset_id].append(loss)

        row.append(loss)
        self.latest_loss = loss

        return loss

    def write_name(self, header):
        header.append(self.metric_name)

    def write_name_and_latest_loss(self, list):
        list.append(self.metric_name)
        list.append('%.4f' % self.latest_loss)

    def write_mean_per_cat(self, row, synset_id):
        row.append(str(np.mean(self.loss_per_cat[synset_id])))

    def write_mean_to_log(self, log, num_total_test):
        log.write('Average ' + self.metric_name + ': %.8f \n' % (self.total_loss / num_total_test))

    def print_mean(self, num_total_test):
        print(colored('Average ' + self.metric_name + ': %f' % (self.total_loss / num_total_test), 'grey', 'on_yellow'))

    def print_mean_all_cat(self):
        print(colored(self.metric_name + ' per category', 'grey', 'on_yellow'))
        for synset_id in self.loss_per_cat.keys():
            print(colored('%s %f' % (synset_id, np.mean(self.loss_per_cat[synset_id])), 'grey', 'on_yellow'))

class FScoreWriter():
    def __init__(self, csv_header, f_score_percent_list):
        self.metric_name = 'f_score'
        self.f_score_percent_list = f_score_percent_list

        self.total_loss = 0
        self.total_f_score = {}
        self.f_score_per_cat = {}
        self.latest_f_score = {}
        for percent in f_score_percent_list:
            csv_header.append('F@' + percent)
            self.total_f_score[percent] = 0
            self.f_score_per_cat[percent] = {}

    def run(self, row, output, gt, synset_id):
        for percent in self.f_score_percent_list:
            f = f_score(recon=output, gt=gt, threshold_ratio=float(percent) / 100).detach().item()
            self.total_f_score[percent] += f

            if not self.f_score_per_cat[percent].get(synset_id):
                self.f_score_per_cat[percent][synset_id] = []
            self.f_score_per_cat[percent][synset_id].append(f)

            row.append(f)
            self.latest_f_score[percent] = f

    def write_name(self, header):
        for percent in self.f_score_percent_list:
            header.append('F-score@' + percent + '%')

    def write_name_and_latest_loss(self, list):
        for percent in self.f_score_percent_list:
            list.append('F@' + percent)
            list.append('%.4f' % self.latest_f_score[percent])

    def write_mean_per_cat(self, row, synset_id):
        for percent in self.f_score_percent_list:
            row.append(str(np.mean(self.f_score_per_cat[percent][synset_id])))

    def write_mean_to_log(self, log, num_total_test):
        for percent in self.f_score_percent_list:
            log.write('Average F-score@' + percent + '%: ' + '%.8f \n' % (
                    self.total_f_score[percent] / num_total_test))

    def print_mean(self, num_total_test):
        for percent in self.f_score_percent_list:
            print(colored('Average F-score@' + percent + '%: ' + '%.8f' %
                          (self.total_f_score[percent] / num_total_test), 'grey', 'on_yellow'))

    def print_mean_all_cat(self):
        for percent in self.f_score_percent_list:
            print(colored('F-score@' + percent + '% per category', 'grey', 'on_yellow'))
            for synset_id in self.f_score_per_cat[percent].keys():
                print(colored('%s %f' % (synset_id, np.mean(self.f_score_per_cat[percent][synset_id])), 'grey', 'on_yellow'))