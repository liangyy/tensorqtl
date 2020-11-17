import pandas as pd
import numpy as np
import scipy.stats as stats
import time
import os
from collections import OrderedDict
import torch
import tensorqtl
from tensorqtl import genotypeio, cis, SimpleLogger

def cleanup_pvalue(df):
    pval = df[ 'pval_meta' ].values
    bhat = df[ 'beta_meta' ].values
    se = df[ 'beta_se_meta' ].values
    bhat[ np.isnan(pval) ] = df[ 'beta_trc' ].values[ np.isnan(pval) ]
    se[ np.isnan(pval) ] = df[ 'beta_se_trc' ].values[ np.isnan(pval) ]
    pval[ np.isnan(pval) ] = df[ 'pval_trc' ].values[ np.isnan(pval) ]
    bhat[ np.isnan(pval) ] = df[ 'beta_asc' ].values[ np.isnan(pval) ]
    se[ np.isnan(pval) ] = df[ 'beta_se_asc' ].values[ np.isnan(pval) ]
    pval[ np.isnan(pval) ] = df[ 'pval_asc' ].values[ np.isnan(pval) ]
    
    df['pval'] = pval
    df['bhat'] = bhat
    df['bhat_se'] = bhat
    return df

class InputGeneratorMix(object):
    """
    Input generator for cis-mapping with mixQTL model

    Inputs:
      genotype_df:      genotype DataFrame (genotypes x samples)
      variant_df:       DataFrame mapping variant_id (index) to chrom, pos
      phenotype_df:     phenotype DataFrame (phenotypes x samples)
      phenotype_pos_df: DataFrame defining position of each phenotype, with columns 'chr' and 'tss'
      window:           cis-window (selects variants within +- cis-window from TSS)

    Generates: phenotype array, genotype array (2D), cis-window indices, phenotype ID
    """
    def __init__(self, hap1_df, hap2_df, variant_df, log_counts_imp_df, counts_df, ref_df, alt_df, phenotype_pos_df, window=1000000):
        assert (hap1_df.index==variant_df.index).all()
        assert np.all(counts_df.index==log_counts_imp_df.index)
        assert np.all(counts_df.index==ref_df.index)
        self.hap1_df = hap1_df
        self.hap2_df = hap2_df
        self.variant_df = variant_df.copy()
        self.variant_df['index'] = np.arange(variant_df.shape[0])
        self.n_samples = counts_df.shape[1]
        # self.phenotype_df = phenotype_df
        self.log_counts_imp_df = log_counts_imp_df
        self.counts_df = counts_df
        self.ref_df = ref_df
        self.alt_df = alt_df
        self.phenotype_pos_df = phenotype_pos_df
        # check for constant phenotypes and drop
        # m = np.all(phenotype_df.values == phenotype_df.values[:,[0]], 1)
        # if m.any():
        #     print('    ** dropping {} constant phenotypes'.format(np.sum(m)))
        #     self.phenotype_df = self.phenotype_df.loc[~m]
        #     self.phenotype_pos_df = self.phenotype_pos_df.loc[~m]
        self.group_s = None
        self.window = window

        self.n_phenotypes =  phenotype_pos_df.shape[0]
        self.phenotype_tss = phenotype_pos_df['tss'].to_dict()
        self.phenotype_chr = phenotype_pos_df['chr'].to_dict()
        self.chrs = phenotype_pos_df['chr'].unique()
        self.chr_variant_dfs = {c:g[['pos', 'index']] for c,g in self.variant_df.groupby('chrom')}

        # check phenotypes & calculate genotype ranges
        # get genotype indexes corresponding to cis-window of each phenotype
        valid_ix = []
        self.cis_ranges = {}
        for k,phenotype_id in enumerate(phenotype_pos_df.index,1):
            if np.mod(k, 1000) == 0:
                print('\r  * checking phenotypes: {}/{}'.format(k, phenotype_pos_df.shape[0]), end='')

            tss = self.phenotype_tss[phenotype_id]
            chrom = self.phenotype_chr[phenotype_id]
            # r = self.chr_variant_dfs[chrom]['index'].values[
            #     (self.chr_variant_dfs[chrom]['pos'].values >= tss - self.window) &
            #     (self.chr_variant_dfs[chrom]['pos'].values <= tss + self.window)
            # ]
            # r = [r[0],r[-1]]

            m = len(self.chr_variant_dfs[chrom]['pos'].values)
            lb = np.searchsorted(self.chr_variant_dfs[chrom]['pos'].values, tss - self.window)
            ub = np.searchsorted(self.chr_variant_dfs[chrom]['pos'].values, tss + self.window, side='right')
            if lb != ub:
                r = self.chr_variant_dfs[chrom]['index'].values[[lb, ub - 1]]
            else:
                r = []

            if len(r) > 0:
                valid_ix.append(phenotype_id)
                self.cis_ranges[phenotype_id] = r

        print('\r  * checking phenotypes: {}/{}'.format(k, phenotype_pos_df.shape[0]))
        if len(valid_ix)!=phenotype_pos_df.shape[0]:
            print('    ** dropping {} phenotypes without variants in cis-window'.format(
                  phenotype_pos_df.shape[0]-len(valid_ix)))
            self.log_counts_imp_df = self.log_counts_imp_df.loc[valid_ix]
            self.counts_df = self.counts_df.loc[valid_ix]
            self.ref_df = self.ref_df.loc[valid_ix]
            self.alt_df = self.alt_df.loc[valid_ix]
            self.phenotype_pos_df = self.phenotype_pos_df.loc[valid_ix]
            self.n_phenotypes = self.phenotype_pos_df.shape[0]
            self.phenotype_tss = phenotype_pos_df['tss'].to_dict()
            self.phenotype_chr = phenotype_pos_df['chr'].to_dict()
        # if group_s is not None:
        #     self.group_s = group_s.loc[self.phenotype_df.index].copy()
        #     self.n_groups = self.group_s.unique().shape[0]


    @genotypeio.background(max_prefetch=6)
    def generate_data(self, chrom=None, verbose=False):
        """
        Generate batches from genotype data

        Returns: phenotype array, genotype matrix, genotype index, phenotype ID(s), [group ID]
        """
        if chrom is None:
            phenotype_ids = self.phenotype_pos_df.index
            chr_offset = 0
        else:
            phenotype_ids = self.phenotype_pos_df[self.phenotype_pos_df['chr']==chrom].index
            if self.group_s is None:
                offset_dict = {i:j for i,j in zip(*np.unique(self.phenotype_pos_df['chr'], return_index=True))}
            else:
                offset_dict = {i:j for i,j in zip(*np.unique(self.phenotype_pos_df['chr'][self.group_s.drop_duplicates().index], return_index=True))}
            chr_offset = offset_dict[chrom]

        index_dict = {j:i for i,j in enumerate(self.phenotype_pos_df.index)}

        if self.group_s is None:
            for k,phenotype_id in enumerate(phenotype_ids, chr_offset+1):
                if verbose:
                    genotypeio.print_progress(k, self.n_phenotypes, 'phenotype')

                c0 = self.counts_df.values[index_dict[phenotype_id]]
                c = self.log_counts_imp_df.values[index_dict[phenotype_id]]
                ref = self.ref_df.values[index_dict[phenotype_id]]
                alt = self.alt_df.values[index_dict[phenotype_id]]
                r = self.cis_ranges[phenotype_id]
                yield c0, c, ref, alt, self.hap1_df.values[r[0]:r[-1]+1], self.hap2_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), phenotype_id

        else:
            raise NotImplementedError
            # gdf = self.group_s[phenotype_ids].groupby(self.group_s, sort=False)
            # for k,(group_id,g) in enumerate(gdf, chr_offset+1):
            #     if verbose:
            #         _print_progress(k, self.n_groups, 'phenotype group')
            #     # check that ranges are the same for all phenotypes within group
            #     assert np.all([self.cis_ranges[g.index[0]][0] == self.cis_ranges[i][0] and self.cis_ranges[g.index[0]][1] == self.cis_ranges[i][1] for i in g.index[1:]])
            #     group_phenotype_ids = g.index.tolist()
            #     # p = self.phenotype_df.loc[group_phenotype_ids].values
            #     p = self.phenotype_df.values[[index_dict[i] for i in group_phenotype_ids]]
            #     r = self.cis_ranges[g.index[0]]
            #     yield p, self.genotype_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), group_phenotype_ids, group_id


def linreg(X_t, y_t):
    """Solve y = Xb"""
    X2inv_t = torch.matmul(X_t.t(), X_t).inverse()
    b_t = torch.matmul(torch.matmul(X2inv_t, X_t.t()), y_t)
    r_t = torch.matmul(X_t, b_t) - y_t
    rss_t = (r_t*r_t).sum()
    dof = X_t.shape[0] - X_t.shape[1]
    b_se_t = torch.sqrt(rss_t/dof * torch.diag(X2inv_t))
    return b_t, b_se_t

def linreg_robust(X_t, y_t):
    """
    Solve y = Xb with pre-scaling on columns 
    to avoid huge condition number on XtX
    """
    x_std = X_t.std(axis = 0)
    x_mean = X_t.mean(axis = 0)
    x_std[0] = 1
    x_mean[0] = 0
    Xtilde = torch.matmul(X_t - x_mean, torch.diag(1 / x_std))
    XtX = torch.matmul(Xtilde.T, Xtilde)
    Xty = torch.matmul(Xtilde.T, y_t)
    b_t, _ = torch.solve(Xty.unsqueeze_(dim = 1), XtX)
    b_t = b_t[:, 0]
    sigma_sq_hat = torch.pow(y_t - torch.matmul(Xtilde, b_t), 2).sum() / (Xtilde.shape[0] - Xtilde.shape[1])
    if not torch.cuda.is_available():
        XtX_inv, _ = torch.solve(torch.diag(torch.ones(Xtilde.shape[1])), XtX)
    else:
        XtX_inv, _ = torch.solve(torch.diag(torch.cuda.FloatTensor(Xtilde.shape[1]).fill_(1)), XtX)
    var_b = sigma_sq_hat * XtX_inv
    b_se_t = torch.sqrt(torch.diag(var_b))
    b_t = b_t / x_std
    b_t[0] = b_t[0] - torch.sum(x_mean * b_t)
    muSigma = x_mean / x_std
    muSigma[0] = 0
    b_se_t = b_se_t / x_std
    b_se_t[0] = torch.sqrt(b_se_t[0] ** 2 + torch.matmul(torch.matmul(muSigma.T, var_b), muSigma))
    
    return b_t, b_se_t

# def linreg_solve(X_t, y_t):
#     """Solve y = Xb using torch.solve"""
#     XtX = torch.matmul(X_t.t(), X_t)
#     Xty = torch.matmul(X_t.t(), y_t)
#     b_t, _ = torch.solve(Xty.unsqueeze_(dim = 1), XtX)
#     r_t = torch.matmul(X_t, b_t) - y_t
#     rss_t = (r_t*r_t).sum()
#     dof = X_t.shape[0] - X_t.shape[1]
#     X2inv_t = torch.matmul(X_t.t(), X_t).inverse()
#     b_t2 = torch.matmul(torch.matmul(X2inv_t, X_t.t()), y_t)
#     b_se_t = torch.sqrt(rss_t/dof * torch.diag(X2inv_t))
#     return b_t, b_se_t, b_t2
    

## added by Yanyu Liang
def regress_out(cov, y):
    b_t, b_se_t = linreg_robust(cov, y)
    y_ = y - torch.einsum('ij,j->i', cov, b_t)
    return y_
def _inner(A, B, M):
    S = torch.mul(A, B)
    return torch.einsum('ij,ik->jk', S, M)
def algo1_matrixLS(Y, X, M, numpy = True):
    # Y = torch.Tensor(Y)
    # X = torch.Tensor(X)
    # M = torch.Tensor(M)
    if not torch.cuda.is_available():
        U = torch.ones(X.shape)
    else:
        U = torch.cuda.FloatTensor(X.shape).fill_(1)
    n = torch.einsum('ij->j', M)
    Y = torch.mul(Y, M)
    T1 = torch.einsum('ij,ik->jk', X, Y)
    T2 = torch.einsum('ij,ik->jk', U, Y)
    S11 = _inner(X, X, M)
    S22 = _inner(U, U, M)
    S12 = _inner(X, U, M)
    delta = torch.abs(torch.mul(S11, S22) - torch.mul(S12, S12))
    Bhat = torch.div(torch.mul(S22, T1) - torch.mul(S12, T2), delta)
    Ahat = torch.div(torch.mul(S11, T2) - torch.mul(S12, T1), delta)
    Ysq = torch.einsum('ik,ik->k', Y, Y)
    Rsq = Ysq - 2 * torch.mul(Bhat, T1) - 2 * torch.mul(Ahat, T2) + 2 * torch.mul(torch.mul(Bhat, Ahat), S12) + torch.mul(torch.mul(Bhat, Bhat), S11) + torch.mul(torch.mul(Ahat, Ahat), S22)
    sigma = torch.sqrt(torch.div(Rsq, n - 2))
    se_B = sigma * torch.sqrt(torch.div(S22, delta))
    se_A = sigma * torch.sqrt(torch.div(S11, delta))
    if numpy is True:
        return Ahat.numpy(), Bhat.numpy(), se_A.numpy(), se_B.numpy()
    else:
        return Ahat, Bhat, se_A, se_B
def wrapper_nominal_algo1_matrixLS(Y, X, M, numpy = False):
    o = algo1_matrixLS(Y, X, M, numpy = numpy)
    slope_t = o[1]
    slope_se_t = o[3]
    tstat_t = slope_t / slope_se_t
    
    # calculate MAF
    genotypes_t = X.T * 2
    n2 = 2 * genotypes_t.shape[1]
    af_t = genotypes_t.sum(1) / n2
    ix_t = af_t <= 0.5
    maf_t = torch.where(ix_t, af_t, 1 - af_t)
    # calculate MA samples and counts
    m = genotypes_t > 0.5
    a = m.sum(1).int()
    b = (genotypes_t < 1.5).sum(1).int()
    ma_samples_t = torch.where(ix_t, a, b)
    a = (genotypes_t * m.float()).sum(1).int()
    ma_count_t = torch.where(ix_t, a, n2-a)
    
    return tstat_t.flatten(), slope_t.flatten(), slope_se_t.flatten(), maf_t, ma_samples_t, ma_count_t
def algo2_matrixLS(Y, X, M, W = None, numpy = True):
    if W is None:
        W = torch.ones(Y.shape)
    # Y = torch.Tensor(Y)
    # X = torch.Tensor(X)
    # M = torch.Tensor(M)
    # W = torch.Tensor(W)
    n = torch.einsum('ij->j', M)
    W = torch.mul(W, M)
    YsqW = torch.mul(Y, torch.sqrt(W))
    Y = torch.mul(Y, W)
    T = torch.einsum('ij,ik->jk', X, Y)
    S = torch.mul(X, X)
    S = torch.einsum('ij,ik->jk', S, W)
    Bhat = torch.div(T, S)
    Ysq = torch.einsum('ij,ij->j', YsqW, YsqW)
    Rsq = Ysq - 2 * torch.mul(Bhat, T) + torch.mul(torch.mul(Bhat, Bhat), S)
    sigma_hat = torch.sqrt(torch.div(Rsq, n - 1))
    se = torch.div(sigma_hat, torch.sqrt(S))
    if numpy is True:
        return Bhat.numpy(), se.numpy()
    else:
        return Bhat, se
def wrapper_nominal_algo2_matrixLS(Y, X, M, W = None, numpy = False):
    o = algo2_matrixLS(Y, X, M, W = W, numpy = numpy)
    slope_t = o[0]
    slope_se_t = o[1]
    tstat_t = slope_t / slope_se_t
    
    return tstat_t.flatten(), slope_t.flatten(), slope_se_t.flatten()
## END

## modified by Yanyu Liang
def trc_calc(genotypes_t, log_counts_t, raw_counts_t, covariates0_t,
             count_threshold=100, select_covariates=True):
    """
    Inputs:
      genotypes_t:  genotype dosages (variants x samples)
      log_counts_t: log(counts/(2*libsize)) --> TODO: use better normalization. CPM/TMM vs size factors?
      raw_counts_t: raw RNA-seq counts
      covariates0_t: covariates including genotype PCs, PEER factors,
                     ***with intercept in first column***
    """
    mask_t = raw_counts_t >= count_threshold
    mask_cov = raw_counts_t != 0
    
    if select_covariates:
        b_t, b_se_t = linreg_robust(covariates0_t[mask_cov, :], log_counts_t[mask_cov])
        tstat_t = b_t / b_se_t
        m = tstat_t.abs() > 2
        m[0] = True
        covariates_t = covariates0_t[:, m]
    else:
        covariates_t = covariates0_t

    M = torch.unsqueeze(mask_t, 1).float()
    M_cov = torch.unsqueeze(mask_cov, 1).float()
    Y = log_counts_t.reshape(1,-1).T
    Y[M_cov == False] = 0
    if covariates_t.shape[1] != 0:
        Y[M_cov == True] = regress_out(covariates_t[mask_cov, :], log_counts_t[mask_cov])
    res = wrapper_nominal_algo1_matrixLS(Y, genotypes_t.T / 2, M, numpy = False)
    dof = M.sum() - 2
    return res, int(mask_t.sum()), dof 
def asc_calc(hap1_t, hap2_t, ref_t, alt_t, ase_threshold=50, ase_max=1000, weight_cap=100):
    """
    Inputs:
      hap1_t: genotypes for haplotype 1 (variants x samples)
      hap2_t: genotypes for haplotype 2 (variants x samples)
      ref_t: ASE counts for REF allele
      alt_t: ASE counts for ALT allele
    """
    mask_t = ((ref_t >= ase_threshold) &
              (alt_t >= ase_threshold) &
              (ref_t <= ase_max) &
              (alt_t <= ase_max))
    M = torch.unsqueeze(mask_t, 1).float()

    X_t = hap1_t - hap2_t
    y_t = torch.log(ref_t / alt_t)
    Y = y_t.reshape(1,-1).T
    Y[M == False] = 0
    
    weights_t = 1 / (1/ref_t + 1/alt_t)
    
    if M.sum() > 0:
        weight_cutoff = torch.min(weights_t[mask_t]) * np.minimum(weight_cap, np.floor(X_t[:,mask_t].shape[1] / 10))
        weights_t[weights_t > weight_cutoff] = weight_cutoff
        
        W = weights_t.reshape(1,-1).T
        W[M == False] = 0
        res = wrapper_nominal_algo2_matrixLS(Y, X_t.T, M, W = W)
        dof = M.sum() - 1
        return res, M.sum(), dof
    else:
        return None, 0, np.NaN
# END

## The original code of the modified
# def trc_calc(genotypes_t, log_counts_t, raw_counts_t, covariates0_t,
#              count_threshold=100, select_covariates=True):
#     """
#     Inputs:
#       genotypes_t:  genotype dosages (variants x samples)
#       log_counts_t: log(counts/(2*libsize)) --> TODO: use better normalization. CPM/TMM vs size factors?
#       raw_counts_t: raw RNA-seq counts
#       covariates0_t: covariates including genotype PCs, PEER factors,
#                      ***with intercept in first column***
#     """
#     if select_covariates:
#         b_t, b_se_t = linreg(covariates0_t, log_counts_t)
#         tstat_t = b_t / b_se_t
#         m = tstat_t.abs() > 2
#         m[0] = False
#         covariates_t = covariates0_t[:, m]
#     else:
#         covariates_t = covariates0_t[:, 1:]
#     # print('  * retained {}/{} covariates'.format(covariates_t.shape[1], covariates0_t.shape[1]-1))
# 
#     mask_t = raw_counts_t >= count_threshold
#     residualizer = tensorqtl.Residualizer(covariates_t[mask_t])
# 
#     res = cis.calculate_cis_nominal(genotypes_t[:, mask_t] / 2, log_counts_t[mask_t].reshape(1,-1), residualizer)
#     # [tstat, beta, beta_se, maf, ma_samples, ma_count], samples
#     return res, int(mask_t.sum()), residualizer.dof
# 
# 
# def asc_calc(hap1_t, hap2_t, ref_t, alt_t, ase_threshold=50, ase_max=1000, weight_cap=100):
#     """
#     Inputs:
#       hap1_t: genotypes for haplotype 1 (variants x samples)
#       hap2_t: genotypes for haplotype 2 (variants x samples)
#       ref_t: ASE counts for REF allele
#       alt_t: ASE counts for ALT allele
#     """
#     mask_t = ((ref_t >= ase_threshold) &
#               (alt_t >= ase_threshold) &
#               (ref_t <= ase_max) &
#               (alt_t <= ase_max))
# 
#     X_t = hap1_t - hap2_t
#     X_t = X_t[:, mask_t]
#     y_t = torch.log(ref_t[mask_t] / alt_t[mask_t])
# 
#     # weighted least squares: transform X and y
#     weights_t = 1 / (1/ref_t[mask_t] + 1/alt_t[mask_t])
#     if weights_t.shape[0] > 0:
#         weight_cutoff = torch.min(weights_t) * np.minimum(weight_cap, np.floor(X_t.shape[1] / 10))
#         weights_t[weights_t > weight_cutoff] = weight_cutoff
#         W_t = torch.diag(torch.sqrt(weights_t))
#         yc_t = torch.matmul(W_t, y_t)
#         # Xc_t = torch.matmul(W_t, X_t.t()).t()
#         Xc_t = torch.matmul(X_t, W_t)
# 
#         gstd_t = Xc_t.std(1)
#         pstd_t = yc_t.std()
#         # normalize (no centering)
#         yc_res_t = yc_t / torch.sqrt((yc_t**2).sum())
#         Xc_res_t = Xc_t / torch.sqrt((Xc_t**2).sum(1, keepdim=True))
# 
#         # correlation
#         r_nominal_t = torch.matmul(Xc_res_t, yc_res_t)
#         r2_nominal_t = r_nominal_t**2
#         beta_t = r_nominal_t * pstd_t / gstd_t
# 
#         dof = Xc_t.shape[1] - 1
#         tstat_t = r_nominal_t * torch.sqrt(dof/(1 - r2_nominal_t))
#         beta_se_t = beta_t / tstat_t
#         res = [tstat_t, beta_t, beta_se_t]
#         return res, yc_t.shape[0], dof
#     else:
#         return None, 0, np.NaN
## END

# def get_maf(genotypes_t):
#     m = genotypes_t == -1
#     a = genotypes_t.sum(1)
#     b = m.sum(1).float()
#     mu = (a + b) / (genotypes_t.shape[1] - b)
#     return mu

def impute_hap(haplotype1_t, haplotype2_t):
    """Impute missing haplotypes to maf"""
    '''
    R code:
    is_na = is.na(geno1) | is.na(geno2)
    geno1 = impute_geno(geno1)
    geno2 = impute_geno(geno2)
    geno1[is_na] = (geno1[is_na] + geno2[is_na]) / 2
    geno2[is_na] = geno1[is_na]
    '''
    m1 = haplotype1_t == -1
    m2 = haplotype2_t == -1
    tensorqtl.impute_mean(haplotype1_t)
    tensorqtl.impute_mean(haplotype2_t)
    ix = m1 | m2
    haplotype1_t[ix] = (haplotype1_t[ix] + haplotype2_t[ix]) / 2
    haplotype2_t[ix] = haplotype1_t[ix]

def map_nominal(hap1_df, hap2_df, variant_df, log_counts_imp_df, counts_df, ref_df, alt_df,
                phenotype_pos_df, covariates_df, prefix,
                window=1000000, output_dir='.', write_stats=True, logger=None, verbose=True,
                count_threshold=100, ase_threshold=50, ase_max=1000, weight_cap=100):
    """
    cis-QTL mapping: mixQTL model, nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.mixQTL.<chr>.parquet
    """
    assert np.all(counts_df.columns==covariates_df.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write('  * {} samples'.format(counts_df.shape[1]))
    logger.write('  * {} phenotypes'.format(counts_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(variant_df.shape[0]))

    genotype_ix = np.array([hap1_df.columns.tolist().index(i) for i in counts_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # add intercept to covariates
    covariates0_t = torch.tensor(np.c_[np.ones(covariates_df.shape[0]), covariates_df],
                                 dtype=torch.float32).to(device)

    igm = InputGeneratorMix(hap1_df, hap2_df, variant_df, log_counts_imp_df, counts_df, 
                            ref_df, alt_df, phenotype_pos_df, window=window)
    # iterate over chromosomes
    start_time = time.time()
    k = 0
    logger.write('  * Computing associations')
    for chrom in igm.chrs:
        logger.write('    Mapping chromosome {}'.format(chrom))
        # allocate arrays
        n = 0
        for i in igm.phenotype_pos_df[igm.phenotype_pos_df['chr']==chrom].index:
            j = igm.cis_ranges[i]
            n += j[1] - j[0] + 1

        chr_res = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variant_id'] = []
        chr_res['tss_distance'] =   np.empty(n, dtype=np.int32)
        chr_res['maf_trc'] =        np.empty(n, dtype=np.float32)
        chr_res['ma_samples_trc'] = np.empty(n, dtype=np.int32)
        chr_res['ma_count_trc'] =   np.empty(n, dtype=np.int32)
        chr_res['beta_trc'] =       np.empty(n, dtype=np.float32)
        chr_res['beta_se_trc'] =    np.empty(n, dtype=np.float32)
        chr_res['tstat_trc'] =      np.empty(n, dtype=np.float32)
        chr_res['pval_trc'] =       np.empty(n, dtype=np.float64)
        chr_res['samples_trc'] =    np.empty(n, dtype=np.int32)
        chr_res['dof_trc'] =        np.empty(n, dtype=np.int32)
        chr_res['beta_asc'] =       np.empty(n, dtype=np.float32)
        chr_res['beta_se_asc'] =    np.empty(n, dtype=np.float32)
        chr_res['tstat_asc'] =      np.empty(n, dtype=np.float32)
        chr_res['pval_asc'] =       np.empty(n, dtype=np.float64)
        chr_res['samples_asc'] =    np.empty(n, dtype=np.int32)
        chr_res['dof_asc'] =        np.empty(n, dtype=np.int32)
        chr_res['beta_meta'] =      np.empty(n, dtype=np.float32)
        chr_res['beta_se_meta'] =   np.empty(n, dtype=np.float32)
        chr_res['tstat_meta'] =     np.empty(n, dtype=np.float32)
        chr_res['pval_meta'] =      np.empty(n, dtype=np.float64)
        chr_res['method_meta'] = []  # to record which method is used to get the meta analysis result

        start = 0
        for k, (raw_counts, log_counts, ref, alt, hap1, hap2, genotype_range, phenotype_id) in enumerate(igm.generate_data(chrom=chrom, verbose=verbose), k+1):

            # copy data to GPU
            hap1_t = torch.tensor(hap1, dtype=torch.float).to(device)
            hap2_t = torch.tensor(hap2, dtype=torch.float).to(device)
            # subset samples
            hap1_t = hap1_t[:,genotype_ix_t]
            hap2_t = hap2_t[:,genotype_ix_t]

            ref_t = torch.tensor(ref, dtype=torch.float).to(device)
            alt_t = torch.tensor(alt, dtype=torch.float).to(device)
            raw_counts_t = torch.tensor(raw_counts, dtype=torch.float).to(device)
            log_counts_t = torch.tensor(log_counts, dtype=torch.float).to(device)

            genotypes_t = hap1_t + hap2_t
            genotypes_t[genotypes_t==-2] = -1
            tensorqtl.impute_mean(genotypes_t)
            # tensorqtl.impute_mean(hap1_t)
            # tensorqtl.impute_mean(hap2_t)
            impute_hap(hap1_t, hap2_t)

            variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
            tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igm.phenotype_tss[phenotype_id])
            
            res_trc, samples_trc, dof_trc = trc_calc(genotypes_t, log_counts_t, raw_counts_t,
                                                     covariates0_t, count_threshold=count_threshold, select_covariates=True)
            # res = [tstat, beta, beta_se, maf, ma_samples, ma_count]

            res_asc, samples_asc, dof_asc = asc_calc(hap1_t, hap2_t, ref_t, alt_t, ase_threshold=ase_threshold, ase_max=ase_max, weight_cap=weight_cap)
            # res = [tstat_t, beta_t, beta_se_t]

            n = len(variant_ids)
            [tstat_trc, beta_trc, beta_se_trc, maf_trc, ma_samples_trc, ma_count_trc] = res_trc

            chr_res['phenotype_id'].extend([phenotype_id]*n)
            chr_res['variant_id'].extend(variant_ids)
            chr_res['tss_distance'][start:start+n] = tss_distance
            chr_res['maf_trc'][start:start+n] = maf_trc.cpu().numpy()
            chr_res['ma_samples_trc'][start:start+n] = ma_samples_trc.cpu().numpy()
            chr_res['ma_count_trc'][start:start+n] = ma_count_trc.cpu().numpy()
            chr_res['beta_trc'][start:start+n] = beta_trc.cpu().numpy()
            chr_res['beta_se_trc'][start:start+n] = beta_se_trc.cpu().numpy()
            chr_res['tstat_trc'][start:start+n] = tstat_trc.cpu().numpy()
            chr_res['samples_trc'][start:start+n] = samples_trc
            chr_res['dof_trc'][start:start+n] = dof_trc.cpu().numpy()

            if res_asc is not None:
                [tstat_asc, beta_asc, beta_se_asc] = res_asc
                chr_res['beta_asc'][start:start+n] = beta_asc.cpu().numpy()
                chr_res['beta_se_asc'][start:start+n] = beta_se_asc.cpu().numpy()
                chr_res['tstat_asc'][start:start+n] = tstat_asc.cpu().numpy()
                chr_res['samples_asc'][start:start+n] = samples_asc.cpu().numpy()
                chr_res['dof_asc'][start:start+n] = dof_asc.cpu().numpy()
            # meta-analysis
            if res_asc is not None and samples_asc >= 15 and samples_trc >= 15:
                chr_res['method_meta'].extend(['meta']*n)
                d = 1/beta_se_trc**2 + 1/beta_se_asc**2
                beta_meta_t = (beta_asc/beta_se_asc**2 + beta_trc/beta_se_trc**2) / d
                beta_se_meta_t = 1 / torch.sqrt(d)
                tstat_meta_t = beta_meta_t / beta_se_meta_t
                chr_res['beta_meta'][start:start+n] = beta_meta_t.cpu().numpy()
                chr_res['beta_se_meta'][start:start+n] = beta_se_meta_t.cpu().numpy()
                chr_res['tstat_meta'][start:start+n] = tstat_meta_t.cpu().numpy()
            else:
                chr_res['method_meta'].extend(['trc']*n)
                chr_res['beta_meta'][start:start+n] = beta_trc.cpu().numpy()
                chr_res['beta_se_meta'][start:start+n] = beta_se_trc.cpu().numpy()
                chr_res['tstat_meta'][start:start+n] = tstat_trc.cpu().numpy()

            start += n  # update pointer

        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

        # convert to dataframe, compute p-values and write current chromosome
        if start < len(chr_res['maf_trc']):
            for x in chr_res:
                chr_res[x] = chr_res[x][:start]

        if write_stats:
            chr_res_df = pd.DataFrame(chr_res)
            # torch.distributions.StudentT.cdf is still not implemented --> use scipy
            # m = chr_res_df['pval_nominal'].notnull()
            # chr_res_df.loc[m, 'pval_nominal'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_nominal'].abs(), dof)
            chr_res_df['pval_trc'] = 2*stats.t.cdf(-chr_res_df['tstat_trc'].abs(), chr_res_df['dof_trc'])
            chr_res_df['pval_asc'] = 2*stats.t.cdf(-chr_res_df['tstat_asc'].abs(), chr_res_df['dof_asc'])
            # chr_res_df['pval_meta'] = 2*stats.norm.cdf(-chr_res_df['tstat_meta'].abs())
            tmp = 2*stats.norm.cdf(-chr_res_df['tstat_meta'].abs())
            tmp[chr_res_df['method_meta'] == 'trc'] = chr_res_df['pval_trc'][chr_res_df['method_meta'] == 'trc'].copy()
            chr_res_df['pval_meta'] = tmp 
 
            chr_res_df = cleanup_pvalue(chr_res_df)
            print('    * writing output')
            chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.mixQTL.{}.parquet'.format(prefix, chrom)))

    logger.write('done.')

