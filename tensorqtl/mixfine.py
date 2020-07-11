'''
A python interface for mixFine in mixqtl R package: https://github.com/hakyimlab/mixqtl
'''


import pandas as pd
import numpy as np
import scipy.stats as stats
import time
import os
from collections import OrderedDict
import torch
import tensorqtl
from tensorqtl import genotypeio, cis, SimpleLogger
import mixqtl

from rpy2.robjects.packages import importr
r_mixqtl = importr('mixqtl')
r_base = importr('base')
from rpy2.robjects import numpy2ri
numpy2ri.activate()
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def finemapper(hap1_t, hap2_t, ref_t, alt_t, raw_counts_t, libsize_t, covariates0_t, 
               count_threshold=100, select_covariates=True, 
               ase_threshold=50, ase_max=1000, weight_cap=100, 
               mode='mixfine'):
    
    """
    Inputs:
      hap1_t: genotypes for haplotype 1 (variants x samples)
      hap2_t: genotypes for haplotype 2 (variants x samples)
      ref_t: ASE counts for REF allele
      alt_t: ASE counts for ALT allele
      raw_counts_t: raw RNA-seq counts
      libsize_t: library size (log_counts_t = log(raw_counts_t/(2 * libsize_t))
      covariates0_t: covariates including genotype PCs, PEER factors,
                     ***with intercept in first column***
    **If mode == 'nefine': raw_counts_t will be interpreted as inrt normalized expression.
    And no cutoff applies.
    """
    
    if mode == 'nefine':
        log_counts_t = raw_counts_t.copy()
        raw_counts_t[:] = count_threshold + 1
    elif mode == 'mixfine' or mode == 'trcfine':
        log_counts_t = torch.log(raw_counts_t / libsize_t / 2).type(torch.float)
        # breakpoint() 
    mask_t = raw_counts_t >= count_threshold
    mask_cov = raw_counts_t > 0
    
    if select_covariates:
        b_t, b_se_t = mixqtl.linreg_robust(covariates0_t[mask_cov, :], log_counts_t[mask_cov])
        tstat_t = b_t / b_se_t
        m = tstat_t.abs() > 2
        m[0] = True
        covariates_t = covariates0_t[:, m]
    else:
        covariates_t = covariates0_t

    M = torch.unsqueeze(mask_t, 1).float()
    M_cov = torch.unsqueeze(mask_cov, 1).float()
    cov_offset = log_counts_t.reshape(1,-1).T
    cov_offset[M_cov == False] = 0
    
    if covariates_t.shape[1] != 0:
        cov_offset[M_cov == True] = mixqtl.get_offset(covariates_t[mask_cov, :], log_counts_t[mask_cov])
    
    if mode == 'mixfine':
        o = {'geno1': hap1_t, 'geno2': hap2_t,
                             'y1': ref_t, 'y2': alt_t,
                             'ytotal': raw_counts_t,
                             'lib_size': libsize_t, 'cov_offset': cov_offset }
        import pickle 
        with open('test.pkl', 'wb') as f:
            pickle.dump(o, f)
        res = r_mixqtl.mixfine(geno1=hap1_t.T.numpy(), geno2=hap2_t.T.numpy(), 
                             y1=ref_t.numpy(), y2=alt_t.numpy(), 
                             ytotal=raw_counts_t.numpy(), 
                             lib_size=libsize_t.numpy(), cov_offset=cov_offset[:, 0].numpy(),
                             trc_cutoff=count_threshold, asc_cutoff=ase_threshold, 
                             asc_cap=ase_max, weight_cap=weight_cap, nobs_asc_cutoff=3)
    elif mode == 'nefine':
        res = mixqtl.run_susie_default(x=hap1_t.numpy() + hap2_t.numpy(), y = log_counts_t.numpy() - cov_offset.numpy())
    elif mode == 'trcfine':
        cov_offset_ = cov_offset[M, 0]
        log_counts_t_ = log_counts_t[M]
        hap1_t_ = hap1_t[M, :]
        hap2_t_ = hap2_t[M, :]
        res = mixqtl.run_susie_default(x=hap1_t_.numpy() + hap2_t_.numpy(), y = log_counts_t_.numpy() - cov_offset_.numpy())
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_vars = rdf2pd(r_base.summary(res).rx2('vars'))
        df_cs = rdf2pd(r_base.summary(res).rx2('cs'))

    # re-order df_vars rows by position rather than significance
    df_ref = pd.DataFrame({'variable': [ i + 1 for i in range(hap1_t.shape[0]) ]})
    df_vars = pd.merge(df_ref, df_vars, on='variable', how='left')
    df_vars['variable_idx'] = df_vars['variable'] - 1
    df_vars = df_vars.drop(columns='variable')
    
    # unpack df_cs
    df_cs = unpack_cs(df_cs)
    df_cs['variable_idx'] = df_cs['variable'] - 1
    df_cs = df_cs.drop(columns='variable')
    df_cs = pd.merge(df_cs, df_vars[['variable_idx', 'variable_prob']], on='variable_idx')

    return df_vars, df_cs

class CS_collector:
    def __init__(self):
        self.base = None
    def init(self, dict_):
        if self.base is None:
            self.base = {}
            for k in dict_.keys():
                self.base[k] = []
    def update(self, dict_):
        self.init(dict_)
        for k in dict_.keys():
            self.base[k].append(dict_[k])
    def to_df(self):
        return pd.DataFrame(self.base)

def unpack_cs(df_cs):
    variable_list = df_cs.variable.to_list()
    base = CS_collector()
    for i in range(df_cs.shape[0]):
        var_str = variable_list[i].split(',')
        row = df_cs.iloc[i, :].to_dict()
        for k in var_str:
            row['variable'] = int(k)
            base.update(row)
    return base.to_df()

def rdf2pd(rdf):
    if r_base.is_null(rdf)[0] is True:
        return pd.DataFrame({})
    else:
        with localconverter(ro.default_converter + pandas2ri.converter):
            pddf = ro.conversion.rpy2py(rdf)
        return pddf

def run_mixfine(hap1_df, hap2_df, variant_df, libsize_df, counts_df, ref_df, alt_df,
                phenotype_pos_df, covariates_df, prefix,
                mode='mixfine',window=1000000, output_dir='.', 
                write_stats=True, logger=None, verbose=True,
                count_threshold=100, ase_threshold=50, ase_max=1000, weight_cap=100):
    """
    Fine-mapping mixFine based on mixQTL model proposed in 
    https://www.biorxiv.org/content/10.1101/2020.04.22.050666v1
    
    Two modes are available: mixfine or trcfine

    Fine-mapping variant-level PIPs and 95% credible set results 
    for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.finemap.mixFine.<chr>.parquet
    """
    assert np.all(counts_df.index==phenotype_pos_df.index)
    assert np.all(counts_df.columns==covariates_df.index)
    assert np.all(counts_df.columns==libsize_df.index)
    device = "cpu"

    if logger is None:
        logger = SimpleLogger()

    logger.write('Fine-mapping: for all phenotypes')
    logger.write('  * {} samples'.format(counts_df.shape[1]))
    logger.write('  * {} phenotypes'.format(counts_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(variant_df.shape[0]))

    genotype_ix = np.array([hap1_df.columns.tolist().index(i) for i in counts_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # add intercept to covariates
    covariates0_t = torch.tensor(np.c_[np.ones(covariates_df.shape[0]), covariates_df],
                                 dtype=torch.float32).to(device)
    
    # prepare library size vector
    libsize_t = torch.from_numpy(libsize_df.values[:, 0])

    igm = mixqtl.InputGeneratorMix(hap1_df, hap2_df, variant_df, counts_df, counts_df, 
                            ref_df, alt_df, phenotype_pos_df, window=window)
    # iterate over chromosomes
    start_time = time.time()
    k = 0
    logger.write('  * Finemapping')
    for chrom in igm.chrs:
        logger.write('    Mapping chromosome {}'.format(chrom))
        # allocate arrays
        n = 0
        for i in igm.phenotype_pos_df[igm.phenotype_pos_df['chr']==chrom].index:
            j = igm.cis_ranges[i]
            n += j[1] - j[0] + 1

        chr_res = OrderedDict()
        chr_res['phenotype_id'] =   []
        chr_res['variant_id'] =     []
        chr_res['tss_distance'] =   np.empty(n, dtype=np.int32)
        chr_res['maf'] =            np.empty(n, dtype=np.float32)
        chr_res['variable_prob'] =  np.empty(n, dtype=np.float32)
        chr_res['credible_set'] =   np.empty(n, dtype=np.int32)
        
        chr_res_cs = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variable'] =     []
        chr_res['credible_set'] = np.empty(n, dtype=np.int32)
        chr_res['cs_log10bf'] =   np.empty(n, dtype=np.float32)
        chr_res['cs_avg_r2'] =    np.empty(n, dtype=np.float32)
        chr_res['cs_min_r2'] =    np.empty(n, dtype=np.float32)

        start = 0
        for k, (raw_counts, _, ref, alt, hap1, hap2, genotype_range, phenotype_id) in enumerate(igm.generate_data(chrom=chrom, verbose=verbose), k+1):
            print(phenotype_id)
            # copy data to GPU
            hap1_t = torch.tensor(hap1, dtype=torch.float).to(device)
            hap2_t = torch.tensor(hap2, dtype=torch.float).to(device)
            # subset samples
            hap1_t = hap1_t[:,genotype_ix_t]
            hap2_t = hap2_t[:,genotype_ix_t]

            ref_t = torch.tensor(ref, dtype=torch.float).to(device)
            alt_t = torch.tensor(alt, dtype=torch.float).to(device)
            raw_counts_t = torch.tensor(raw_counts, dtype=torch.float).to(device)

            mixqtl.impute_hap(hap1_t, hap2_t)

            variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
            tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igm.phenotype_tss[phenotype_id])
            
            res, res_cs = finemapper(hap1_t, hap2_t, ref_t, alt_t, raw_counts_t, libsize_t, covariates0_t, 
                                     count_threshold=count_threshold, select_covariates=True, 
                                     ase_threshold=ase_threshold, ase_max=ase_max, weight_cap=weight_cap, 
                                     mode=mode)
            breakpoint() 
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
            chr_res_df['pval_meta'] = 2*stats.norm.cdf(-chr_res_df['tstat_meta'].abs())
            chr_res_df['pval_meta'][chr_res_df['method_meta'] == 'trc'] = chr_res_df['pval_trc'][chr_res_df['method_meta'] == 'trc'] 
 

            print('    * writing output')
            chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.mixQTL.{}.parquet'.format(prefix, chrom)))

    logger.write('done.')
