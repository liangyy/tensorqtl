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
        log_counts_t = raw_counts_t.clone()
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
    cov_offset = log_counts_t.reshape(1,-1).T.clone()
    cov_offset[:] = 0
    
    if covariates_t.shape[1] != 0:
        cov_offset[M_cov == True] = mixqtl.get_offset(covariates_t[mask_cov, :], log_counts_t[mask_cov])
    
    if mode == 'mixfine':
        # o = {'geno1': hap1_t, 'geno2': hap2_t,
        #                      'y1': ref_t, 'y2': alt_t,
        #                      'ytotal': raw_counts_t,
        #                      'lib_size': libsize_t, 'cov_offset': cov_offset }
        # import pickle 
        # with open('test.pkl', 'wb') as f:
        #     pickle.dump(o, f)
        res = r_mixqtl.mixfine(geno1=hap1_t.T.numpy(), geno2=hap2_t.T.numpy(), 
                             y1=ref_t.numpy(), y2=alt_t.numpy(), 
                             ytotal=raw_counts_t.numpy(), 
                             lib_size=libsize_t.numpy(), cov_offset=cov_offset[:, 0].numpy(),
                             trc_cutoff=count_threshold, asc_cutoff=ase_threshold, 
                             asc_cap=ase_max, weight_cap=weight_cap, nobs_asc_cutoff=3)
    elif mode == 'nefine':
        # o = {'geno': hap1_t.T.numpy() + hap2_t.T.numpy(),
        #                      'y': log_counts_t.numpy(),
        #                      'cov_offset': cov_offset[:, 0].numpy() }
        # import pickle
        # with open('test.pkl', 'wb') as f:
        #     pickle.dump(o, f)
        # breakpoint()
        res = r_mixqtl.run_susie_default(x=hap1_t.T.numpy() + hap2_t.T.numpy(), y = log_counts_t.numpy() - cov_offset[:, 0].numpy())
    elif mode == 'trcfine':
        cov_offset_ = cov_offset[M, 0]
        log_counts_t_ = log_counts_t[M]
        hap1_t_ = hap1_t[:, M]
        hap2_t_ = hap2_t[:, M]
        res = r_mixqtl.run_susie_default(x=hap1_t_.T.numpy() + hap2_t_.T.numpy(), y = log_counts_t_.numpy() - cov_offset_[:, 0].numpy())
    if 'cs' in r_base.names(res):
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_cs = res.rx2('cs')
            df_vars = res.rx2('vars')
    else:    
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_vars = ro.conversion.rpy2py(r_base.summary(res).rx2('vars'))
            df_cs = ro.conversion.rpy2py(r_base.summary(res).rx2('cs'))
    
    df_vars = post_check(df_vars)
    df_cs = post_check(df_cs) 
    # print(df_cs.head())

    # re-order df_vars rows by position rather than significance
    df_ref = pd.DataFrame({'variable': [ i + 1 for i in range(hap1_t.shape[0]) ]})
    df_vars = pd.merge(df_ref, df_vars, on='variable', how='left')
    df_vars['variable_idx'] = df_vars['variable'] - 1
    df_vars = df_vars.drop(columns='variable')
    
    # unpack df_cs
    if df_cs.shape[0] > 0:
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
    tmp = base.to_df()
    tmp['cs'] = tmp['cs'].astype(int)
    return tmp

def post_check(rdf):
    if isinstance(rdf, pd.DataFrame):
        return rdf
    else:
        return pd.DataFrame({})

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
    in the format 
    <output_dir>/<prefix>.finemap_pip.<mode>.<chr>.parquet
    <output_dir>/<prefix>.finemap_cs.<mode>.<chr>.parquet
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

        chr_res = []
        chr_res_cols = ['phenotype_id', 'variant_id', 'tss_distance', 'variable_prob', 'cs']
        
        chr_res_cs = []
        chr_res_cs_cols = ['phenotype_id', 'variant_id', 'cs', 'cs_log10bf', 'cs_avg_r2', 'cs_min_r2', 'variable_prob']

        start = 0
        for k, (raw_counts, _, ref, alt, hap1, hap2, genotype_range, phenotype_id) in enumerate(igm.generate_data(chrom=chrom, verbose=verbose), k+1):
            
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
            
            res['phenotype_id'] = phenotype_id
            res['variant_id'] = variant_ids
            res['tss_distance'] = tss_distance
            res_append = res[chr_res_cols]
            chr_res.append(res_append.copy())
            
            if res_cs.shape[0] > 0:
                res_cs['phenotype_id'] = phenotype_id
                res_cs['variant_id'] = variant_ids[res_cs.variable_idx]
                res_cs = res_cs[chr_res_cs_cols]
            chr_res_cs.append(res_cs.copy())


        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

        # convert to dataframe, compute p-values and write current chromosome
        df_res = pd.concat(chr_res, axis=0)
        df_res_cs = pd.concat(chr_res_cs, axis=0)

 

        print('    * writing output')
        df_res.to_parquet(os.path.join(output_dir, '{}.finemap_pip.{}.{}.parquet'.format(prefix, mode, chrom)))
        df_res_cs.to_parquet(os.path.join(output_dir, '{}.finemap_cs.{}.{}.parquet'.format(prefix, mode, chrom)))

    logger.write('done.')
