import torch
import numpy as np
import pandas as pd
import time

import pdb

import genotypeio
from core import SimpleLogger, impute_mean, filter_maf


def map_trans(genotype_df, phenotype_df, covariates_df, mapper, pval_threshold=1e-5, 
              maf_threshold=0.05, batch_size=20000,
              logger=None, verbose=True, kwargs={}):
    '''
    Wrapper for trans-QTL mapping.
    The QTL caller is `mapper` which should have 
    * mapper.init(phenotype, covariate)
    * mapper.map(genotype) 
    implemented.
    mapper.map should return 'bhat', 'pval' in a dictionary.
    
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger(verbose=verbose)
    assert np.all(phenotype_df.columns==covariates_df.index)

    variant_ids = genotype_df.index.tolist()
    variant_dict = {i:j for i,j in enumerate(variant_ids)}
    n_variants = len(variant_ids)
    n_samples = phenotype_df.shape[1]

    logger.write('trans-QTL mapping')
    logger.write('  * {} samples'.format(n_samples))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(n_variants))
    

    phenotypes_t = torch.tensor(phenotype_df.values, dtype=torch.float32).to(device)
    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    
    ## mapper call
    mapper.init(phenotypes_t.T, covariates_t, **kwargs)
    # genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    # genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    
    ggt = genotypeio.GenotypeGeneratorTrans(genotype_df, batch_size=batch_size)
    start_time = time.time()
    res = []
    n_variants = 0
    for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(verbose=verbose), 1):
        # copy genotypes to GPU
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)

        # filter by MAF
        # genotypes_t = genotypes_t[:,genotype_ix_t]
        impute_mean(genotypes_t)
        genotypes_t, variant_ids, maf_t = filter_maf(genotypes_t, variant_ids, maf_threshold)
        n_variants += genotypes_t.shape[0]
        
        ## mapper call
        res_i = mapper.map(genotypes_t.T)
        
        del genotypes_t
        
        res_i = np.c_[ 
            np.repeat(variant_ids, phenotype_df.index.shape[0]),
            np.tile(phenotype_df.index, variant_ids.shape[0]),
            res_i[0],
            res_i[1],
            np.repeat(maf_t.cpu(), phenotype_df.index.shape[0])
        ]
        res.append(res_i)
        
    logger.write('    elapsed time: {:.2f} min'.format((time.time()-start_time)/60))
    del phenotypes_t

    # post-processing: concatenate batches
    res = np.concatenate(res)
    pval_df = pd.DataFrame(res, columns=['variant_id', 'phenotype_id', 'bhat', 'pval', 'maf'])

    if maf_threshold > 0:
        logger.write('  * {} variants passed MAF >= {:.2f} filtering'.format(n_variants, maf_threshold))
    logger.write('done.')
    return pval_df

def map_cis(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
            covariates_df, mapper, prefix, 
            window=1000000, output_dir='.', 
            logger=None, verbose=True, kwargs={}):
    '''
    Wrapper for cis-QTL mapping.
    The QTL caller is `mapper` which should have 
    * mapper.init(phenotype, covariate)
    * mapper.map(genotype) 
    implemented.
    mapper.map_one should return 'bhat', 'pval' in a dictionary.
    '''
    
    assert np.all(phenotype_df.columns==covariates_df.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()
    if group_s is not None:
        group_dict = group_s.to_dict()

    logger.write('cis-QTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(variant_df.shape[0]))
    

    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    phenotypes_t = torch.tensor(phenotype_df.values, dtype=torch.float32).to(device)
    
    # FIXME: this is not ideal since we may initialize for some phenotypes that does not have cis genotype.
    # So, for now, as it is not taken care of inside the caller, 
    # we need to make sure that these phenotypes are not part of the input.
    ## mapper call
    mapper.init(phenotypes_t, covariates_t, **kwargs)
    phenotype_names = phenotype_df.index.to_list()

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
    
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=group_s, window=window)
    # iterate over chromosomes
    best_assoc = []
    start_time = time.time()
    k = 0
    logger.write('  * Computing associations')
    for chrom in igc.chrs:
        logger.write('    Mapping chromosome {}'.format(chrom))
        # allocate arrays
        n = 0
        for i in igc.phenotype_pos_df[igc.phenotype_pos_df['chr']==chrom].index:
            j = igc.cis_ranges[i]
            n += j[1] - j[0] + 1
        
        chr_res = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variant_id'] = []
        chr_res['tss_distance'] = np.empty(n, dtype=np.int32)
        chr_res['maf'] =          np.empty(n, dtype=np.float32)
        chr_res['pval'] = np.empty(n, dtype=np.float64)
        chr_res['b'] =        np.empty(n, dtype=np.float32)
        chr_res['b_se'] =     np.empty(n, dtype=np.float32)
        
        start = 0
        if group_s is None:
            for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):
                # copy genotypes to GPU
                genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
                genotypes_t = genotypes_t[:,genotype_ix_t]
                impute_mean(genotypes_t)

                variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
                tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_id])

                ## mapper call
                res_i = mapper.map_one(genotypes_t, phenotype_idx)
                
                res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
                tstat, slope, slope_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                n = len(variant_ids)
                

                if n > 0:
                    chr_res['phenotype_id'].extend([phenotype_id]*n)
                    chr_res['variant_id'].extend(variant_ids)
                    chr_res['tss_distance'][start:start+n] = tss_distance
                    chr_res['maf'][start:start+n] = maf
                    chr_res['ma_samples'][start:start+n] = ma_samples
                    chr_res['ma_count'][start:start+n] = ma_count
                    if interaction_s is None:
                        chr_res['pval_nominal'][start:start+n] = tstat
                        chr_res['slope'][start:start+n] = slope
                        chr_res['slope_se'][start:start+n] = slope_se
                    else:
                        chr_res['pval_g'][start:start+n]  = tstat[:,0]
                        chr_res['b_g'][start:start+n]     = b[:,0]
                        chr_res['b_g_se'][start:start+n]  = b_se[:,0]
                        chr_res['pval_i'][start:start+n]  = tstat[:,1]
                        chr_res['b_i'][start:start+n]     = b[:,1]
                        chr_res['b_i_se'][start:start+n]  = b_se[:,1]
                        chr_res['pval_gi'][start:start+n] = tstat[:,2]
                        chr_res['b_gi'][start:start+n]    = b[:,2]
                        chr_res['b_gi_se'][start:start+n] = b_se[:,2]
                start += n  # update pointer
        else:  # groups
            for k, (phenotypes, genotypes, genotype_range, phenotype_ids, group_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):

                # copy genotypes to GPU
                genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
                genotypes_t = genotypes_t[:,genotype_ix_t]
                impute_mean(genotypes_t)

                variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
                # assuming that the TSS for all grouped phenotypes is the same
                tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_ids[0]])

                if interaction_s is not None:
                    genotypes_t, mask_t = filter_maf_interaction(genotypes_t, interaction_mask_t=interaction_mask_t,
                                                                 maf_threshold_interaction=maf_threshold_interaction)
                    mask = mask_t.cpu().numpy()
                    variant_ids = variant_ids[mask]
                    tss_distance = tss_distance[mask]

                n = len(variant_ids)

                if genotypes_t.shape[0]>0:
                    # process first phenotype in group
                    phenotype_id = phenotype_ids[0]
                    phenotype_t = torch.tensor(phenotypes[0], dtype=torch.float).to(device)

                    if interaction_s is None:
                        res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
                        tstat, slope, slope_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                    else:
                        res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), interaction_t, residualizer, return_sparse=False)
                        tstat, b, b_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                    px = [phenotype_id]*n

                    # iterate over remaining phenotypes in group
                    for phenotype, phenotype_id in zip(phenotypes[1:], phenotype_ids[1:]):
                        phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                        if interaction_s is None:
                            res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
                            tstat0, slope0, slope_se0, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                        else:
                            res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), interaction_t, residualizer, return_sparse=False)
                            tstat0, b0, b_se0, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]

                        # find associations that are stronger for current phenotype
                        if interaction_s is None:
                            ix = np.where(np.abs(tstat0) > np.abs(tstat))[0]
                        else:
                            ix = np.where(np.abs(tstat0[:,2]) > np.abs(tstat[:,2]))[0]

                        # update relevant positions
                        for j in ix:
                            px[j] = phenotype_id
                        if interaction_s is None:
                            tstat[ix] = tstat0[ix]
                            slope[ix] = slope0[ix]
                            slope_se[ix] = slope_se0[ix]
                        else:
                            tstat[ix] = tstat0[ix]
                            b[ix] = b0[ix]
                            b_se[ix] = b_se0[ix]

                    chr_res['phenotype_id'].extend(px)
                    chr_res['variant_id'].extend(variant_ids)
                    chr_res['tss_distance'][start:start+n] = tss_distance
                    chr_res['maf'][start:start+n] = maf
                    chr_res['ma_samples'][start:start+n] = ma_samples
                    chr_res['ma_count'][start:start+n] = ma_count
                    if interaction_s is None:
                        chr_res['pval_nominal'][start:start+n] = tstat
                        chr_res['slope'][start:start+n] = slope
                        chr_res['slope_se'][start:start+n] = slope_se
                    else:
                        chr_res['pval_g'][start:start+n]  = tstat[:,0]
                        chr_res['b_g'][start:start+n]     = b[:,0]
                        chr_res['b_g_se'][start:start+n]  = b_se[:,0]
                        chr_res['pval_i'][start:start+n]  = tstat[:,1]
                        chr_res['b_i'][start:start+n]     = b[:,1]
                        chr_res['b_i_se'][start:start+n]  = b_se[:,1]
                        chr_res['pval_gi'][start:start+n] = tstat[:,2]
                        chr_res['b_gi'][start:start+n]    = b[:,2]
                        chr_res['b_gi_se'][start:start+n] = b_se[:,2]

                    # top association for the group
                    if interaction_s is not None:
                        ix = np.nanargmax(np.abs(tstat[:,2]))
                        top_s = pd.Series([chr_res['phenotype_id'][start:start+n][ix], variant_ids[ix], tss_distance[ix], maf[ix], ma_samples[ix], ma_count[ix],
                                           tstat[ix,0], b[ix,0], b_se[ix,0],
                                           tstat[ix,1], b[ix,1], b_se[ix,1],
                                           tstat[ix,2], b[ix,2], b_se[ix,2]], index=chr_res.keys())
                        top_s['num_phenotypes'] = len(phenotype_ids)
                        if run_eigenmt:  # compute eigenMT correction
                            top_s['tests_emt'] = eigenmt.compute_tests(genotypes_t, var_thresh=0.99, variant_window=200)
                        best_assoc.append(top_s)

                start += n  # update pointer

        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

        # convert to dataframe, compute p-values and write current chromosome
        if start < len(chr_res['maf']):
            for x in chr_res:
                chr_res[x] = chr_res[x][:start]

        if write_stats:
            chr_res_df = pd.DataFrame(chr_res)
            if interaction_s is None:
                m = chr_res_df['pval_nominal'].notnull()
                chr_res_df.loc[m, 'pval_nominal'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_nominal'].abs(), dof)
            else:
                m = chr_res_df['pval_gi'].notnull()
                chr_res_df.loc[m, 'pval_g'] =  2*stats.t.cdf(-chr_res_df.loc[m, 'pval_g'].abs(), dof)
                chr_res_df.loc[m, 'pval_i'] =  2*stats.t.cdf(-chr_res_df.loc[m, 'pval_i'].abs(), dof)
                chr_res_df.loc[m, 'pval_gi'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_gi'].abs(), dof)
            print('    * writing output')
            chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.{}.parquet'.format(prefix, chrom)))

    
    logger.write('done.')

    
