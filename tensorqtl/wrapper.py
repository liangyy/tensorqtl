import torch
import numpy as np
import pandas as pd
import time, os

from collections import OrderedDict
import pdb

import genotypeio
from core import SimpleLogger, impute_mean, filter_maf

def name_to_index(mylist, name):
    return np.where(np.array(mylist) == name)[0][0]

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

    logger.write('cis-QTL mapping')
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
    mapper.init(phenotypes_t.T, covariates_t, **kwargs)
    phenotype_names = phenotype_df.index.to_list()

    
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=None, window=window)
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
        chr_res['pval'] = np.empty(n, dtype=np.float64)
        chr_res['b'] =        np.empty(n, dtype=np.float32)
        
        start = 0
        
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):
            # copy genotypes to GPU
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            impute_mean(genotypes_t)

            variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
            tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_id])
            
            phenotype_idx = name_to_index(phenotype_names, phenotype_id)

            ## mapper call
            res_i = mapper.map_one(genotypes_t.T, phenotype_idx)
            
            n = len(variant_ids)
            

            if n > 0:
                chr_res['phenotype_id'].extend([phenotype_id]*n)
                chr_res['variant_id'].extend(variant_ids)
                chr_res['tss_distance'][start:start+n] = tss_distance
                chr_res['pval'][start:start+n] = res_i[1]
                chr_res['b'][start:start+n] = res_i[0]
                
            start += n  # update pointer
        

        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

        # prepare output
        if start < len(chr_res['tss_distance']):
            for x in chr_res:
                chr_res[x] = chr_res[x][:start]

        chr_res_df = pd.DataFrame(chr_res)
            
        print('    * writing output')
        chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.{}.parquet'.format(prefix, chrom)))

    
    logger.write('done.')

    
