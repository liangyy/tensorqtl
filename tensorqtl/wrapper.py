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

    
