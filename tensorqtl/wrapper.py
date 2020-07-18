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
    mapper.init(phenotypes_t, covariates_t, **kwargs)
    
    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    
    ggt = genotypeio.GenotypeGeneratorTrans(genotype_df, batch_size=batch_size)
    start_time = time.time()
    res = []
    n_variants = 0
    for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(verbose=verbose), 1):
        # copy genotypes to GPU
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)

        # filter by MAF
        genotypes_t = genotypes_t[:,genotype_ix_t]
        impute_mean(genotypes_t)
        genotypes_t, variant_ids, maf_t = filter_maf(genotypes_t, variant_ids, maf_threshold)
        n_variants += genotypes_t.shape[0]
        
        ## mapper call
        res_i = mapper.map(genotypes_t)
        
        del genotypes_t

        # if return_sparse:
        #     m = r_t.abs() >= r_threshold
        #     ix_t = m.nonzero()  # sparse index
        #     ix = ix_t.cpu().numpy()
        # 
        #     r_t = r_t.masked_select(m).type(torch.float64)
        #     r2_t = r_t.pow(2)
        #     tstat_t = r_t * torch.sqrt(dof / (1 - r2_t))
        #     std_ratio_t = torch.sqrt(phenotype_var_t[ix_t[:,1]] / genotype_var_t[ix_t[:,0]])
        #     b_t = r_t * std_ratio_t
        #     b_se_t = (b_t / tstat_t).type(torch.float32)
        # 
        #     res.append(np.c_[
        #         variant_ids[ix[:,0]], phenotype_df.index[ix[:,1]],
        #         tstat_t.cpu(), b_t.cpu(), b_se_t.cpu(),
        #         r2_t.float().cpu(), maf_t[ix_t[:,0]].cpu()
        #     ])
        # else:
        #     r_t = r_t.type(torch.float64)
        #     tstat_t = r_t * torch.sqrt(dof / (1 - r_t.pow(2)))
        #     res.append(np.c_[variant_ids, tstat_t.cpu()])
        res.append(res_i)
        
    logger.write('    elapsed time: {:.2f} min'.format((time.time()-start_time)/60))
    del phenotypes_t

    # post-processing: concatenate batches
    res = np.concatenate(res)
    if return_sparse:
        res[:,2] = 2*stats.t.cdf(-np.abs(res[:,2].astype(np.float64)), dof)
        pval_df = pd.DataFrame(res, columns=['variant_id', 'phenotype_id', 'pval', 'b', 'b_se', 'r2', 'maf'])
        pval_df['pval'] = pval_df['pval'].astype(np.float64)
        pval_df['b'] = pval_df['b'].astype(np.float32)
        pval_df['b_se'] = pval_df['b_se'].astype(np.float32)
        pval_df['r2'] = pval_df['r2'].astype(np.float32)
        pval_df['maf'] = pval_df['maf'].astype(np.float32)
        if not return_r2:
            pval_df.drop('r2', axis=1, inplace=True)
    else:
        pval = 2*stats.t.cdf(-np.abs(res[:,1:].astype(np.float64)), dof)
        pval_df = pd.DataFrame(pval, index=res[:,0], columns=phenotype_df.index)
        pval_df.index.name = 'variant_id'

    if maf_threshold > 0:
        logger.write('  * {} variants passed MAF >= {:.2f} filtering'.format(n_variants, maf_threshold))
    logger.write('done.')
    return pval_df

    