


def get_rbg_cluster(data_file, Number_of_clusters, red_groups_index):
    
    """ Combines the clustered baselines groups with the original get_reds groups and also changes their
        polarisation of the groups from 'nn' to 'ee'.
        
    Arguments:
        data_file: HERAData object,that has been calibrated by redcal.
        Number_of_clusters: The number of clusters you want to cluster the redundant baseline group into.
        red_groups_index: The index for that redundant baseline group to use for the clustering.
        
        
    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each list has a list of baselines that are clustered into the same group by the 
            clustering algorithm.
    """
    
    
    ant_nums = np.unique(np.append(data_file.ant_1_array, data_file.ant_2_array))
    pols=['nn']
    pol_mode='2pol'
    bl_error_tol=1.0

    #def _get_pol_load_list(pols, pol_mode='1pol'):
    '''Get a list of lists of polarizations to load simultaneously, depending on the polarizations
    in the data and the pol_mode (which can be 1pol, 2pol, 4pol, or 4pol_minV)'''

    if pol_mode in ['1pol', '2pol']:
        pol_load_list = [[pol] for pol in pols if split_pol(pol)[0] == split_pol(pol)[1]]
    elif pol_mode in ['4pol', '4pol_minV']:
        assert len(pols) == 4, 'For 4pol calibration, there must be four polarizations in the data file.'
        pol_load_list = [pols]
    else:
        raise ValueError('Unrecognized pol_mode: {}'.format(pol_mode))

    reds_all = get_reds({ant: data_file.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                            pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    

    ##############################  'all_reds_function complete' #######################################
    ####################################################################################################
    
    
    ########################--------------    Clustering Algorithm     ----------------#####################
    """ Using a clustering algorithm (K_means In this case) to classify baselines from a
        redundant_baseline_group into groups based on their visibilities.
    Returns:
        n : A list of baseline IDs for the baselnes in the redundant baseline group
        true_labels: A list of lables from the clustering algorithm where baselines of similar
                    visibilities are clustered into the same group. P.S. the number of groups we get
                    depend on the number of clusters specified.
    """
    Number_of_clusters = Number_of_clusters
    
    red_group_number = red_groups_index

    ## Changing the antenna numbers in to baselines for every redundant baseline group.
    redundant_baseline_group = []
    for bg in range(len(reds_all)):
        redundant_baselines = []
        for a_num in range(len(reds_all[bg])):
            bID = data_file.antnums_to_baseline(reds_all[bg][a_num][0],reds_all[bg][a_num][1])
            redundant_baselines.append(bID)
        redundant_baseline_group.append(redundant_baselines) ## Saving a list of redundant baselines in an  
                                                             ## array.


    n = redundant_baseline_group[red_group_number]   ## Getting one redundant baseline group


    ##  Combining a list (of visibilities) of 2D arrays into a 3D array
    array_3D = []
    for i in range(len(n)):
        array_3D.append(data_file.get_data(n[i]) )

    array_3D = np.array(array_3D)

    ## Appending multiple time sample into one array_per_baseline.
    ## e.g. for a nfreq = 120,ntimes=1 we have 1 X 120=120 elements per_baseline
    ###     for a nfreq = 120,ntimes=60 we have 60 X 120=7200 elements per_baseline
    new_time = []
    for i in range(len(n)):
        abc = abs(array_3D[i])
        d1 = []
        for j in range(len(abc)):
            d1.append(abc[j])
            d2 = np.concatenate(d1)  ### appending all time samples for each baseline
        new_time.append(d2)

    new_time = np.array(new_time)


    X = new_time
    y = n
    # Incorrect number of clusters
    true_labels = KMeans(n_clusters=Number_of_clusters).fit_predict(X)

    cluster_baselines2 = [n, true_labels] # Storing clustered baselines and their labels

    ############################  'cluster_baselines2 complete' ######################################
    ##################################################################################################


    """ It uses the labels we get when we run the clustering algorithm, to get the index of each label
            in order to find out which label belongs to which baseline. 
    Returns:
        A 2D list of Baseline ID that are grouped in the way the clustering algorithm clustered them. The
            result is in the form, len(2D_list) = Number_of_clusters.
    """

    ## Calling the cluster_baselines function to get the labels
    labels = cluster_baselines2[1]

    ## Getting rid of repeated labels
    p = list(dict.fromkeys(labels))

    ## Getting the index of each label in order to find out which label belongs to which baseline
    x = labels
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    clusters = []

    for j in range(len(p)):
        cluster1 = get_indexes(p[j],x) ## a list of indeces for each label
        baseline_array = np.array(cluster_baselines2[0])
        base_ant_idx = baseline_array[cluster1] ## index[0]=n 
        clusters.append(base_ant_idx)

    ############################  'get_baseline_cluster2 complete' #####################################
    ####################################################################################################

    """ Combines cluster_baselines() and get_baseline_cluster().

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each list has a list of baselines that are clustered into the same group by the 
            clustering algorithm.
    """

    antpp_c = clusters

    reds_all_cluster = []
    for j in range(len(antpp_c)):
        ant_pair_cluster1 =[]
        for i in antpp_c[j]:
            #Print baselines with the antenna numbers that makeup that baseline
            ant_pair_pol=np.int64(data_file.baseline_to_antnums(i)[0]), np.int64(data_file.baseline_to_antnums(i)[1]),'ee'

            ant_pair_cluster1.append(ant_pair_pol)
        reds_all_cluster.append(ant_pair_cluster1)

        
    return reds_all_cluster

