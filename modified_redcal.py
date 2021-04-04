



import numpy as np
from copy import deepcopy
import argparse
import os
import linsolve

from hera_cal import utils
from hera_cal import version
from hera_cal.noise import predict_noise_variance_from_autos
from hera_cal.datacontainer import DataContainer
from hera_cal.utils import split_pol, conj_pol, split_bl, reverse_bl, join_bl, join_pol, comply_pol
from hera_cal.io import HERAData, HERACal, write_cal, save_redcal_meta
from hera_cal.apply_cal import calibrate_in_place

## Importing functions
from hera_cal.redcal import _get_pol_load_list
from hera_cal.redcal import get_reds
from hera_cal.redcal import filter_reds
from hera_cal.redcal import redundantly_calibrate
from hera_cal.redcal import expand_omni_sol
from hera_cal.redcal import get_pos_reds
from hera_cal.redcal import add_pol_reds

### Fixing degenaracies
import hera_pspec as hp
import hera_cal as hc
from hera_sim import io

## Clustering algorithm
from sklearn.cluster import KMeans

SEC_PER_DAY = 86400.
IDEALIZED_BL_TOL = 1e-8  # bl_error_tol for redcal.get_reds when using antenna positions calculated from reds


def get_reds(antpos, pols=['nn'], pol_mode='1pol', bl_error_tol=1.0, include_autos=False):
    pos_reds = get_pos_reds(antpos, bl_error_tol=bl_error_tol, include_autos=include_autos)
    
    return add_pol_reds(pos_reds, pols=pols, pol_mode=pol_mode)


def get_custom_reds(hd, data_file, Number_of_clusters, red_groups_index, nInt_to_load=None, pol_mode='2pol', 
                    bl_error_tol=1.0, ex_ants=[], solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0,
                    fc_conv_crit=1e-6, fc_maxiter=50, oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, 
                    check_after=50, gain=.4, max_dims=2, verbose=False, **filter_reds_kwargs):
    
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
    
    clustered_baseline_groups = []
    for red_group_number in range(red_groups_index):
    
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

        ############################  'cluster_baselines2 complete' ######################################
        ##################################################################################################
        
        cluster_baselines2 = [n, true_labels] # Storing clustered baselines and their labels

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
                aa=np.int64(data_file.baseline_to_antnums(i)[0]), np.int64(data_file.baseline_to_antnums(i)[1]),'ee'
                
                ant_pair_cluster1.append(aa)
            reds_all_cluster.append(ant_pair_cluster1)
            
        clustered_baseline_groups.append(reds_all_cluster) ## Contains clustered baselines
        ################################  'get_custom_reds complete' #######################################
        ####################################################################################################
        
    """ Combines the clustered baselines groups with the original get_reds groups and also changes their
        polarisation of the groups from 'nn' to 'ee'.

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each list has a list of baselines that are clustered into the same group by the 
            clustering algorithm.
    """
    if nInt_to_load is not None:
        assert hd.filetype == 'uvh5', 'Partial loading only available for uvh5 filetype.'
    else:
        if hd.data_array is None:  # if data loading hasn't happened yet, load the whole file
            hd.read()
        if hd.times is None:  # load metadata into HERAData object if necessary
            for key, value in hd.get_metadata_dict().items():
                setattr(hd, key, value)

    # get basic antenna, polarization, and observation info
    nTimes, nFreqs = len(hd.times), len(hd.freqs)
    fSlice = slice(flag_nchan_low, nFreqs - flag_nchan_high)
    antpols = list(set([ap for pol in hd.pols for ap in split_pol(pol)]))
    ant_nums = np.unique(np.append(hd.ant_1_array, hd.ant_2_array))
    ants = [(ant, antpol) for ant in ant_nums for antpol in antpols]
    pol_load_list = _get_pol_load_list(hd.pols, pol_mode=pol_mode)

    # initialize gains to 1s, gain flags to True, and chisq to 0s
    rv = {}  # dictionary of return values
    rv['g_firstcal'] = {ant: np.ones((nTimes, nFreqs), dtype=np.complex64) for ant in ants}
    rv['gf_firstcal'] = {ant: np.ones((nTimes, nFreqs), dtype=bool) for ant in ants}
    rv['g_omnical'] = {ant: np.ones((nTimes, nFreqs), dtype=np.complex64) for ant in ants}
    rv['gf_omnical'] = {ant: np.ones((nTimes, nFreqs), dtype=bool) for ant in ants}
    rv['chisq'] = {antpol: np.zeros((nTimes, nFreqs), dtype=np.float32) for antpol in antpols}
    rv['chisq_per_ant'] = {ant: np.zeros((nTimes, nFreqs), dtype=np.float32) for ant in ants}

    # get reds and then intitialize omnical visibility solutions to all 1s and all flagged
    rd = get_reds({ant: hd.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                        pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    

    ## Replaces the original redundant groups by its clustered subgroups.
    all_reds = []
    #appends groups that are clustered
    for k1 in range(len(clustered_baseline_groups)):
        for k2 in range(len(clustered_baseline_groups[k1])):
            all_reds.append(clustered_baseline_groups[k1][k2])
    
    # Appends the un-clustered groups
    for k3 in range(len(clustered_baseline_groups),len(rd)):
        all_reds.append(rd[k3])

    return all_reds   








