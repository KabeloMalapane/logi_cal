


# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
plt.rcParams.update({'font.size': 14})

from copy import deepcopy
import argparse
import os
import linsolve

from hera_cal import utils
from hera_cal import version
from hera_cal.noise import predict_noise_variance_from_autos
from hera_cal.datacontainer import DataContainer
from hera_cal.utils import split_pol, conj_pol, split_bl, reverse_bl, join_bl, join_pol, comply_pol
from hera_qm.ant_metrics import per_antenna_modified_z_scores
from hera_cal.redcal import _redcal_run_write_results
from hera_cal.io import HERAData, HERACal, write_cal, save_redcal_meta
from hera_cal.apply_cal import calibrate_in_place

## Importing functions
from hera_cal.redcal import _get_pol_load_list, filter_reds, redundantly_calibrate, expand_omni_sol,get_pos_reds ,add_pol_reds


### Fixing degenaracies


import hera_pspec as hp
import hera_cal as hc
from hera_sim import io

## Classification
from sklearn.cluster import KMeans

# import uvtools
# import hera_cal as hc
# import hera_pspec as hp
# from pyuvdata import UVCal, UVData
# import pyuvdata.utils as uvutils




SEC_PER_DAY = 86400.
IDEALIZED_BL_TOL = 1e-8  # bl_error_tol for redcal.get_reds when using antenna positions calculated from reds





def cluster_baselines(data_file, Number_of_clusters):
    """ Using a clustering algorithm (K_means In this case) to classify baselines from a
        redundant_baseline_group into groups based on their visibilities.
    Returns:
        n : A list of baseline IDs for the baselnes in the redundant baseline group
        true_labels: A list of lables from the clustering algorithm where baselines of similar
                    visibilities are clustered into the same group. P.S. the number of groups we get
                    depend on the number of clusters specified.
    """
    Number_of_clusters = Number_of_clusters
    
    ## Getting one redundant baseline group
    red_base = data_file.get_redundancies(tol=1.0, use_antpos=False, include_conjugates=False,include_autos=True,
                                   conjugate_bls=False)

    n = red_base[0][1]  
    n = np.array(n)     ## Saving a list of redundant baselines in an array.

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
    return n, true_labels



def get_baseline_cluster(data_file, Number_of_clusters):
    """ It uses the labels we get when we run the clustering algorithm, to get the index of each label
            in order to find out which label belongs to which baseline. 
    Returns:
        A 2D list of Baseline ID that are grouped in the way the clustering algorithm clustered them. The
            result is in the form, len(2D_list) = Number_of_clusters.
    """
    ## Calling the cluster_baselines function to get the labels
    labels = cluster_baselines(data_file, Number_of_clusters)[1]
    
    ## Getting rid of repeated labels
    p = list(dict.fromkeys(labels))
    
    ## Getting the index of each label in order to find out which label belongs to which baseline
    x = labels
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    
    clusters = []
    for j in range(len(p)):
        cluster1 = get_indexes(p[j],x) ## a list of indeces for each label
        base_ant_idx = cluster_baselines(data_file, Number_of_clusters)[0][cluster1] ## index[0]=n
        clusters.append(base_ant_idx)
    return clusters


def custom_reds(data_file, Number_of_clusters):
    
    """ Combines cluster_baselines() and get_baseline_cluster().

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each list has a list of baselines that are clustered into the same group by the 
            clustering algorithm.
    """
    
    antpp_c = get_baseline_cluster(data_file,Number_of_clusters)

    #print(aa)
    reds_all_cluster = []
    for j in range(len(antpp_c)):
        ant_pair_cluster1 =[]
        for i in antpp_c[j]:
            #Print baselines with the antenna numbers that makeup that baseline
            aa =  np.int64(data_file.baseline_to_antnums(i)[0]), np.int64(data_file.baseline_to_antnums(i)[1]), 'ee'
            ant_pair_cluster1.append(aa)
        reds_all_cluster.append(ant_pair_cluster1)
    return reds_all_cluster

def get_custom_reds(calib_data_file,uncalib_data_file,Number_of_clusters):
    
    rd = get_reds({ant: uncalib_data_file.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                        pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    
    all_reds =[]
    clust =  custom_reds(calib_data_file,Number_of_clusters)
    for i in range(len(rd)):
        if i==2:
            for j in range(len(clust)):
                all_reds.append(clust[j])
        else:
            all_reds.append(rd[i])

    return all_reds



#############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################





def get_reds(antpos, pols=['nn'], pol_mode='1pol', bl_error_tol=1.0, include_autos=False):
    pos_reds = get_pos_reds(antpos, bl_error_tol=bl_error_tol, include_autos=include_autos)
    

    return add_pol_reds(pos_reds, pols=pols, pol_mode=pol_mode)

def all_reds_function(data_file):
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

    return reds_all




def cluster_baselines2(data_file, Number_of_clusters, red_groups_index):
    """ Using a clustering algorithm (K_means In this case) to classify baselines from a
        redundant_baseline_group into groups based on their visibilities.
    Returns:
        n : A list of baseline IDs for the baselnes in the redundant baseline group
        true_labels: A list of lables from the clustering algorithm where baselines of similar
                    visibilities are clustered into the same group. P.S. the number of groups we get
                    depend on the number of clusters specified.
    """
    Number_of_clusters = Number_of_clusters
    reds_all = all_reds_function(data_file)
    
    ## Changing the antenna numbers in to baselines for every redundant baseline group.
    redundant_baseline_group = []
    for bg in range(len(reds_all)):
        redundant_baselines = []
        for a_num in range(len(reds_all[bg])):
            bID = data_file.antnums_to_baseline(reds_all[bg][a_num][0],reds_all[bg][a_num][1])
            redundant_baselines.append(bID)
        redundant_baseline_group.append(redundant_baselines) ## Saving a list of redundant baselines in an array.
    
    
    n = redundant_baseline_group[red_groups_index]   ## Getting one redundant baseline group
    
    
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
    
  
    return n, true_labels



def get_baseline_cluster2(data_file, Number_of_clusters,red_groups_index):
    """ It uses the labels we get when we run the clustering algorithm, to get the index of each label
            in order to find out which label belongs to which baseline. 
    Returns:
        A 2D list of Baseline ID that are grouped in the way the clustering algorithm clustered them. The
            result is in the form, len(2D_list) = Number_of_clusters.
    """
    ## Calling the cluster_baselines function to get the labels
    labels = cluster_baselines2(data_file, Number_of_clusters,red_groups_index)[1]
    
    ## Getting rid of repeated labels
    p = list(dict.fromkeys(labels))
    
    ## Getting the index of each label in order to find out which label belongs to which baseline
    x = labels
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    
    clusters = []
    for j in range(len(p)):
        cluster1 = get_indexes(p[j],x) ## a list of indeces for each label
        baseline_array = np.array(cluster_baselines2(data_file, Number_of_clusters,red_groups_index)[0])
        base_ant_idx = baseline_array[cluster1] ## index[0]=n 
        clusters.append(base_ant_idx)
    

    return clusters


def get_custom_reds2(data_file, Number_of_clusters,red_groups_index):
    
    """ Combines cluster_baselines() and get_baseline_cluster().

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each list has a list of baselines that are clustered into the same group by the 
            clustering algorithm.
    """
    
    antpp_c = get_baseline_cluster2(data_file,Number_of_clusters,red_groups_index)


    reds_all_cluster = []
    for j in range(len(antpp_c)):
        ant_pair_cluster1 =[]
        for i in antpp_c[j]:
            #Print baselines with the antenna numbers that makeup that baseline
            aa =  np.int64(data_file.baseline_to_antnums(i)[0]), np.int64(data_file.baseline_to_antnums(i)[1]), 'ee'
            ant_pair_cluster1.append(aa)
        reds_all_cluster.append(ant_pair_cluster1)
        

    return reds_all_cluster



def custom_reds2(hd, data_file, Number_of_clusters, red_groups_index, nInt_to_load=None, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[],
                     solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0, fc_conv_crit=1e-6,
                     fc_maxiter=50, oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50,
                     gain=.4, max_dims=2, verbose=False, **filter_reds_kwargs):
    
        
    """ Combines the clustered baselines groups with the original get_reds groups and also changes their polarisation
        of the groups from 'nn' to 'ee'.

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

#    get reds and then intitialize omnical visibility solutions to all 1s and all flagged
    rd = get_reds({ant: hd.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                        pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    
    clustered_baseline_groups = []
    for z in range(red_groups_index):
        clustered_baseline_groups.append(get_custom_reds2(data_file,Number_of_clusters,z))

    ## Replaces the original redundand groups by its clustered subgroups.
    all_reds = []
    #appends groups that are clustered
    for k1 in range(len(clustered_baseline_groups)):
        for k2 in range(len(clustered_baseline_groups[k1])):
            all_reds.append(clustered_baseline_groups[k1][k2])
    
    # Appends the un-clustered groups
    for k3 in range(len(clustered_baseline_groups),len(rd)):
        all_reds.append(rd[k3])
    

    return all_reds




########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


def custom_reds_option2(hd, data_file, k_value_start, Number_of_clusters, red_groups_index, nInt_to_load=None, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[],
                     solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0, fc_conv_crit=1e-6,
                     fc_maxiter=50, oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50,
                     gain=.4, max_dims=2, verbose=False, **filter_reds_kwargs):
    
        
    """ Combines the clustered baselines groups with the original get_reds groups and also changes their polarisation
        of the groups from 'nn' to 'ee'.
        - This function involves clustering the earlier specified redundant baseline groups(RBGs) into more subgroups and 
        the later RBGs into fewer subgroups. As an example for the 60 specified RBGs, the groups could be split into four,
        0-15 will use k=6, 15-30 will use k=5, 30-45 will suse k=4 and 45-60 will use k=3. 

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

#    get reds and then intitialize omnical visibility solutions to all 1s and all flagged
    rd = get_reds({ant: hd.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                        pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    
    
    clustered_baseline_groups =[]
    RBG = red_groups_index

    ## Save k-values in a list to be used in the calibration process
    k_v_list = np.arange(k_value_start ,Number_of_clusters+1)
    k_v_list = list(k_v_list)
    ## reverse the k-values so that ealier RBGs get higher k-values and later RBGs get lower k-values e.g.RBG = [0,10],k=5 and RBG = [10,20],k=4  
    k_v_list.sort(reverse=True)  

    ## Get a list of Redundant baseline groups to be used as ranges based on the overall k-value used.
    array1 = np.linspace(RBG/(len(k_v_list)), RBG , len(k_v_list))

    RBG_range = [int(x) for x in array1]
    RBG_range.append(0)
    RBG_range.sort()

    
    for rbg in range(len(RBG_range)-1):
        for z in range(RBG_range[rbg],RBG_range[rbg+1]):
#             print(k_v_list[rbg],'{},{}'.format(RBG_range[rbg],RBG_range[rbg+1]))
            clustered_baseline_groups.append(get_custom_reds2(data_file,k_v_list[rbg] ,z))


    ## Replaces the original redundand groups by its clustered subgroups.
    all_reds = []
    #appends groups that are clustered
    for k1 in range(len(clustered_baseline_groups)):
        for k2 in range(len(clustered_baseline_groups[k1])):
            all_reds.append(clustered_baseline_groups[k1][k2])
    
    # Appends the un-clustered groups
    for k3 in range(len(clustered_baseline_groups),len(rd)):
        all_reds.append(rd[k3])
    

    return all_reds

####################################################################################################################################################################################################################################################################################################################################################################################################################




def custom_reds_option3(hd, data_file, k_value_start, Number_of_clusters, red_groups_index, nInt_to_load=None, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[],
                     solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0, fc_conv_crit=1e-6,
                     fc_maxiter=50, oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50,
                     gain=.4, max_dims=2, verbose=False, **filter_reds_kwargs):
    
        
    """ Combines the clustered baselines groups with the original get_reds groups and also changes their polarisation
        of the groups from 'nn' to 'ee'.
        - Clustering all the specified RBGs using a range of random k-values. Every specifies Redundant baseline group is
        clustered using a random k-value (from a given range).

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

#    get reds and then intitialize omnical visibility solutions to all 1s and all flagged
    rd = get_reds({ant: hd.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                        pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    
    clustered_baseline_groups = []
    
    k_values = np.arange(k_value_start,Number_of_clusters+1)
    k_values_used = []
    
    for z in range(red_groups_index):
        k_v = random.choice(k_values)
        clustered_baseline_groups.append(get_custom_reds2(data_file,k_v,z))
        k_values_used.append(k_v)


    ## Replaces the original redundand groups by its clustered subgroups.
    all_reds = []
    #appends groups that are clustered
    for k1 in range(len(clustered_baseline_groups)):
        for k2 in range(len(clustered_baseline_groups[k1])):
            all_reds.append(clustered_baseline_groups[k1][k2])
    
    # Appends the un-clustered groups
    for k3 in range(len(clustered_baseline_groups),len(rd)):
        all_reds.append(rd[k3])
    

    return all_reds, k_values_used




####################################################################################################################################################################################################################################################################################################################################################################################################################




def redcal_iteration_custom2(hd, customized_groups,min_bl_cut,max_bl_cut, nInt_to_load=None, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[],
                     solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0, fc_conv_crit=1e-6,
                     fc_maxiter=50, oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50,
                     gain=.4, max_dims=2, verbose=False, **filter_reds_kwargs):
    '''Perform redundant calibration (firstcal, logcal, and omnical) an entire HERAData object, loading only
    nInt_to_load integrations at a time and skipping and flagging times when the sun is above solar_horizon.
    Arguments:
        hd: HERAData object, instantiated with the datafile or files to calibrate. Must be loaded using uvh5.
            Assumed to have no prior flags.
        nInt_to_load: number of integrations to load and calibrate simultaneously. Default None loads all integrations.
            Partial io requires 'uvh5' filetype for hd. Lower numbers save memory, but incur a CPU overhead.
        pol_mode: polarization mode of redundancies. Can be '1pol', '2pol', '4pol', or '4pol_minV'.
            See recal.get_reds for more information.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        ex_ants: list of antennas to exclude from calibration and flag. Can be either antenna numbers or
            antenna-polarization tuples. In the former case, all pols for an antenna will be excluded.
        solar_horizon: float, Solar altitude flagging threshold [degrees]. When the Sun is above
            this altitude, calibration is skipped and the integrations are flagged.
        flag_nchan_low: integer number of channels at the low frequency end of the band to always flag (default 0)
        flag_nchan_high: integer number of channels at the high frequency end of the band to always flag (default 0)
        fc_conv_crit: maximum allowed changed in firstcal phases for convergence
        fc_maxiter: maximum number of firstcal iterations allowed for finding per-antenna phases
        oc_conv_crit: maximum allowed relative change in omnical solutions for convergence
        oc_maxiter: maximum number of omnical iterations allowed before it gives up
        check_every: compute omnical convergence every Nth iteration (saves computation).
        check_after: start computing omnical convergence only after N iterations (saves computation).
        gain: The fractional step made toward the new solution each omnical iteration. Values in the
            range 0.1 to 0.5 are generally safe. Increasing values trade speed for stability.
        max_dims: maximum allowed generalized tip/tilt phase degeneracies of redcal that are fixed
            with remove_degen() and must be later abscaled. None is no limit. 2 is a classically
            "redundantly calibratable" planar array.  More than 2 usually arises with subarrays of
            redundant baselines. Antennas will be excluded from reds to satisfy this.
        verbose: print calibration progress updates
        filter_reds_kwargs: additional filters for the redundancies (see redcal.filter_reds for documentation)
    Returns a dictionary of results with the following keywords:
        'g_firstcal': firstcal gains in dictionary keyed by ant-pol tuples like (1,'Jnn').
            Gains are Ntimes x Nfreqs gains but fully described by a per-antenna delay.
        'gf_firstcal': firstcal gain flags in the same format as 'g_firstcal'. Will be all False.
        'g_omnical': full omnical gain dictionary (which include firstcal gains) in the same format.
            Flagged gains will be 1.0s.
        'gf_omnical': omnical flag dictionary in the same format. Flags arise from NaNs in log/omnical.
        'v_omnical': omnical visibility solutions dictionary with baseline-pol tuple keys that are the
            first elements in each of the sub-lists of reds. Flagged visibilities will be 0.0s.
        'vf_omnical': omnical visibility flag dictionary in the same format. Flags arise from NaNs.
        'vns_omnical': omnical visibility nsample dictionary that counts the number of unflagged redundancies.
        'chisq': chi^2 per degree of freedom for the omnical solution. Normalized using noise derived
            from autocorrelations. If the inferred pol_mode from reds (see redcal.parse_pol_mode) is
            '1pol' or '2pol', this is a dictionary mapping antenna polarization (e.g. 'Jnn') to chi^2.
            Otherwise, there is a single chisq (because polarizations mix) and this is a numpy array.
        'chisq_per_ant': dictionary mapping ant-pol tuples like (1,'Jnn') to the average chisq
            for all visibilities that an antenna participates in.
        'fc_meta' : dictionary that includes delays and identifies flipped antennas
        'omni_meta': dictionary of information about the omnical convergence and chi^2 of the solution
    '''
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

#    get reds and then intitialize omnical visibility solutions to all 1s and all flagged

    all_reds = customized_groups
            
#    all_reds = get_custom_reds(hd,1)
    

    
    rv['v_omnical'] = DataContainer({red[0]: np.ones((nTimes, nFreqs), dtype=np.complex64) for red in all_reds})
    rv['vf_omnical'] = DataContainer({red[0]: np.ones((nTimes, nFreqs), dtype=bool) for red in all_reds})
    rv['vns_omnical'] = DataContainer({red[0]: np.zeros((nTimes, nFreqs), dtype=np.float32) for red in all_reds})
    filtered_reds = filter_reds(all_reds, ex_ants=ex_ants, antpos=hd.antpos,min_bl_cut=min_bl_cut,max_bl_cut=max_bl_cut, **filter_reds_kwargs)

    
    # setup metadata dictionaries
    rv['fc_meta'] = {'dlys': {ant: np.full(nTimes, np.nan) for ant in ants}}
    rv['fc_meta']['polarity_flips'] = {ant: np.full(nTimes, np.nan) for ant in ants}
    rv['omni_meta'] = {'chisq': {str(pols): np.zeros((nTimes, nFreqs), dtype=float) for pols in pol_load_list}}
    rv['omni_meta']['iter'] = {str(pols): np.zeros((nTimes, nFreqs), dtype=int) for pols in pol_load_list}
    rv['omni_meta']['conv_crit'] = {str(pols): np.zeros((nTimes, nFreqs), dtype=float) for pols in pol_load_list}

    # solar flagging
    lat, lon, alt = hd.telescope_location_lat_lon_alt_degrees
    solar_alts = utils.get_sun_alt(hd.times, latitude=lat, longitude=lon)
    solar_flagged = solar_alts > solar_horizon
    if verbose and np.any(solar_flagged):
        print(len(hd.times[solar_flagged]), 'integrations flagged due to sun above', solar_horizon, 'degrees.')

    # loop over polarizations and times, performing partial loading if desired
    for pols in pol_load_list:
        if verbose:
            print('Now calibrating', pols, 'polarization(s)...')
        reds = filter_reds(filtered_reds, ex_ants=ex_ants, pols=pols)
        if nInt_to_load is not None:  # split up the integrations to load nInt_to_load at a time
            tind_groups = np.split(np.arange(nTimes)[~solar_flagged],
                                   np.arange(nInt_to_load, len(hd.times[~solar_flagged]), nInt_to_load))
        else:
            tind_groups = [np.arange(nTimes)[~solar_flagged]]  # just load a single group
        for tinds in tind_groups:
            if len(tinds) > 0:
                if verbose:
                    print('    Now calibrating times', hd.times[tinds[0]], 'through', hd.times[tinds[-1]], '...')
                if nInt_to_load is None:  # don't perform partial I/O
                    data, _, nsamples = hd.build_datacontainers()  # this may contain unused polarizations, but that's OK
                    for bl in data:
                        data[bl] = data[bl][tinds, fSlice]  # cut down size of DataContainers to match unflagged indices
                        nsamples[bl] = nsamples[bl][tinds, fSlice]
                else:  # perform partial i/o
                    data, _, nsamples = hd.read(times=hd.times[tinds], frequencies=hd.freqs[fSlice], polarizations=pols)
                cal = redundantly_calibrate(data, reds, freqs=hd.freqs[fSlice], times_by_bl=hd.times_by_bl,
                                            fc_conv_crit=fc_conv_crit, fc_maxiter=fc_maxiter,
                                            oc_conv_crit=oc_conv_crit, oc_maxiter=oc_maxiter,
                                            check_every=check_every, check_after=check_after, max_dims=max_dims, gain=gain)
                expand_omni_sol(cal, filter_reds(all_reds, pols=pols), data, nsamples)

                # gather results
                for ant in cal['g_omnical'].keys():
                    rv['g_firstcal'][ant][tinds, fSlice] = cal['g_firstcal'][ant]
                    rv['gf_firstcal'][ant][tinds, fSlice] = cal['gf_firstcal'][ant]
                    rv['g_omnical'][ant][tinds, fSlice] = cal['g_omnical'][ant]
                    rv['gf_omnical'][ant][tinds, fSlice] = cal['gf_omnical'][ant]
                    rv['chisq_per_ant'][ant][tinds, fSlice] = cal['chisq_per_ant'][ant]
                for ant in cal['fc_meta']['dlys'].keys():
                    rv['fc_meta']['dlys'][ant][tinds] = cal['fc_meta']['dlys'][ant]
                    rv['fc_meta']['polarity_flips'][ant][tinds] = cal['fc_meta']['polarity_flips'][ant]
                for bl in cal['v_omnical'].keys():
                    rv['v_omnical'][bl][tinds, fSlice] = cal['v_omnical'][bl]
                    rv['vf_omnical'][bl][tinds, fSlice] = cal['vf_omnical'][bl]
                    rv['vns_omnical'][bl][tinds, fSlice] = cal['vns_omnical'][bl]
                if pol_mode in ['1pol', '2pol']:
                    for antpol in cal['chisq'].keys():
                        rv['chisq'][antpol][tinds, fSlice] = cal['chisq'][antpol]
                else:  # duplicate chi^2 into both antenna polarizations
                    for antpol in rv['chisq'].keys():
                        rv['chisq'][antpol][tinds, fSlice] = cal['chisq']
                rv['omni_meta']['chisq'][str(pols)][tinds, fSlice] = cal['omni_meta']['chisq']
                rv['omni_meta']['iter'][str(pols)][tinds, fSlice] = cal['omni_meta']['iter']
                rv['omni_meta']['conv_crit'][str(pols)][tinds, fSlice] = cal['omni_meta']['conv_crit']
    
    print('redcal_iteration_custom2 Complete')
    return rv



def redcal_run_custom(input_data,c_data,k_value,N_red_clusters,min_bl_cut, max_bl_cut, filetype='uvh5', firstcal_ext='.first.calfits', omnical_ext='.omni.calfits',
               omnivis_ext='.omni_vis.uvh5', meta_ext='.redcal_meta.hdf5', iter0_prefix='', outdir=None,
               metrics_files=[], a_priori_ex_ants_yaml=None, clobber=False, nInt_to_load=None, pol_mode='2pol',
               bl_error_tol=1.0, ex_ants=[], ant_z_thresh=4.0, max_rerun=5, solar_horizon=0.0,
               flag_nchan_low=0, flag_nchan_high=0, fc_conv_crit=1e-6, fc_maxiter=50,
               oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50, gain=.4, add_to_history='',
               max_dims=2, verbose=False, **filter_reds_kwargs):
    '''Perform redundant calibration (firstcal, logcal, and omnical) an uvh5 data file, saving firstcal and omnical
    results to calfits and uvh5. Uses partial io if desired, performs solar flagging, and iteratively removes antennas
    with high chi^2, rerunning calibration as necessary.
    Arguments:
        input_data: path to visibility data file to calibrate or HERAData object
        filetype: filetype of input_data (if it's a path). Supports 'uvh5' (defualt), 'miriad', 'uvfits'
        firstcal_ext: string to replace file extension of input_data for saving firstcal calfits
        omnical_ext: string to replace file extension of input_data for saving omnical calfits
        omnivis_ext: string to replace file extension of input_data for saving omnical visibilities as uvh5
        meta_ext: string to replace file extension of input_data for saving metadata as hdf5
        iter0_prefix: if not '', save the omnical results with this prefix appended to each file after the 0th
            iteration, but only if redcal has found any antennas to exclude and re-run without
        outdir: folder to save data products. If None, will be the same as the folder containing input_data
        metrics_files: path or list of paths to file(s) containing ant_metrics or auto_metrics readable by 
            hera_qm.metrics_io.load_metric_file. Used for finding ex_ants and is combined with antennas
            excluded via ex_ants.
        a_priori_ex_ants_yaml : path to YAML with antenna flagging information parsable by
            hera_qm.metrics_io.read_a_priori_ant_flags(). Frequency and time flags in the YAML
            are ignored. Flags are combined with ant_metrics's xants and ex_ants. If any
            polarization is flagged for an antenna, all polarizations are flagged.
        clobber: if True, overwrites existing files for the firstcal and omnical results
        nInt_to_load: number of integrations to load and calibrate simultaneously. Default None loads all integrations.
            Partial io requires 'uvh5' filetype. Lower numbers save memory, but incur a CPU overhead.
        pol_mode: polarization mode of redundancies. Can be '1pol', '2pol', '4pol', or '4pol_minV'.
            See recal.get_reds for more information.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        ex_ants: list of antennas to exclude from calibration and flag. Can be either antenna numbers or
            antenna-polarization tuples. In the former case, all pols for an antenna will be excluded.
        ant_z_thresh: threshold of modified z-score (like number of sigmas but with medians) for chi^2 per
            antenna above which antennas are thrown away and calibration is re-run iteratively. Z-scores are
            computed independently for each antenna polarization, but either polarization being excluded
            triggers the entire antenna to get flagged (when multiple polarizations are calibrated)
        max_rerun: maximum number of times to run redundant calibration
        solar_horizon: float, Solar altitude flagging threshold [degrees]. When the Sun is above
            this altitude, calibration is skipped and the integrations are flagged.
        flag_nchan_low: integer number of channels at the low frequency end of the band to always flag (default 0)
        flag_nchan_high: integer number of channels at the high frequency end of the band to always flag (default 0)
        fc_conv_crit: maximum allowed changed in firstcal phases for convergence
        fc_maxiter: maximum number of firstcal iterations allowed for finding per-antenna phases
        oc_conv_crit: maximum allowed relative change in omnical solutions for convergence
        oc_maxiter: maximum number of omnical iterations allowed before it gives up
        check_every: compute omnical convergence every Nth iteration (saves computation).
        check_after: start computing omnical convergence only after N iterations (saves computation).
        gain: The fractional step made toward the new solution each omnical iteration. Values in the
            range 0.1 to 0.5 are generally safe. Increasing values trade speed for stability.
        max_dims: maximum allowed generalized tip/tilt phase degeneracies of redcal that are fixed
            with remove_degen() and must be later abscaled. None is no limit. 2 is a classically
            "redundantly calibratable" planar array.  More than 2 usually arises with subarrays of
            redundant baselines. Antennas will be excluded from reds to satisfy this.
        add_to_history: string to add to history of output firstcal and omnical files
        verbose: print calibration progress updates
        filter_reds_kwargs: additional filters for the redundancies (see redcal.filter_reds for documentation)
    Returns:
        cal: the dictionary result of the final run of redcal_iteration (see above for details)
    '''
    if isinstance(input_data, str):
        hd = HERAData(input_data, filetype=filetype)
        if filetype != 'uvh5' or nInt_to_load is None:
            hd.read()

    elif isinstance(input_data, HERAData):
        hd = input_data
        input_data = hd.filepaths[0]
    else:
        raise TypeError('input_data must be a single string path to a visibility data file or a HERAData object')

    # parse ex_ants from function, metrics_files, and apriori yamls
    ex_ants = set(ex_ants)
    if metrics_files is not None:
        if isinstance(metrics_files, str):
            metrics_files = [metrics_files]
        if len(metrics_files) > 0:
            from hera_qm.metrics_io import load_metric_file
            for mf in metrics_files:
                metrics = load_metric_file(mf)
                # load from an ant_metrics file
                if 'xants' in metrics:
                    for ant in metrics['xants']:
                        ex_ants.add(ant[0])  # Just take the antenna number, flagging both polarizations
                # load from an auto_metrics file
                elif 'ex_ants' in metrics and 'r2_ex_ants' in metrics['ex_ants']:
                    for ant in metrics['ex_ants']['r2_ex_ants']:
                        ex_ants.add(ant)  # Auto metrics reports just antenna numbers
    if a_priori_ex_ants_yaml is not None:
        from hera_qm.metrics_io import read_a_priori_ant_flags
        ex_ants = ex_ants.union(set(read_a_priori_ant_flags(a_priori_ex_ants_yaml, ant_indices_only=True)))
    high_z_ant_hist = ''

    # setup output
    filename_no_ext = os.path.splitext(os.path.basename(input_data))[0]
    if outdir is None:
        outdir = os.path.dirname(input_data)

    # loop over calibration, removing bad antennas and re-running if necessary
    run_number = 0
    while True:
        # Run redundant calibration
        if verbose:
            print('\nNow running redundant calibration without antennas', list(ex_ants), '...')
            
        N_red_clusters = N_red_clusters
        k_value = k_value
        #Running Logi_Cal 
        customized_groups = custom_reds2(hd,c_data,k_value,N_red_clusters)
        custom_red_gains = redcal_iteration_custom2(hd,customized_groups,min_bl_cut,max_bl_cut)
        cal = custom_red_gains

        # Determine whether to add additional antennas to exclude
        z_scores = per_antenna_modified_z_scores({ant: np.nanmedian(cspa) for ant, cspa in cal['chisq_per_ant'].items()
                                                  if (ant[0] not in ex_ants) and not np.all(cspa == 0)})
        n_ex = len(ex_ants)
        for ant, score in z_scores.items():
            if (score >= ant_z_thresh):
                ex_ants.add(ant[0])
                bad_ant_str = 'Throwing out antenna ' + str(ant[0]) + ' for a z-score of ' + str(score) + ' on polarization ' + str(ant[1]) + '.\n'
                high_z_ant_hist += bad_ant_str
                if verbose:
                    print(bad_ant_str)
        run_number += 1
        if len(ex_ants) == n_ex or run_number >= max_rerun:
            break
        # If there is going to be a re-run and if iter0_prefix is not the empty string, then save the iter0 results.
        if run_number == 1 and len(iter0_prefix) > 0:
            _redcal_run_write_results(cal, hd, filename_no_ext + iter0_prefix + firstcal_ext, filename_no_ext + iter0_prefix + omnical_ext,
                                      filename_no_ext + iter0_prefix + omnivis_ext, filename_no_ext + iter0_prefix + meta_ext, outdir,
                                      clobber=clobber, verbose=verbose, add_to_history=add_to_history + '\n' + 'Iteration 0 Results.\n')

    # output results files
    _redcal_run_write_results(cal, hd, filename_no_ext + firstcal_ext, filename_no_ext + omnical_ext,
                              filename_no_ext + omnivis_ext, filename_no_ext + meta_ext, outdir, clobber=clobber,
                              verbose=verbose, add_to_history=add_to_history + '\n' + high_z_ant_hist)

    return cal






def fix_redcal_degeneracies(data_file, red_gains, true_gains, outfile=None, 
                            overwrite=False):
    """
    Use the true (input) gains to fix the degeneracy directions in a set of 
    redundantly-calibrated gain solutions. This replaces the absolute 
    calibration that would normally be applied to a real dataset in order to 
    fix the degeneracies.
    
    Note that this step should only be using the true gains to fix the 
    degeneracies, and shouldn't add any more information beyond that.
    
    N.B. This is just a convenience function for calling the 
    remove_degen_gains() method of the redcal.RedundantCalibrator class. It 
    also assumes that only the 'ee' polarization will be used.
    
    Parameters
    ----------
    data_file : str
        Filename of the data file (uvh5 format) that is being calibrated. This 
        is only used to extract information about redundant baseline groups.
    
    red_gains : dict of array_like
        Dict containing 2D array of complex gain solutions for each antenna 
        (and polarization).
    
    true_gains : dict
        Dictionary of true (input) gains as a function of frequency. 
        Expected format: 
            key = antenna number (int)
            value = 1D numpy array of shape (Nfreqs,)
    
    outfile : str, optional
        If specified, save the updated gains to a calfits file. Default: None.
    
    overwrite : bool, optional
        If the output file already exists, whether to overwrite it. 
        Default: False.
    
    Returns
    -------
    new_gains : dict
        Dictionary with the same items as red_gains, but where the degeneracies 
        have been fixed in the gain solutions.
    
    uvc : UVCal, optional
        If outfile is specified, also returns a UVCal object containing the 
        updated gain solutions.
    """
    # Get ntimes from gain array belonging to first key in the dict
    #ntimes = red_gains[list(red_gains.keys())[0]].shape[0]
    ntimes = 1
    # Load data file and get redundancy information
    hd = hc.io.HERAData(data_file)
    reds = hc.redcal.get_reds(hd.antpos, pols=['ee',])
    
    # Create calibrator and fix degeneracies
    RedCal = hc.redcal.RedundantCalibrator(reds)
    new_gains = RedCal.remove_degen_gains(red_gains, 
                                          degen_gains=true_gains, 
                                          mode='complex')
    
    # Save as file if requested
    if outfile is not None:
        uvc = hc.redcal.write_cal(outfile,
                                  new_gains,
                                  hd.freqs,
                                  hd.times,
                                  write_file=True,
                                  return_uvc=True,
                                  overwrite=overwrite)
        return new_gains, uvc
    else:
        # Just return the updated gain dict
        return new_gains


    
    
def get_baseline_length_angle(data_file,redundant_b_groups):
    """ 
    This function calculates the baseline length and angle for each redundant baseline group, that we get when running 
        get_reds() function.
        parameters
            index_number: is always zero, just needed an input to put on the function
    """
    
    reds_all = redundant_b_groups
    
    
    baseline_length = []
    baseline_angle = []
    for i in range(len(reds_all)):
        k = 0 ## Just the first index of the array, to get into the individual antenna pairs.
        a = reds_all[i][k][0]
        b = reds_all[i][k][1]

        x1 = data_file.antpos[a][0]
        y1 = data_file.antpos[a][1]

        x2 = data_file.antpos[b][0]
        y2 = data_file.antpos[b][1]
        
        r = np.sqrt( np.square(x1-x2)+ np.square(y1-y2) )
        baseline_length.append(r)
        
        bl_ang = np.arccos(abs(x1-x2)/abs(r)) *180/np.pi
        baseline_angle.append(bl_ang)

    return baseline_length, baseline_angle


def len_ang_dictionary(data_file,redundant_b_groups):
    """ 
    This function saves the calculated baseline length and angle for each redundant baseline group into a dictionary.
        parameters 
        redundant_b_groups : The list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol), we get when running
                            get_reds()
    """
    reds_all = redundant_b_groups
        
    get_reds_array = [reds_all, get_baseline_length_angle(data_file,reds_all)[0], get_baseline_length_angle(data_file,reds_all)[1]]

    len_ang_dict = {'length':{},'angle':{}}

    index_tot=[]
    for i in range(len(reds_all)):
        bl_len = get_reds_array[1][i]
        bl_len_key = str(int(np.round( (10. * bl_len), 0)))

        index_len =[]
        if bl_len_key in len_ang_dict['length']:
            len_ang_dict['length'][bl_len_key].append(reds_all[i])

        else:
            len_ang_dict['length'][bl_len_key]=[]
            len_ang_dict['length'][bl_len_key].append(reds_all[i])

        bl_ang = get_reds_array[2][i]
        bl_ang_key = str(int(np.round( (10. * bl_ang), 0)))

        if bl_ang_key in len_ang_dict['angle']:
            len_ang_dict['angle'][bl_ang_key].append(reds_all[i])
        else:
            len_ang_dict['angle'][bl_ang_key]=[]
            len_ang_dict['angle'][bl_ang_key].append(reds_all[i])
            
    return len_ang_dict





##########################################################################################################################################################################################################################################################################################
#############################################################################################################################################

              ###########              #####            #####         ###   #########            #########      #####                 #####
              ############            #######           ### ##        ###   ##########          ###########     ### ##               ## ###
              ####      ###          ###   ###          ###  ##       ###   ###      ###       ###       ###    ###  ##             ##  ###
              ####      ###         ###     ###         ###   ##      ###   ###       ###     ###         ###   ###   ##           ##   ###
              ###########          #############        ###    ##     ###   ###        ###   ###           ###  ###    ##         ##    ###
              ############        ###############       ###     ##    ###   ###        ###   ###           ###  ###     ##       ##     ###
              ####      ###      ###           ###      ###      ##   ###   ###       ###     ###          ###  ###      ##     ##      ###
              ####      ###     ###             ###     ###       ##  ###   ###      ###       ###        ###   ###       ##   ##       ###
              ####      ###    ###               ###    ###        ## ###   ##########          ############    ###        ## ##        ###
              ####      ###   ###                 ###   ###         #####   #########            ##########     ###         ###         ###  

                ######################################################################################################################################################################################################################################################################################################################################################################################################################################



def random_cluster_baselines2(data_file, Number_of_clusters, red_groups_index):
    """ Using a random function to classify baselines from a redundant_baseline_group into groups based on their visibilities.
    Returns:
        n : A list of baseline IDs for the baselnes in the redundant baseline group
        true_labels: A list of lables from the clustering algorithm where baselines of similar
                    visibilities are clustered into the same group. P.S. the number of groups we get
                    depend on the number of clusters specified.
    """
    reds_all = all_reds_function(data_file)
    
    ## Changing the antenna numbers in to baselines for every redundant baseline group.
    redundant_baseline_group = []
    for bg in range(len(reds_all)):
        redundant_baselines = []
        for a_num in range(len(reds_all[bg])):
            bID = data_file.antnums_to_baseline(reds_all[bg][a_num][0],reds_all[bg][a_num][1])
            redundant_baselines.append(bID)
        redundant_baseline_group.append(redundant_baselines) ## Saving a list of redundant baselines in an array.
    
    
    n = redundant_baseline_group[red_groups_index]   ## Getting one redundant baseline group
    
    
    ## chooses a random number from the k_value_list, where that number will be stored in a list corresponding to the index of every baseline
    ## in a redundant baseline group.
    k_value_list = np.arange(Number_of_clusters)
    
    random_labels = []
    for i in range(len(n)):
        new_list = random.choice(k_value_list) 
        random_labels.append(new_list)
    random_labels = np.array(random_labels)
  
    return n, random_labels



def random_get_baseline_cluster2(data_file, Number_of_clusters,red_groups_index):
    """ It uses the labels we get when we run the random function, to get the index of each label
            in order to find out which label belongs to which baseline. 
    Returns:
        clusters[0] : A 2D list of Baseline ID that are grouped in the way the random function clustered them. The
            result is in the form, len(2D_list) = Number_of_clusters.
        k_value[1]: k values used in clustering the redundant baseline group.
    """
    ## Calling the cluster_baselines function to get the labels
    labels = random_cluster_baselines2(data_file, Number_of_clusters,red_groups_index)[1]
    
    ## Getting rid of repeated labels
    p = list(dict.fromkeys(labels))
    
    ## Getting the index of each label in order to find out which label belongs to which baseline
    x = labels
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    
    clusters = []
    for j in range(len(p)):
        cluster1 = get_indexes(p[j],x) ## a list of indeces for each label
        baseline_array = np.array(random_cluster_baselines2(data_file, Number_of_clusters,red_groups_index)[0])
        base_ant_idx = baseline_array[cluster1] ## index[0]=n 
        clusters.append(base_ant_idx)
    

    return clusters, len(p)


def random_get_custom_reds2(data_file, Number_of_clusters,red_groups_index):
    
    """ Combines cluster_baselines() and get_baseline_cluster().

    Returns:
        reds[0]: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each list has a list of baselines that are clustered into the same group by the 
            random function.
        k_value_list[1]: list of k values that each redundant baseline group has been clustered into.
    """
    list_array = random_get_baseline_cluster2(data_file,Number_of_clusters,red_groups_index)
    
    antpp_c = list_array[0]


    reds_all_cluster = []
    for j in range(len(antpp_c)):
        ant_pair_cluster1 =[]
        for i in antpp_c[j]:
            #Print baselines with the antenna numbers that makeup that baseline
            aa =  np.int64(data_file.baseline_to_antnums(i)[0]), np.int64(data_file.baseline_to_antnums(i)[1]), 'ee'
            ant_pair_cluster1.append(aa)
        reds_all_cluster.append(ant_pair_cluster1)
        

    return reds_all_cluster , list_array[1]



def random_custom_reds2(hd, data_file, Number_of_clusters, red_groups_index, nInt_to_load=None, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[],
                     solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0, fc_conv_crit=1e-6,
                     fc_maxiter=50, oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50,
                     gain=.4, max_dims=2, verbose=False, **filter_reds_kwargs):
    
        
    """ Combines the clustered baselines groups with the original get_reds groups and also changes their polarisation
        of the groups from 'nn' to 'ee'.

    Returns:
        reds[0]: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each list has a list of baselines that are clustered into the same group by the 
            random function.
        k_value_list[1]: list of k values that each redundant baseline group has been clustered into.
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

#    get reds and then intitialize omnical visibility solutions to all 1s and all flagged
    rd = get_reds({ant: hd.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                        pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    
    k_value_list = []
    clustered_baseline_groups = []
    for z in range(red_groups_index):
        list_array = random_get_custom_reds2(data_file,Number_of_clusters,z)
        clustered_baseline_groups.append(list_array[0])
        k_value_list.append(list_array[1])

    ## Replaces the original redundand groups by its clustered subgroups.
    all_reds = []
    #appends groups that are clustered
    for k1 in range(len(clustered_baseline_groups)):
        for k2 in range(len(clustered_baseline_groups[k1])):
            all_reds.append(clustered_baseline_groups[k1][k2])
    
    # Appends the un-clustered groups
    for k3 in range(len(clustered_baseline_groups),len(rd)):
        all_reds.append(rd[k3])
    

    return all_reds , k_value_list





######################################################################################################################################################################################################################################################################################################################################################################################################################################
######################################################   CALCULATING CHI SQUARED    #########################################################





def calulate_X2_all(true_gains,red_gains_fixed, custom_red_gains_fixed ):
    """ Takes in the gains from calibration to calculate the chi_sq-like statistic. The statistic is calculated with
        [(gain.real - gain_true.real)^2 + (gain.imag - gain_true.imag)^2] and without summing over all times and 
        frequencies, therefore each antenna will have multiple values for every time and frequency.
    parameters:
            true_gains: Expected gains (from simulations with no added noise and gains)
            redcal_gains_fixed: gains after running redcal with their degeneracies fixed.
            custom_red_gains_fixed: gains after running logi_cal with their degeneracies fixed.
    Returns:
        redcal_X^2: X^2 values between redcal gains and the expected(true) gains
        logi_cal_X^2: X^2 values between logi_cal gains and the expected(true) gains
    """
    X_custom_true_real , X_custom_true_imag = [], []
    X_red_true_real , X_red_true_imag = [], []


    for i in range(len(custom_red_gains_fixed)):
        a1 = np.real(true_gains[( int('{}'.format(i)) , 'Jee' )] )  ## true gains
        b1 = np.real(custom_red_gains_fixed[( int('{}'.format(i)) , 'Jee' )] )  ## Modified Redcal code
        c1 = np.real(red_gains_fixed[( int('{}'.format(i)) , 'Jee' )] )  ## original Redcal code

        a2 = np.imag(true_gains[( int('{}'.format(i)) , 'Jee' )] )  ## true gains
        b2 = np.imag(custom_red_gains_fixed[( int('{}'.format(i)) , 'Jee' )] )  ## Modified Redcal code
        c2 = np.imag(red_gains_fixed[( int('{}'.format(i)) , 'Jee' )] )  ## original Redcal code


        X1_custom_true_real = (np.square(a1-b1) )
        X1_red_true_real = (np.square(a1-c1) )

        X1_custom_true_imag = (np.square(a2-b2) )
        X1_red_true_imag = (np.square(a2-c2) )

        X_custom_true_real.append((X1_custom_true_real))   ### Chi squared-like statistic between gains from Modified Redcal code and True gains
        X_red_true_real.append((X1_red_true_real))   

        X_custom_true_imag.append((X1_custom_true_imag))   ### Chi squared-like statistic between gains from Modified Redcal code and True gains
        X_red_true_imag.append((X1_red_true_imag))     ### Chi squared-like statistic between gains from original Redcal code and True gains

    X_custom_true_tot = np.array(X_custom_true_real)+np.array(X_custom_true_imag)
    X_red_true_tot = np.array(X_red_true_real) + np.array(X_red_true_imag)

    return X_red_true_tot, X_custom_true_tot


#############################################################################################################################################
def calulate_X2_sum(true_gains,red_gains_fixed, custom_red_gains_fixed ):
    """ Takes in the gains from calibration to calculate the chi_sq-like statistic. The statistic takes the
        Sum [(gain.real - gain_true.real)^2 + (gain.imag - gain_true.imag)^2] and summing over all times and 
        frequencies, therefore each antenna will have only one value.
    parameters:
            true_gains: Expected gains (from simulations with no added noise and gains)
            redcal_gains_fixed: gains after running redcal with their degeneracies fixed.
            custom_red_gains_fixed: gains after running logi_cal with their degeneracies fixed.
    Returns:
        redcal_X^2: X^2 values between redcal gains and the expected(true) gains
        logi_cal_X^2: X^2 values between logi_cal gains and the expected(true) gains
    """
    X_custom_true_real , X_custom_true_imag = [], []
    X_red_true_real , X_red_true_imag = [], []


    for i in range(len(custom_red_gains_fixed)):
        a1 = np.real(true_gains[( int('{}'.format(i)) , 'Jee' )] )  ## true gains
        b1 = np.real(custom_red_gains_fixed[( int('{}'.format(i)) , 'Jee' )] )  ## Modified Redcal code
        c1 = np.real(red_gains_fixed[( int('{}'.format(i)) , 'Jee' )] )  ## original Redcal code

        a2 = np.imag(true_gains[( int('{}'.format(i)) , 'Jee' )] )  ## true gains
        b2 = np.imag(custom_red_gains_fixed[( int('{}'.format(i)) , 'Jee' )] )  ## Modified Redcal code
        c2 = np.imag(red_gains_fixed[( int('{}'.format(i)) , 'Jee' )] )  ## original Redcal code


        X1_custom_true_real = np.sum(np.square(a1-b1) )
        X1_red_true_real = np.sum(np.square(a1-c1) )

        X1_custom_true_imag = np.sum(np.square(a2-b2) )
        X1_red_true_imag = np.sum(np.square(a2-c2) )

        X_custom_true_real.append((X1_custom_true_real))   ### Chi squared-like statistic between gains from Modified Redcal code and True gains
        X_red_true_real.append((X1_red_true_real))   

        X_custom_true_imag.append((X1_custom_true_imag))   ### Chi squared-like statistic between gains from Modified Redcal code and True gains
        X_red_true_imag.append((X1_red_true_imag))     ### Chi squared-like statistic between gains from original Redcal code and True gains

    X_custom_true_tot = np.array(X_custom_true_real)+np.array(X_custom_true_imag)
    X_red_true_tot = np.array(X_red_true_real) + np.array(X_red_true_imag)

    return X_red_true_tot, X_custom_true_tot




def get_statistics_visibility(true_file, calib_data,rbg_index):
    """
        Use statistical methods to compare similarities in visibility solutions between the what is known (true data), Redcal (the original
        redundant calibration code) and Logi_cal (the modified redcal). 

        Parameters
        ----------
        true_file: HERAData object, data file containing visibility real or true solutions. Must be loaded using uvh5.
            Assumed to have no prior flags..

        calib_file : hera_cal.io.HERAData container already opened using UVcal. For more information on how to open the file refer
            to 'Save_Load_data.py'.

        rbg_index : int
            Specify the redundant baseline group needed to calculate the statistics from.
                max_value = len(red_base[0])
                
        Statistical Methods
        ----------- -------
        dh: The directed Hausdorff distance
        df: Discrete Fréchet distance
        dtw: Dynamic Time Warping (DTW)
        pcm: Partial Curve Mapping (PCM)
        area: Area between two curves 
        cl: A Curve-Length distance metric (uses arc length distance from beginning to end)

        Returns
        -------
        An array of statistical values of each method for each baseline in a redundant baseline group.
            NB.The closer to ZERO these values are, the more similar the two data files are in comparison.
        """
    
    red_base = true_file.get_redundancies(tol=1.0, use_antpos=False, include_conjugates=False,include_autos=True,
                               conjugate_bls=False)
    rbg = red_base[0][rbg_index]  
    rbg = np.array(rbg)     ## Saving a list of redundant baselines in an array.
    
    stat_values = []
    #original_stat = []
    for i in range(len(rbg)):
        y1 = np.abs(calib_data.get_data(rbg[i])[0] )
        y2 = np.abs(true_file.get_data(rbg[i])[0] )
        x = np.arange(len(y1))

        P = np.array([x, y1]).T
        Q = np.array([x, y2]).T

        dh, ind1, ind2 = directed_hausdorff(P, Q)
        df = similaritymeasures.frechet_dist(P, Q)
        dtw, d = similaritymeasures.dtw(P, Q)
        pcm = similaritymeasures.pcm(P, Q)
        area = similaritymeasures.area_between_two_curves(P, Q)
        cl = similaritymeasures.curve_length_measure(P, Q)

        # all methods will return 0.0 when P and Q are the same
        stat_values.append([dh, df, dtw, pcm, cl, area])

        #print('{} = '.format( true_data.baseline_to_antnums(rbg[i])), dh, df, dtw, pcm, cl, area)
    stat_values = np.array(stat_values)     
    return stat_values




def get_statistics_gains(true_file, calib_data,time_sample):
    """
        Use statistical methods to compare similarities in visibility solutions between the what is known (true data), Redcal (the original
        redundant calibration code) and Logi_cal (the modified redcal). 

        Parameters
        ----------
        true_file: true gains

        calib_file : calibrated gains with fixed degeneracies (from RedCal or Logi_Cal)

        time_sample : int
            Specify the time sample needed to calculate the statistics from.
                max_value = len(hd.ntimes)
                
        Statistical Methods
        ----------- -------
        dh: The directed Hausdorff distance
        df: Discrete Fréchet distance
        dtw: Dynamic Time Warping (DTW)
        pcm: Partial Curve Mapping (PCM)
        area: Area between two curves 
        cl: A Curve-Length distance metric (uses arc length distance from beginning to end)

        Returns
        -------
        An array of statistical values of each method for each antenna.
            NB.The closer to ZERO these values are, the more similar the two data files are in comparison.
        """
    
    
    stat_values = []
    #original_stat = []
    for i in range(len(true_file)):
        y1 = np.abs(calib_data[( int('{}'.format(i)) , 'Jee' )][time_sample] )
        y2 = np.abs(true_file[( int('{}'.format(i)) , 'Jee' )][time_sample] )
        x = np.arange(len(y1))

        P = np.array([x, y1]).T
        Q = np.array([x, y2]).T

        dh, ind1, ind2 = directed_hausdorff(P, Q)
        df = similaritymeasures.frechet_dist(P, Q)
        dtw, d = similaritymeasures.dtw(P, Q)
        pcm = similaritymeasures.pcm(P, Q)
        area = similaritymeasures.area_between_two_curves(P, Q)
        cl = similaritymeasures.curve_length_measure(P, Q)

        # all methods will return 0.0 when P and Q are the same
        stat_values.append([dh, df, dtw, pcm, cl, area])

        #print('{} = '.format( true_data.baseline_to_antnums(rbg[i])), dh, df, dtw, pcm, cl, area)
    stat_values = np.array(stat_values)   
    return stat_values





##################################################################################################################################################################################################     PLOTS   ###################################################3

def bar_plot_124(data_array, plot_tittle):

    cal_vulues = data_array.T

    # Numbers of pairs of bars you want
    N = len(data_array)

    # Data on X-axis

    legend_label = ['2','3','4','5','6']
    labels = ['15', '25', '30', '35', '40','50','60']

    # Position of bars on x-axis
    ind = np.arange(N)
    
    # Width of a bar 
    width = 0.18

    # Plotting
    # Figure size
    plt.figure(figsize=(14,8))

    for i in range(len(cal_vulues)):
        value = cal_vulues[i]
        plt.barh(ind + (i+0.9)*width, value , width, label= legend_label[i], alpha = 0.8 )

    plt.xlabel('Percentage change')
    plt.title('Percentage change per Non_red Case ({})'.format(plot_tittle))
    plt.ylabel("Number of redundant baseline groups")
    plt.axvline(color='k', alpha=0.5)
    plt.hlines([1.03,2.03,3.03,4.03,5.03,6.03],cal_vulues.min() , cal_vulues.max(), color='grey',linestyles='--', linewidth=1)

    x_ticks = labels
    plt.yticks(ind + width+0.25, x_ticks)

    # Finding the best position for legends and putting it
    plt.legend(loc='best',title="K_values")
    plt.show()




def bar_plot_124_annotate(data_array, plot_tittle):

    cal_vulues = data_array.T

    # Numbers of pairs of bars you want
    N = len(data_array)

    # Data on X-axis

    legend_label = ['2','3','4','5','6']
    labels = ['15', '25', '30', '35', '40','50','60']

    # Position of bars on x-axis
    ind = np.arange(N)
    
    # Width of a bar 
    width = 0.18

    # Plotting
    # Figure size
    plt.figure(figsize=(14,8))

    rects_list = []
    for i in range(len(cal_vulues)):
        value = cal_vulues[i]
        ax = plt.barh(ind + (i+0.9)*width, value , width, label= legend_label[i], alpha = 0.8 )
        rects_list.append(ax.patches)
    plt.xlabel('Percentage change')
    plt.title('Percentage change per Non_red Case ({})'.format(plot_tittle))
    plt.ylabel("Number of redundant baseline groups")
    plt.axvline(color='k', alpha=0.5)
    plt.hlines([1.03,2.03,3.03,4.03,5.03,6.03],cal_vulues.min() , cal_vulues.max(), color='grey',linestyles='--', linewidth=1)

    x_ticks = labels
    plt.yticks(ind + width+0.25, x_ticks)

    # Finding the best position for legends and putting it
    plt.legend(loc='best',title="K_values")

    # For each bar: Place a label
    for k in range(len(rects_list)):
        for rect in rects_list[k]:
            # Get X and Y placement of label from rect.
            x_value = rect.get_width()
            y_value = rect.get_y() + rect.get_height() / 2

            # Number of points between bar and label. Change to your liking.
            space = 5
            # Vertical alignment for positive values
            ha = 'left'

            # If value of bar is negative: Place label left of bar
            if x_value < 0:
                # Invert space to place label to the left
                space *= -1
                # Horizontally align label at right
                ha = 'right'

            # Use X value as label and format number with one decimal place
            label = "{:.1f}%".format(x_value)

            # Create annotation
            plt.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(space, 0),          # Horizontally shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                va='center',                # Vertically center label
                ha=ha)                      # Horizontally align label differently for
                                            # positive and negative values.
    plt.show()






