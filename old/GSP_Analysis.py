#
# Graph Signal Processing of OCD baseline dataset
#
# Eigen-decomposition of SC used for FC graph filtering
#
# Author: Sebastien Naze
# -----------------------------------------------------------------------
import argparse
import bct
import h5py
import itertools
import importlib
import joblib
from joblib import Parallel, delayed
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import nilearn
from nilearn.image import load_img
from nilearn.plotting import plot_matrix, plot_glass_brain
import numpy as np
import os
import pickle
import pandas as pd
import pygsp
from pygsp import graphs, filters, plotting
import scipy
from scipy.io import loadmat
import statsmodels
from statsmodels.stats import multitest
import sys

proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'

#sys.path.insert(0, os.path.join(code_dir))
from ..old import qsiprep_analysis
from ..utils import ttest_ind_FWE
from ..utils import atlaser

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)
subjs = pd.read_table(os.path.join(code_dir, 'subject_list.txt'), names=['name'])['name']



def get_bold_ts(atlas, fc_metric, subj):
    tsfname = subj+'_task-rest_atlas-'+atlas+'_desc-'+fc_metric+'_scrub.h5'
    hf = h5py.File(os.path.join(deriv_dir, 'post-fmriprep-fix', subj, 'timeseries', tsfname), 'r')
    ts = list(hf.items())
    (at, ts) = ts[0]
    return np.array(ts)

def get_bold_dict(atlases, fc_metrics, subjs):
    bold_ts = dict( ((at,fcm,coh),[]) \
        for at, fcm, coh in itertools.product(atlases,fc_metrics,['controls','patients']))
    for at, fcm, subj in itertools.product(atlases,fc_metrics,subjs):
        if 'control' in subj:
            bold_ts[at,fcm,'controls'].append(get_bold_ts(at,fcm,subj))
        elif 'patient' in subj:
            bold_ts[at,fcm,'patients'].append(get_bold_ts(at,fcm,subj))
    return bold_ts


def get_subnet_indices(atlas):
    """ returns dictionary of indices corresponding to each subnetwork (indices are 'atlas indices')"""
    sub_inds = {'wholebrain':np.arange(len(qsiprep_analysis.atlas_cfg[atlas]['node_ids'])), \
            'fspt':qsiprep_analysis.get_fspt_node_ids(atlas)-1, \
            'Fr':qsiprep_analysis.get_fspt_Fr_node_ids(atlas)[0]-1, \
            'StrTh':qsiprep_analysis.get_fspt_Fr_node_ids(atlas, \
                        subctx=['PFC', 'OFC', 'Fr', 'FEF', 'ACC', 'Cing', 'PrC', 'ParMed'])[0]-1}
    return sub_inds


def print_significant_stats(subnet,atlas,scm,fcm,outp,gfreqs):
    """ Print name of ROIs with significant p-values with and without correction for multiple comparisons """
    atlazer = atlaser.Atlaser(atlas)
    s_inds, = np.where(outp[subnet,atlas,scm,fcm]['p_'+gfreq] <= 0.01)
    s_node_ids = get_subnet_indices(atlas)[subnet]+1  #atlazer.node_ids[s_inds]
    s_node_ids = s_node_ids[s_inds]
    s_node_names = atlazer.get_roi_names(s_node_ids)
    for i,ind in enumerate(s_inds):
        print('{} graph {} freq \t {:30} \t\t p={:.3f} \t t={:.3f} \t p_fdr={:.3f}'.format( \
                subnet, gfreq, s_node_names[i], \
                outp[subnet,atlas,scm,fcm]['p_'+gfreq][ind], \
                outp[subnet,atlas,scm,fcm]['t_'+gfreq][ind], \
                outp[subnet,atlas,scm,fcm]['p_corrected']['fdr_bh',gfreq][ind]))


def plot_stats_on_glass_brain(subnet, atlas, scm, fcm, outp, gfreq='low'):
    """ Display color-coded T stats and p-values on glass brain """
    # create nifti imgs
    atlazer = atlaser.Atlaser(atlas)
    node_ids = get_subnet_indices(atlas)[subnet]+1
    t_img = atlazer.create_brain_map(node_ids, outp[subnet,atlas,scm,fcm]['t_'+gfreq])
    p_img = atlazer.create_brain_map(node_ids, 1-outp[subnet,atlas,scm,fcm]['p_'+gfreq])

    # plotting
    fig = plt.figure(figsize=[30,4])
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    plt.suptitle('{} {} {} {}'.format(subnet,atlas,scm,fcm))
    display1 = plot_glass_brain(t_img, display_mode='lzry', colorbar=True, title=gfreq, plot_abs=False, vmin=-5, vmax=5, axes=ax1)
    display2 = plot_glass_brain(None, display_mode='lzry', colorbar=True, title='p_'+gfreq, axes=ax2)
    display2.add_contours(p_img, filled='green', levels=[0.99])
    plt.show(block=False)



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--lap_type', default='normalized', help="Laplacian type, 'normalized' (default) or 'combinatorial'")
    parser.add_argument('--plot_conns', default=False, action='store_true', help='plot connectivity matrices')
    parser.add_argument('--plot_eigenvectors', default=False, action='store_true', help='plot eigenvectors on glass brain (takes a lot of time!)')
    args = parser.parse_args()

    # load data
    with open(os.path.join(proj_dir, 'postprocessing', 'conns_SC.pkl'), 'rb') as scf:
        conns_sc = pickle.load(scf)
    with open(os.path.join(proj_dir, 'postprocessing', 'conns_FC.pkl'), 'rb') as fcf:
        conns_fc = pickle.load(fcf)

    # remove patients 28 & 55 from FC since not in SC
    pt_inds, = np.where(['patient' in subj for subj in subjs])
    pt28_idx, = np.where(['patient28' in subj for subj in subjs[pt_inds]])
    for k,conn in conns_fc.items():
        if k[2]=='patients':
            conns_fc[k] = np.delete(conns_fc[k][:,:,:-1], pt28_idx, axis=-1)
    # patients and controls list
    patients = np.delete(np.array(subjs[pt_inds])[:-1], pt28_idx)
    controls = subjs[np.where(['control' in subj for subj in subjs])[0]]
    subjects = {'controls':controls, 'patients':patients}

    # options
    atlases = ['schaefer100_tianS1', 'schaefer200_tianS2', 'schaefer400_tianS4']
    sc_metrics = ['count_nosift']
    fc_metrics = ['detrend_filtered']
    cohorts = ['controls', 'patients']
    subnets = ['wholebrain', 'fspt', 'Fr', 'StrTh']

    # new data structures
    sc_t = dict()   # thresholded and weight adjusted SC
    gsp = dict()      # graphs & GSP data

    # create graph laplacian
    for subnet, atlas, scm, coh in itertools.product(subnets, atlases, sc_metrics, cohorts):
        sub_inds = get_subnet_indices(atlas)[subnet]
        # threshold SC
        sct, t_ = qsiprep_analysis.threshold_connectivity(conns_sc[atlas,scm,coh][np.ix_(sub_inds,sub_inds)], quantile=0.8)
        sc_t[subnet,atlas,scm,coh] = sct

        graphs = []
        for sc in sct.T:
            np.fill_diagonal(sc,0)
            g = pygsp.graphs.Graph(sc.squeeze(), lap_type=args.lap_type)
            #print('{} {} {} -- graph connected: {}'.format(atlas,scm,coh, g.is_connected()))
            g.compute_fourier_basis()
            graphs.append(g)
        gsp[subnet,atlas,scm,coh] = {'g':graphs}

        # TODO: plot eigenvectors for each individuals

    # plot average connectivity matrices       /!\ not implemented for sub-networks..
    #if args.plot_conns:
    #    plot_conns_matrices(sc_t, atlases, metrics)

    # get bold time series
    bold_ts = get_bold_dict(atlases, fc_metrics, np.concatenate([controls, patients]))

    # apply graph filter to bold time series
    for subnet,atlas,scm,fcm,coh in itertools.product(subnets,atlases,sc_metrics,fc_metrics,cohorts):
        gsp[subnet,atlas,scm,fcm,coh] = {'spect':[], 'l2_low':[], 'l2_high':[]}
        sub_inds = get_subnet_indices(atlas)[subnet]
        for i,sub in enumerate(subjects[coh]):
            g = gsp[subnet,atlas,scm,coh]['g'][i]
            ts = bold_ts[atlas,fcm,coh][i]
            sig = np.expand_dims(ts.T, axis=-1)
            gts = g.gft(sig[sub_inds])
            spect = np.linalg.norm(gts.squeeze(), axis=-1)*g.e
            gsp[subnet,atlas,scm,fcm,coh]['spect'].append(spect)

            lowp = pygsp.filters.Rectangular(g, band_min=0., band_max=0.2)
            highp = pygsp.filters.Rectangular(g, band_min=0.8, band_max=1.)
            # spectral filtering
            lowf = np.expand_dims(lowp.evaluate(g.e).T, axis=-1)
            highf = np.expand_dims(highp.evaluate(g.e).T, axis=-1)
            lows = np.matmul(gts, lowf)
            highs = np.matmul(gts, highf)
            lts = g.igft(lows)
            hts = g.igft(highs)
            gsp[subnet,atlas,scm,fcm,coh]['l2_low'].append(np.linalg.norm(lts.squeeze(), axis=-1))
            gsp[subnet,atlas,scm,fcm,coh]['l2_high'].append(np.linalg.norm(hts.squeeze(), axis=-1))

    # statistical analysis of controls vs patients
    outp_gsp = dict()
    gfreqs = ['low', 'high']
    mcm = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'] # multiple comparison method
    for subnet,atlas,scm,fcm in itertools.product(subnets,atlases,sc_metrics,fc_metrics):
        t_low,p_low = scipy.stats.ttest_ind(gsp[subnet,atlas,scm,fcm,'controls']['l2_low'], \
                                    gsp[subnet,atlas,scm,fcm,'patients']['l2_low'], \
                                    permutations=1000)
        t_high,p_high = scipy.stats.ttest_ind(gsp[subnet,atlas,scm,fcm,'controls']['l2_high'], \
                                    gsp[subnet,atlas,scm,fcm,'patients']['l2_high'], \
                                    permutations=1000)
        p_corrected = dict()
        for m in mcm:
            for i,pvals in enumerate([p_low, p_high]):
                r = multitest.multipletests(pvals, method=m)
                p_corrected[m, gfreqs[i]] = r[1]
        outp_gsp[subnet,atlas,scm,fcm] = {'t_low':t_low, 'p_low':p_low, 't_high':t_high, 'p_high':p_high, 'p_corrected':p_corrected}

        #for gfreq in gfreqs:
            #print_significant_stats(subnet,atlas,scm,fcm,outp_gsp,gfreq)
            #plot_stats_on_glass_brain(subnet,atlas,scm,fcm,outp_gsp,gfreq)
