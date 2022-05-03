################################################################################
# Functional analysis
#
# Author: Sebastien Naze
# QIMR Berghofer
# 2021
################################################################################
import argparse
import bct
import h5py
import itertools
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

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
        atlas_cfg = json.load(jsf)
        subjs = pd.read_table(os.path.join(code_dir, 'subject_list.txt'), names=['name'])['name']

def get_FC_connectomes(subjs, atlases, metrics, opts={'smoothing_fwhm':8, 'verbose':True}):
    """ Get connectomes from controls and patients
    inputs:
        subjs: list of subjects
        atlas: list of  atlases
        metrics: list of metrics used for filtering
        opts: dict of extras
    outputs:
        connectivity matrices: dict. with keys 'controls' and 'patients'
    """
    cohorts = ['controls', 'patients']
    conns = dict( ( ((atl,met,coh),[]) for atl,met,coh in itertools.product(atlases,metrics,cohorts) ) )
    discarded = []
    for i,subj in enumerate(subjs):
        subj_dir = os.path.join(deriv_dir, 'post-fmriprep-fix', subj, 'fc')
        for atlas,metric in itertools.product(atlases, metrics):
            if opts['verbose']:
                print('Creating {} functional connectivity with {} {} fwhm={}'.format(
                        subj,atlas,metric,opts['smoothing_fwhm']))

            fname = subj+'_task-rest_atlas-'+atlas+'_desc-correlation-'+metric+ \
                                '_scrub_fwhm'+str(opts['smoothing_fwhm'])+'.h5'
            fpath = os.path.join(subj_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as hf:
                    f = h5py.File(hf)
                    c = np.array(f['fc'])
                    if 'control' in subj:
                        conns[(atlas,metric,'controls')].append(c)
                    elif 'patient' in subj:
                        conns[(atlas,metric,'patients')].append(c)
                    else:
                        discarded.append((i,subj))
                        #subjs.drop(i)
                        print('Subject {} neither control nor patient? Subjected discarded, check name spelling'.format([subj]))
                        continue;
            else:
                discarded.append((i,subj))
                #subjs.drop(i)
                print(fpath, '\nSubject {} preprocessing not found, subject has been removed for subjects list.'.format([subj]))
                continue;

    # print summary
    for k,v in conns.items():
        conns[k] = np.transpose(np.array(v),(1,2,0))
        if opts['verbose']:
            print(' --> shape: '.join([str(k), str(conns[k].shape)]))

    subjs = subjs.drop([disc[0] for disc in discarded])
    return conns,subjs,discarded


def plot_min_edges_analysis(res, atlases, metrics, subrois):
    fig = plt.figure(figsize=[18,10])
    gs = fig.add_gridspec(2,3)
    for i,atlas in enumerate(atlases):
        for j,metric in enumerate(metrics):
            ax = fig.add_subplot(gs[j,i])
            for roi in subrois:
                l = []
                for k in min_edges:
                    if res[k][atlas,metric,roi]!=None:
                        l.append(np.min(res[k][atlas,metric,roi]['qvals']))
                    else:
                        l.append(np.nan)
                ax.plot(min_edges, l, '.-')
            ax.legend(subrois)
            ax.set_xlabel('n_min_edges')
            ax.set_ylabel('min(q)')
            ax.set_ylim([0,1])
            ax.axhline(y=0.05, color='red')
            ax.annotate('q=0.05', xy=(0,0.07), xycoords='data', color='red')
            ax.set_title('{} {}'.format(atlas, metric))
    plt.show(block=False)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    args = parser.parse_args()

    # options
    atlases = ['schaefer100_tianS1', 'schaefer200_tianS2', 'schaefer400_tianS4']
    metrics = ['detrend_gsr_filtered', 'detrend_filtered']

    conns, subjs, discarded = get_FC_connectomes(subjs, atlases, metrics)

    # save FC connectomes
    if args.save_outputs:
        fname = os.path.join(proj_dir,'postprocessing','conns_FC.pkl')
        with open(fname, 'wb') as pf:
            pickle.dump(conns,pf)

    qsiprep_analysis.plot_conns_matrices(conns, atlases, metrics, scale='lin', vmin=-1, vmax=1)

    qsiprep_analysis.plot_conns_hists(conns, atlases, metrics, scale='lin')

    subrois = ['Acc', 'Caud', 'Put']
    ## INCLUDING (VS NOT INCLUDING) THALAMUS IN suprois to treat is as non-Frontal (vs Frontal)
    outp = qsiprep_analysis.run_stat_analysis(conns, atlases, metrics, subrois, threshold=True, quantile=0.8)
    #outp = qsiprep_analysis.run_stat_analysis(conns, atlases, metrics, subrois, suprois=['Pal','Put','Caud','Acc'], threshold=True, quantile=0.8)

    # save stats analysis
    if args.save_outputs:
        fname = os.path.join(proj_dir,'postprocessing','outp_FC.pkl')
        with open(fname, 'wb') as pf:
            pickle.dump(outp,pf)

    rois = np.concatenate([subrois, ['all']])
    # plot results on glass brain
    for atlas,metric,roi in itertools.product(atlases, metrics, rois):
        if np.any(outp[atlas,metric,roi]['pvals']<=0.05):
                qsiprep_analysis.plot_pq_values(outp, atlas, metric, roi)
                qsiprep_analysis.plot_stats_on_glass_brain(atlas, metric, roi, outp)


    # minimum edges analysis (number of subjects with needed to perform t-test)
    min_edges = np.arange(40)
    res=[]
    for nme in min_edges:
        outp = qsiprep_analysis.run_stat_analysis(conns, atlases, metrics, subrois, suprois=['Thal', 'Pal','Put','Caud','Acc'], threshold=True, quantile=0.8, n_min_edges=nme, verbose=False)
        res.append(outp)
    plot_min_edges_analysis(res, atlases, metrics, subrois)
