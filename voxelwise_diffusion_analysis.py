# Script to perform FA analysis on masks derived from track density
# Author: Sebastien Naze
# QIMR Berghofer 2021-2022

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
from nilearn.image import load_img, binarize_img, threshold_img, mean_img, new_img_like
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison, plot_roi, view_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
import numpy as np
import os
import pickle
import pandas as pd
import scipy
from scipy.io import loadmat
import seaborn as sbn
import sklearn
import statsmodels
from statsmodels.stats import multitest
import time
from time import time
import warnings
warnings.filterwarnings('once')

# dirs
proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
lukeH_proj_dir = '/home/sebastin/working/lab_lucac/lukeH/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
lukeH_deriv_dir = os.path.join(lukeH_proj_dir, 'data/derivatives')
atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)
subjs = pd.read_table(os.path.join(code_dir, 'subject_list.txt'), names=['name'])['name']

# options
sc_metrics = ['sift'] #unecessary
rois = ['ACC_OFC_ROI', 'PUT_PFC_ROI']
dsi_metrics = ['gfa'] #, 'iso', 'fa0', 'fa1', 'fa2']
subroi = '3-2' #'3-1'#'3-2'  # 3-1=contralateral, 3-2=ipsilateral, 3-3=cortico-cortical
#thresholds = [s+'%' for s in np.arange(0,100,20).astype(str)]
thresholds = np.arange(0,101,10) # in %
cohorts = ['controls', 'patients']


def get_tdi_img(subj, scm, roi):
    """ import track density images """
    fpath = os.path.join(proj_dir, 'postprocessing', subj)
    fname = '_'.join([roi, scm,subroi]) + '.nii.gz'
    fname = os.path.join(fpath, fname)
    if os.path.exists(fname):
        img = load_img(fname)
    else:
        img = None
    return img


# import FA & mean diffusivity (MD) images
def get_diffusion_metric(subj, metric='gfa'):
    """ import diffusion metrics from DSI Studio such as generalized fractional anisotropy (GFA) """
    fpath = os.path.join(proj_dir, 'data/derivatives/qsirecon', subj, 'dwi')
    fname = subj+'_acq-88_dir-AP_space-T1w_desc-preproc_space-T1w_desc-'+metric+'_gqiscalar.nii.gz'
    fname = os.path.join(fpath, fname)
    if os.path.exists(fname):
        img = load_img(fname)
    else:
        img = None
    return img


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_indiv_masks', default=False, action='store_true', help='flag to plot individual masks')
    parser.add_argument('--keep_masks', default=False, action='store_true', help='flag to keep subjects in dict (take ~16Gb)')
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    args = parser.parse_args()

    # main
    dsi_m = dict( ( ((dsm,scm,roi,thr,coh),{'mean_dsi':[], 'masks':[]}) for dsm,scm,roi,thr,coh in itertools.product(
        dsi_metrics, sc_metrics, rois, thresholds, cohorts)) )
    for dsm,scm,roi,thr in itertools.product(dsi_metrics, sc_metrics, rois, thresholds):
        for i,subj in enumerate(subjs):
            # import track density images (TDI)
            img = get_tdi_img(subj, scm, roi)
            if img==None:
                subjs.pop(i)
                print("{} removed, TDI img not found".format(subj))
                continue

            # threshold TDI to get binary mask
            img_data = img.get_fdata().copy()
            q_thr = np.percentile(img_data[img_data!=0], thr)
            mask = binarize_img(img, threshold=q_thr)
            if args.plot_indiv_masks:
                plot_roi(mask, title='_'.join([subj, roi, str(thr)+'%']),
                         output_file=os.path.join(proj_dir, 'postprocessing', subj, '_'.join([subj, roi, str(thr)])+'.png'))

            img = get_diffusion_metric(subj, metric=dsm)
            if img==None:
                subjs.pop(i)
                print("{} removed, DSI metric img not found".format(subj))
                continue

            # use TDI mask to extract mean FA & MD values for each subject
            data = threshold_img(img, threshold=0, mask_img=mask).get_fdata()
            mean_dsi = np.mean(data[data!=0])
            if 'control' in subj:
                dsi_m[dsm,scm,roi,thr, 'controls']['mean_dsi'].append(mean_dsi)
                if args.keep_masks:
                    dsi_m[dsm,scm,roi,thr, 'controls']['masks'].append(mask)

            else:
                dsi_m[dsm,scm,roi,thr, 'patients']['mean_dsi'].append(mean_dsi)
                if args.keep_masks:
                    dsi_m[dsm,scm,roi,thr, 'patients']['masks'].append(mask)


    # stats
    stats = dict()
    for dsm,scm,roi,thr in itertools.product(dsi_metrics, sc_metrics, rois, thresholds):
        t,p = scipy.stats.ttest_ind(np.array(dsi_m[dsm,scm,roi,thr, 'controls']['mean_dsi']), np.array(dsi_m[dsm,scm,roi,thr, 'patients']['mean_dsi']))
        print("{} {} {} {} - t={:.3f} p={:.3f}".format(dsm,scm,roi,thr,t,p))
        stats[dsm,scm,roi,thr] = {'t':t, 'p':p}
    with open(os.path.join(proj_dir, 'postprocessing', 'tdi_fa_stats_dict.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    # summary df
    summary_df = pd.DataFrame()
    for dsm,scm,roi,thr in itertools.product(dsi_metrics, sc_metrics, rois, thresholds):
        df = pd.DataFrame()
        df['subj'] = subjs.copy()
        df['mean_dsi'] = np.concatenate([dsi_m[dsm,scm,roi,thr, 'controls']['mean_dsi'], dsi_m[dsm,scm,roi,thr, 'patients']['mean_dsi']])
        df['cohort'] = np.concatenate([np.repeat('controls', len(dsi_m[dsm,scm,roi,thr, 'controls']['mean_dsi'])),
                                       np.repeat('patients', len(dsi_m[dsm,scm,roi,thr, 'patients']['mean_dsi']))])
        df['thr'] = np.repeat(thr, len(subjs))
        df['roi'] = np.repeat(roi, len(subjs))
        df['dsi_metric'] = np.repeat(dsm, len(subjs))

        summary_df = summary_df.append(df, ignore_index=True)

    with open(os.path.join(proj_dir, 'postprocessing', 'tdi_fa_summary_df.pkl'), 'wb') as f:
        pickle.dump(summary_df, f)


    # relation to Y-BOCS
