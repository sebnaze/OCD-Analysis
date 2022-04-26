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
subjs = pd.read_table(os.path.join(code_dir, 'subject_list_all.txt'), names=['name'])['name']

# options
sc_metrics = ['sift'] #unecessary
rois = ["AccR_dPutR_vPutL_lvPFC_lPFC_dPFC_sphere6-12mm"] #['ACC_OFC_ROI', 'PUT_PFC_ROI']
dsi_metrics = ['gfa'] #, 'iso', 'fa0', 'fa1', 'fa2']
subroi = 'vPutdPFC' #'3-2' #'3-1'#'3-2'  # 3-1=contralateral, 3-2=ipsilateral, 3-3=cortico-cortical
#thresholds = [s+'%' for s in np.arange(0,100,20).astype(str)]
cohorts = ['controls', 'patients']

# coords for plotting global masks
xyz = {'ACC_OFC_ROI':(20, 42, -4), 'PUT_PFC_ROI':None, "AccR_dPutR_vPutL_lvPFC_lPFC_dPFC_sphere6-12mm":None}


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

def compute_tdi_maps(thresholds, args):
    """ computes track density maps between pathways ROIs """
    dsi_m = dict( ( ((dsm,scm,roi,thr,coh),{'mean_dsi':[], 'masks':[]}) for dsm,scm,roi,thr,coh in itertools.product(
        dsi_metrics, sc_metrics, rois, thresholds, cohorts)) )
    for dsm,scm,roi,thr in itertools.product(dsi_metrics, sc_metrics, rois, thresholds):
        for i,subj in enumerate(subjs):
            # import track density images (TDI)
            img = get_tdi_img(subj, scm, roi)
            if img==None :
                #subjs.pop(i)
                print("{} TDI img is not found, will apply group level TD mask instead".format(subj))
                mask = None
            elif np.sum(img.get_fdata())==0:
                 print("{} TDI img is empty, will apply group level TD mask instead".format(subj))
                 mask = None
            else:
                # threshold TDI to get binary mask
                img_data = img.get_fdata().copy()
                q_thr = np.percentile(img_data[img_data!=0], thr)
                mask = binarize_img(img, threshold=q_thr)

                if args.plot_indiv_masks:
                    plot_roi(mask, title='_'.join([subj, roi, str(thr)+'%']),
                             output_file=os.path.join(proj_dir, 'postprocessing', subj, '_'.join([subj, roi, str(thr)])+'.png'))

                if 'control' in subj:
                    dsi_m[dsm,scm,roi,thr, 'controls']['masks'].append(mask)
                else:
                    dsi_m[dsm,scm,roi,thr, 'patients']['masks'].append(mask)
            dsi_m[dsm,scm,roi,thr,subj, 'mask'] = mask

    return dsi_m


def compute_diffusion_maps(dsi_m, args):
    """ computes the diffusion metric accross the track density mask """
    for dsm,scm,roi,thr in itertools.product(dsi_metrics, sc_metrics, rois, args.thresholds):
        for i,subj in enumerate(subjs):
            # import TD map
            mask = dsi_m[dsm,scm,roi,thr,subj, 'mask']
            # import diffusion maps (i.e. GFA)
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
            else:
                dsi_m[dsm,scm,roi,thr, 'patients']['mean_dsi'].append(mean_dsi)
            dsi_m[dsm,scm,roi,thr,subj, 'mean_dsi'] = mean_dsi
    return dsi_m

def create_df_dsi(dsi_m, thr, scm='sift', dsm='gfa'):
    """ create dataframe from dict with DSI metrics """
    df_dsi = pd.DataFrame(columns=['mean_dsi', 'cohorts', 'roi'])
    for roi, coh in itertools.product(rois, cohorts):
        n = len(dsi_m[dsm,scm,roi,thr,coh]['mean_dsi'])
        df_ = pd.DataFrame.from_dict({'mean_dsi':dsi_m[dsm,scm,roi,thr,coh]['mean_dsi'],
                                      'cohorts':np.repeat(coh,n),
                                      'roi':np.repeat(roi,n)})
        df_dsi = df_dsi.append(df_, ignore_index=True)
    return df_dsi

def create_summary(dsi_m, args):
    """ Summarize TDI outputs in dataframe """
    summary_df = pd.DataFrame()
    for dsm,scm,roi,thr in itertools.product(dsi_metrics, sc_metrics, rois, args.thresholds):
        df = pd.DataFrame()
        df['subj'] = subjs.copy()
        df['mean_dsi'] = np.concatenate([dsi_m[dsm,scm,roi,thr, 'controls']['mean_dsi'], dsi_m[dsm,scm,roi,thr, 'patients']['mean_dsi']])
        df['cohort'] = np.concatenate([np.repeat('controls', len(dsi_m[dsm,scm,roi,thr, 'controls']['mean_dsi'])),
                                       np.repeat('patients', len(dsi_m[dsm,scm,roi,thr, 'patients']['mean_dsi']))])
        df['thr'] = np.repeat(thr, len(subjs))
        df['roi'] = np.repeat(roi, len(subjs))
        df['dsi_metric'] = np.repeat(dsm, len(subjs))

        summary_df = summary_df.append(df, ignore_index=True)

    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'tdi_fa_summary_df.pkl'), 'wb') as f:
            pickle.dump(summary_df, f)


def threshold_normalize_img(img, thr=0, min_val=0, max_val=1):
    """  Threshold image and normalize the remaining so that it takes values between min_val and max_val """
    data = img.get_fdata().copy()
    thr_data = data[data>thr]
    rescaled = (thr_data - thr_data.min()) / (thr_data.max() - thr_data.min())
    rescaled_data = np.zeros(data.shape)
    rescaled_data[data>thr] = rescaled
    return new_img_like(img, rescaled_data)


def cohen_d(x,y):
    """ Calculates effect size as cohen's d """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def compute_stats(thresholds, args):
    """ performs student's t-test between groups' mean TDI in pathways """
    stats = dict()
    for dsm,scm,roi,thr in itertools.product(dsi_metrics, sc_metrics, rois, thresholds):
        t,p = scipy.stats.ttest_ind(np.array(dsi_m[dsm,scm,roi,thr, 'controls']['mean_dsi']), np.array(dsi_m[dsm,scm,roi,thr, 'patients']['mean_dsi']))
        print("{} {} {} {} - t={:.3f} p={:.3f}".format(dsm,scm,roi,thr,t,p))
        stats[dsm,scm,roi,thr] = {'t':t, 'p':p}
    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'tdi_fa_stats_dict.pkl'), 'wb') as f:
            pickle.dump(stats, f)



## PLOTTING FUNCTIONS ##
def get_group_masks(dsi_m, args):
    for dsm,scm,roi,thr in itertools.product(dsi_metrics, sc_metrics, rois, args.thresholds):
        unscaled_img = mean_img(np.concatenate([dsi_m[dsm,scm,roi,thr,'controls']['masks'], \
                                                dsi_m[dsm,scm,roi,thr,'patients']['masks']]))
        rescaled_img = threshold_normalize_img(unscaled_img)
        mask = binarize_img(rescaled_img, threshold=0.5)

        for i,subj in enumerate(subjs):
            if dsi_m[dsm,scm,roi,thr,subj,'mask'] == None :
                dsi_m[dsm,scm,roi,thr,subj,'mask'] = mask

        if args.plot_group_masks:
            if args.save_figs:
                plot_roi(rescaled_img, cmap='Reds', threshold=0.00001, draw_cross=False,
                          cut_coords=xyz[roi], alpha=0.8, title='{} avg mask at {}%'.format(roi, str(thr)),
                          output_file=os.path.join(proj_dir, 'postprocessing', 'avg_mask_{}_{}percent.svg'.format(roi, str(thr))) )
            if args.plot_figs:
                plot_roi(rescaled_img, cmap='Reds', threshold=0.00001, draw_cross=False,
                          cut_coords=xyz[roi], alpha=0.8)

    return dsi_m


def plot_tdi_distrib(df_dsi, thr, args, ylims=[0.09, 0.14]):
    fig = plt.figure(figsize=[8,4])

    ax1 = plt.subplot(1,3,1)
    sbn.violinplot(data=df_dsi, y='mean_dsi', x='roi', hue='cohorts', orient='v', ax=ax1, split=True, scale_hue=True,
                   inner='quartile', dodge=True, width=0.8, cut=2)
    #sbn.stripplot(data=df_dsi, y='mean_dsi', x='roi', hue='cohorts', orient='v', ax=ax1, split=True, dodge=False,
    #              size=2, edgecolor='black', linewidth=0.5, jitter=0.25)
    plt.ylim(ylims);
    ax1.legend([])

    ax2 = plt.subplot(1,3,2)
    sbn.stripplot(data=df_dsi, y='mean_dsi', x='roi', hue='cohorts', orient='v', ax=ax2, dodge=True, linewidth=1, size=2)
    plt.ylim(ylims);

    ax3 = plt.subplot(1,3,3)
    bplot = sbn.boxplot(data=df_dsi, y='mean_dsi', x='roi', hue='cohorts', orient='v', ax=ax3, fliersize=0)
    #bplot.set_facecolor('blue')
    #bplot['boxes'][1].set_facecolor('orange')
    splot = sbn.swarmplot(data=df_dsi, y='mean_dsi', x='roi', hue='cohorts', orient='v', ax=ax3, dodge=True, linewidth=1, size=6, alpha=0.6)
    plt.ylim(ylims);
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    lgd = ax3.legend(handles=bplot.patches, labels=['HC', 'OCD'], bbox_to_anchor=(1, 1))
    #ax3.set_xticklabels(labels=['NAcc', 'dPut'], minor=True)
    fig.tight_layout()

    if args.save_figs:
        plt.savefig(os.path.join(proj_dir, 'img', 'TD_FA_con_vs_pat_vPutdPFC_'+str(thr)+'_v02.svg'))

    if args.plot_figs:
        plt.show(block=False)

    plt.close()



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_indiv_masks', default=False, action='store_true', help='flag to plot individual masks')
    parser.add_argument('--plot_group_masks', default=False, action='store_true', help='flag to plot group masks')
    parser.add_argument('--keep_masks', default=False, action='store_true', help='flag to keep subjects in dict (take ~16Gb)')
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--compute_tdi', default=False, action='store_true', help='flag to compute track density maps, if not switched it tries to load them')
    parser.add_argument('--plot_tdi_distrib', default=False, action='store_true', help='Plot difference in track density distributions (violin, strip, box plots)')
    args = parser.parse_args()

    # parameters
    thresholds = np.arange(20,101,20) # in %
    args.thresholds = thresholds
    tdi_threshold_to_plot = thresholds[3] # plot 80%

    print(subroi)

    if args.compute_tdi:
        dsi_m = compute_tdi_maps(thresholds, args)
        dsi_m = get_group_masks(dsi_m, args)
        dsi_m = compute_diffusion_maps(dsi_m, args)
    else:
        with open(os.path.join(proj_dir, 'postprocessing', 'dsi_m.pkl'), 'rb') as f:
            dsi_m = pickle.load(f)

    compute_stats(thresholds, args)

    # Effect size
    df_dsi = create_df_dsi(dsi_m, tdi_threshold_to_plot)
    for roi in rois:
      print(roi)
      cons = df_dsi[(df_dsi['cohorts']=='controls') & (df_dsi['roi']==roi)].mean_dsi
      pats = df_dsi[(df_dsi['cohorts']=='patients') & (df_dsi['roi']==roi)].mean_dsi
      print('Controls: mean={:.5f} std={:.5f} n={}'.format(cons.mean(), cons.std(), str(len(cons))))
      print('Patients: mean={:.5f} std={:.5f} n={}'.format(pats.mean(), pats.std(), str(len(pats))))
      print('cohen\'s d at threshold {} = {:.2f}'.format(str(tdi_threshold_to_plot), cohen_d(cons, pats)))

    create_summary(dsi_m, args)

    #avg_mask = plot_group_masks(dsi_m, thresholds, args)

    if args.plot_tdi_distrib:
        plot_tdi_distrib(df_dsi, tdi_threshold_to_plot, args)

    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'dsi_m.pkl'), 'wb') as f:
            pickle.dump(dsi_m, f)
















    # TODO: relation to Y-BOCS
