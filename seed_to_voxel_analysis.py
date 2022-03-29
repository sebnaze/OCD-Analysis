# Script to perform FC analysis based on seed or parcellation to voxel correlations
# Author: Sebastien Naze
# QIMR Berghofer 2021-2022

import argparse
import bct
import glob
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
from nilearn import datasets
from nilearn.image import load_img, new_img_like
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison, plot_img, plot_roi, view_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
import numpy as np
import os
import pickle
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy import ndimage
import statsmodels
from statsmodels.stats import multitest
import time
from time import time
import platform

# get computer name to set paths
if platform.node()=='qimr18844':
    working_dir = '/home/sebastin/working/'
elif 'hpcnode' in platform.node():
    working_dir = '/mnt/lustre/working/'
else:
    print('Computer unknown! Setting working dir as /working')
    working_dir = '/working/'

# general paths
proj_dir = working_dir+'lab_lucac/sebastiN/projects/OCDbaseline'
lukeH_proj_dir = working_dir+'lab_lucac/lukeH/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
lukeH_deriv_dir = os.path.join(lukeH_proj_dir, 'data/derivatives')
atlas_dir = working_dir+'lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'
#in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/seed_not_smoothed/detrend_gsr_filtered')
#in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/unscrubbed_seed_not_smoothed/detrend_gsr_filtered')

# This section should be replaced with propper packaging one day
import sys
sys.path.insert(0, os.path.join(code_dir))
import importlib
import qsiprep_analysis
import atlaser
from atlaser import Atlaser

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)

# Harrison 2009 seed locations:
seed_loc = {'AccL':[-9,9,-8], 'AccR':[9,9,-8], \
        'dCaudL':[-13,15,9], 'dCaudR':[13,15,9], \
        'dPutL':[-28,1,3], 'dPutR':[28,1,3], \
        'vPutL':[-20,12,-3] , 'vPutR':[20,12,-3], \
        'vCaudSupL':[-10,15,0], 'vCaudSupR':[10,15,0], \
        'drPutL':[-25,8,6], 'drPutR':[25,8,6]}


def seed_to_voxel(subj, subrois, metrics, atlases):
    """ perform seed-to-voxel analysis of bold data """
    # prepare output directory
    out_dir = os.path.join(proj_dir, 'postprocessing', subj)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time()

    for metric in metrics:
        # get bold time series for each voxel
        img_space = 'MNI152NLin2009cAsym'
        bold_file = os.path.join(lukeH_deriv_dir, 'post-fmriprep-fix', subj,'func', \
                                 subj+'_task-rest_space-'+img_space+'_desc-'+metric+'_scrub.nii.gz')
        bold_img = nib.load(bold_file)
        brain_masker = NiftiMasker(smoothing_fwhm=4, standardize=True, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, memory='nilearn_cache', memory_level=1, verbose=0)
        voxels_ts = brain_masker.fit_transform(bold_img)

        for atlas in atlases:
            # prepare output file
            hfname = subj+'_task-rest_'+atlas+'_desc-'+metric+'_'+''.join(subrois)+'_seeds_ts.h5'
            hf = h5py.File(os.path.join(deriv_dir, 'post-fmriprep-fix', subj, 'timeseries' ,hfname), 'w')

            # get atlas utility
            atlazer = Atlaser(atlas)

            # extract seed timeseries and perform seed-to-voxel correlation
            for roi in subrois:
                roi_img = atlazer.create_subatlas_img(roi)
                roi_masker = NiftiLabelsMasker(roi_img, standardize='zscore')
                roi_ts = np.squeeze(roi_masker.fit_transform(bold_img))
                seed_to_voxel_corr = np.dot(voxels_ts.T, roi_ts)/roi_ts.shape[0]
                seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_corr.mean(axis=-1).T)
                fname = '_'.join([subj,atlas,metric,roi])+'_seed_to_voxel_corr.nii.gz'
                nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))
                hf.create_dataset(roi+'_ts', data=roi_ts)
            hf.close()
    print('{} seed_to_voxel performed in {}s'.format(subj,int(time()-t0)))


# TODO: should refactor this function, only a few lines changed from the one above
def sphere_seed_to_voxel(subj, subrois, metrics, atlases=['Harrison2009']):
    """ perform seed-to-voxel analysis of bold data using Harrison2009 3.5mm sphere seeds"""
    # prepare output directory
    out_dir = os.path.join(proj_dir, 'postprocessing', subj)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time()

    for atlas,metric in itertools.product(atlases,metrics):
        # get bold time series for each voxel
        img_space = 'MNI152NLin2009cAsym'
        #bold_file = os.path.join(lukeH_deriv_dir, 'post-fmriprep-fix', subj,'func', \
        #                       subj+'_task-rest_space-'+img_space+'_desc-'+metric+'_scrub.nii.gz')
        bold_file = os.path.join(deriv_dir, 'post-fmriprep-fix', subj,'func', \
                                 subj+'_task-rest_space-'+img_space+'_desc-'+metric+'.nii.gz')
        bold_img = nib.load(bold_file)
        brain_masker = NiftiMasker(smoothing_fwhm=8, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, memory='nilearn_cache', memory_level=1, verbose=0)
        voxels_ts = brain_masker.fit_transform(bold_img)

        # extract seed timeseries and perform seed-to-voxel correlation
        for roi in subrois:
            roi_masker = NiftiSpheresMasker([np.array(seed_loc[roi])], radius=3.5, t_r=0.81, \
                                low_pass=0.1, high_pass=0.01, memory='nilearn_cache', memory_level=1, verbose=0)
            roi_ts = np.squeeze(roi_masker.fit_transform(bold_img))
            seed_to_voxel_corr = np.dot(voxels_ts.T, roi_ts)/roi_ts.shape[0]
            seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_corr)
            fname = '_'.join([subj,metric,atlas,roi,'ns_sphere_seed_to_voxel_corr.nii.gz'])
            nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))
    print('{} seed_to_voxel correlation performed in {}s'.format(subj,int(time()-t0)))

# TODO: adapt to Tian parcellatin, atm works only for Harrison2009 preprocessing
def merge_LR_hemis(subjs, subrois, metrics, seed_type='sphere_seed_to_voxel', atlas='Harrison2009', args=None):
    """ merge the left and right correlation images for each seed in each subject """
    hemis = ['L', 'R']
    in_fnames = dict( ((subroi,[]) for subroi in subrois) )
    for metric in metrics:
        for i,subroi in enumerate(subrois):
            for k,subj in enumerate(subjs):
                if 'control' in subj:
                    coh = 'controls'
                else:
                    coh = 'patients'

                fnames = [os.path.join(proj_dir, 'postprocessing', subj, '_'.join([subj,metric,atlas,subroi+hemi,'ns_sphere_seed_to_voxel_corr.nii.gz']))
                          for hemi in hemis]
                new_img = nilearn.image.mean_img(fnames)
                #fname = s+'_detrend_gsr_filtered_'+subroi+'_sphere_seed_to_voxel_corr.nii'
                fname = '_'.join([subj,metric,atlas,subroi])+'_ns_sphere_seed_to_voxel_corr.nii'
                os.makedirs(os.path.join(args.in_dir, metric, subroi, coh), exist_ok=True)
                nib.save(new_img, os.path.join(args.in_dir, metric, subroi, coh, fname))
                in_fnames[subroi].append(os.path.join(args.in_dir, metric, subroi, coh, fname))


def create_design_matrix(subjs):
    """ Create a simple group difference design matrix """
    n_con = np.sum(['control' in s for s in subjs])
    n_pat = np.sum(['patient' in s for s in subjs])

    design_mat = np.zeros((n_con+n_pat,2), dtype=int)
    design_mat[:n_con,0] = 1
    design_mat[-n_pat:, 1] = 1

    design_matrix = pd.DataFrame()
    design_matrix['con'] = design_mat[:,0]
    design_matrix['pat'] = design_mat[:,1]
    return design_matrix

def perform_second_level_analysis(seed, metric, design_matrix, cohorts=['controls', 'patients'], args=None):
    """ Perform second level analysis based on seed-to-voxel correlation maps """
    glm = SecondLevelModel()
    con_flist = glob.glob(os.path.join(args.in_dir, metric, seed, 'controls', '*'))
    pat_flist = glob.glob(os.path.join(args.in_dir, metric, seed, 'patients', '*'))
    flist = np.hstack([con_flist, pat_flist])

    glm.fit(list(flist), design_matrix=design_matrix)
    contrasts = glm.compute_contrast(np.array([1, -1]), output_type='all')
    n_voxels = np.sum(nilearn.image.get_data(glm.masker_.mask_img_))
    return contrasts, n_voxels

def threshold_contrast(contrast, height_control='fpr', alpha=0.001, cluster_threshold=10):
    """ cluster threshold contrast at alpha with height_control method for multiple comparisons """
    thresholded_img, thresh = threshold_stats_img(
        contrast, alpha=alpha, height_control=height_control, cluster_threshold=cluster_threshold)
    cluster_table = get_clusters_table(
        contrast, stat_threshold=thresh, cluster_threshold=cluster_threshold,
        two_sided=True, min_distance=5.0)
    return thresholded_img, thresh, cluster_table

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--seed_type', default='Harrison2009', type=str, action='store', help='choose Harrison2009, TianS4, etc')
    parser.add_argument('--compute_seed_corr', default=False, action='store_true', help="Flag to (re)compute seed to voxel correlations")
    parser.add_argument('--merge_LR_hemis', default=False, action='store_true', help="Flag to merge hemisphere's correlations")
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--subj', default=None, action='store', help='to process a single subject, give subject ID (default: process all subjects)')
    parser.add_argument('--run_second_level', default=False, action='store_true', help='run second level statistics')
    args = parser.parse_args()

    if args.subj!=None:
        subjs = [args.subj]
    else:
        subjs = pd.read_table(os.path.join(code_dir, 'subject_list.txt'), names=['name'])['name']

    # options
    atlases= ['Harrison2009'] #['schaefer100_tianS1', 'schaefer200_tianS2', 'schaefer400_tianS4'] #schaefer400_harrison2009
    #metrics = ['detrend_filtered', 'detrend_gsr_filtered']
    pre_metric = 'seed_not_smoothed' #'unscrubbed_seed_not_smoothed'
    metrics = ['detrend_gsr_filtered_scrubFD08']

    #TODO: in_dir must be tailored to the atlas. ATM everything is put in Harrison2009 folder
    args.in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/', args.seed_type+'Rep', pre_metric)

    seeds = list(seed_loc.keys()) #['AccL', 'AccR', 'dCaudL', 'dCaudR', 'dPutL', 'dPutR', 'vPutL', 'vPutR', 'vCaudSupL', 'vCaudSupR', 'drPutL', 'drPutR']
    subrois = np.unique([seed[:-1] for seed in seeds])#['Acc', 'dCaud', 'dPut', 'vPut', 'drPut']


    seedfunc = {'Harrison2009':sphere_seed_to_voxel,
            'TianS4':seed_to_voxel}

    if args.compute_seed_corr:
        for atlas in atlases:
            Parallel(n_jobs=args.n_jobs)(delayed(seedfunc[args.seed_type])(subj,seeds,metrics,atlases) for subj in subjs)


    if args.merge_LR_hemis:
        merge_LR_hemis(subjs, subrois, metrics, seed_type=str(seedfunc[args.seed_type]), args=args)

    if args.run_second_level:
        design_matrix = create_design_matrix(subjs)

        out_dir = os.path.join(proj_dir, 'postprocessing', 'glm', pre_metric)
        os.makedirs(out_dir, exist_ok=True)

        glm_results = dict()
        for metric,subroi in itertools.product(metrics,subrois):
            print('Starting 2nd level analysis for '+subroi+' subroi.')
            t0 = time()
            glm_results[subroi] = dict()
            glm_results[subroi]['contrasts'], glm_results[subroi]['n_voxels']  = perform_second_level_analysis(subroi, metric, design_matrix, args=args)
            # Correcting the p-values for multiple testing and taking negative logarithm
            neg_log_pval = nilearn.image.math_img("-np.log10(np.minimum(1, img * {}))"
                            .format(str(glm_results[subroi]['n_voxels'])),
                            img=glm_results[subroi]['contrasts']['p_value'])
            glm_results[subroi]['neg_log_pval'] = neg_log_pval
            #for k in glm_results[subroi]['contrasts'].keys():
            #glm_results[subroi][k] = dict()
            glm_results[subroi]['thresholded_img'], glm_results[subroi]['thresh'], glm_results[subroi]['cluster_table'] = threshold_contrast(glm_results[subroi]['contrasts']['z_score'])

            if args.plot_figs:
                plot_stat_map(glm_results[subroi]['contrasts']['stat'], draw_cross=False, threshold=glm_results[subroi]['thresh'],
                                title=subroi+'_contrast_fpr0001')

            if args.save_figs:
                plot_stat_map(glm_results[subroi]['contrasts']['stat'], draw_cross=False, threshold=glm_results[subroi]['thresh'],
                output_file=os.path.join(out_dir,subroi+'_contrast_fpr0001.pdf'))

            # savings
            if args.save_outputs:
                with open(os.path.join(out_dir,subroi+'_contrast_fpr0001.pkl'), 'wb') as of:
                    pickle.dump(glm_results, of)

            print('Finished 2nd level analysis for '+subroi+' subroi in {:.2f}s'.format(time()-t0))
