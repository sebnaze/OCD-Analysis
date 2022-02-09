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


proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
lukeH_proj_dir = '/home/sebastin/working/lab_lucac/lukeH/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
lukeH_deriv_dir = os.path.join(lukeH_proj_dir, 'data/derivatives')
atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'
in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/seed_not_smoothed/detrend_gsr_filtered')

# This section should be replaced with propper packaging
import sys
sys.path.insert(0, os.path.join(code_dir))
import importlib
import qsiprep_analysis
import atlaser
from atlaser import Atlaser

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)
subjs = pd.read_table(os.path.join(code_dir, 'subject_list.txt'), names=['name'])['name']

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


# TODO: refactor this function, only a few lines changed from the one above
def sphere_seed_to_voxel(subj, subrois, metrics, atlas='harrison2009'):
    """ perform seed-to-voxel analysis of bold data using Harrison2009 3.5mm sphere seeds"""
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
            fname = '_'.join([subj,metric,roi])+'_sphere_seed_to_voxel_corr.nii.gz'
            nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))
    print('{} seed_to_voxel correlation performed in {}s'.format(subj,int(time()-t0)))

# TODO: adapt to Tian parcellatin, atm works only for Harrison2009 preprocessing
def merge_LR_hemis(subjs, seed_locs, seed_type='sphere_seed_to_voxel'):
    """ merge the left and right correlation images for each seed in each subject """
    hemis = ['L', 'R']
    in_fnames = dict( ((s,[]) for s in seed_locs) )
    for i,seed_loc in enumerate(seed_locs):
        for k,s in enumerate(subjs):
            if 'control' in s:
                coh = 'controls'
            else:
                coh = 'patients'
            fnames = [os.path.join(in_dir, seed_loc+hemi, coh, s+'_detrend_gsr_filtered_'+seed_loc+hemi+'_sphere_seed_to_voxel_corr.nii')
                      for hemi in hemis]
            new_img = nilearn.image.mean_img(fnames)
            fname = s+'_detrend_gsr_filtered_'+seed_loc+'_sphere_seed_to_voxel_corr.nii'
            os.makedirs(os.path.join(in_dir, seed_loc, coh), exist_ok=True)
            nib.save(new_img, os.path.join(in_dir, seed_loc, coh, fname))
            in_fnames[seed_loc].append(os.path.join(in_dir, seed_loc, coh, fname))


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

def perform_second_level_analysis(seed, design_matrix, cohorts=['controls', 'patients']):
    """ Perform second level analysis based on seed-to-voxel correlation maps """
    glm = SecondLevelModel()
    con_flist = glob.glob(os.path.join(in_dir, seed, 'controls', '*'))
    pat_flist = glob.glob(os.path.join(in_dir, seed, 'patients', '*'))
    flist = np.hstack([con_flist, pat_flist])

    glm.fit(list(flist), design_matrix=design_matrix)
    contrast = glm.compute_contrast(np.array([1, -1]), output_type='stat')
    return contrast

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
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    args = parser.parse_args()

    # options
    atlases= ['schaefer400_harrison2009'] #['schaefer100_tianS1', 'schaefer200_tianS2', 'schaefer400_tianS4'] #schaefer400_harrison2009
    metrics = ['detrend_filtered', 'detrend_gsr_filtered']
    subrois = ['Acc', 'dCaud', 'dPut', 'vPut', 'drPut'] #['Acc', 'Caud', 'Put']

    #subrois = ['AccL', 'AccR', 'dCaudL', 'dCaudR', 'dPutL', 'dPutR', 'vPutL', 'vPutR', 'vCaudSupL', 'vCaudSupR', 'drPutL', 'drPutR']

    #TODO: in_dir must be tailored to the atlas. ATM everything is put in Harrison2009 folder

    seedfunc = {'Harrison2009':sphere_seed_to_voxel,
            'TianS4':seed_to_voxel}

    if args.compute_seed_corr:
        Parallel(n_jobs=10)(delayed(seedfunc[args.seed_type])(subj,subrois,metrics,atlases) for subj in subjs)

    seeds = subrois
    if args.compute_seed_corr:
        merge_LR_hemis(subjs, seeds, seed_type=str(seed_func[args.seed_type]))
    design_matrix = create_design_matrix(subjs)

    out_dir = os.path.join(proj_dir, 'postprocessing', 'glm')
    os.makedirs(out_dir, exist_ok=True)

    glm_results = dict()
    for seed in seeds:
        glm_results[seed] = dict()
        glm_results[seed]['contrast'] = perform_second_level_analysis(seed, design_matrix)
        glm_results[seed]['thresholded_img'], glm_results[seed]['thresh'], glm_results[seed]['cluster_table'] = threshold_contrast(glm_results[seed]['contrast'])

        if args.plot_figs:
            plot_stat_map(glm_results[seed]['contrast'], draw_cross=False, threshold=glm_results[seed]['thresh'],
                            title=seed+'_contrast_fpr0001')

        if args.save_figs:
            plot_stat_map(glm_results[seed]['contrast'], draw_cross=False, threshold=glm_results[seed]['thresh'],
            output_file=os.path.join(out_dir,seed+'_contrast_fpr0001.pdf'))

    # savings
    if args.save_outputs:
        with open(os.path.join(out_dir,seed+'_contrast_fpr0001.pkl'), 'wb') as of:
            pickle.dump(glm_results, of)
