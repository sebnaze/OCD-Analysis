# Script to perform FC analysis based on seed or parcellation to voxel correlations
# Author: Sebastien Naze
# QIMR Berghofer 2021-2022

import argparse
import bct
import datetime
import glob
import gzip
import h5py
import importlib
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
from nilearn.image import load_img, new_img_like, resample_to_img, binarize_img, iter_img, math_img
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison, plot_img, plot_roi, view_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
import numpy as np
import os
import pickle
import pandas as pd
import pdb
import scipy
from scipy.io import loadmat
from scipy import ndimage
import seaborn as sbn
import shutil
import statsmodels
from statsmodels.stats import multitest
import sys
import time
from time import time
import platform
import warnings
warnings.filterwarnings('once')

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

# This section should be replaced with propper packaging one day
sys.path.insert(0, os.path.join(code_dir, 'old'))
sys.path.insert(0, os.path.join(code_dir, 'utils'))
sys.path.insert(0, os.path.join(code_dir, 'structural'))
import qsiprep_analysis
import atlaser
from voxelwise_diffusion_analysis import cohen_d

# there you go:
#from ..old import qsiprep_analysis
#from ..utils import atlaser
#from ..structural.voxelwise_diffusion_analysis import cohen_d

from atlaser import Atlaser

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)

# Harrison 2009 seed locations:
seed_loc = {'AccL':[-9,9,-8], 'AccR':[9,9,-8], \
        'dCaudL':[-13,15,9], 'dCaudR':[13,15,9], \
        'dPutL':[-28,1,3], 'dPutR':[28,1,3], \
        'vPutL':[-20,12,-3] , 'vPutR':[20,12,-3]} #, \
        #'vCaudSupL':[-10,15,0], 'vCaudSupR':[10,15,0], \
        #'drPutL':[-25,8,6], 'drPutR':[25,8,6]}

cohorts = ['controls', 'patients']

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

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


def seed_to_voxel(subj, seeds, metrics, atlases, smoothing_fwhm=8.):
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
        brain_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm, standardize=True, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, verbose=0)
        voxels_ts = brain_masker.fit_transform(bold_img)

        for atlas in atlases:
            # prepare output file
            hfname = subj+'_task-rest_'+atlas+'_desc-'+metric+'_'+''.join(seeds)+'_seeds_ts.h5'
            hf = h5py.File(os.path.join(deriv_dir, 'post-fmriprep-fix', subj, 'timeseries' ,hfname), 'w')

            # get atlas utility
            atlazer = Atlaser(atlas)

            # extract seed timeseries and perform seed-to-voxel correlation
            for seed in seeds:
                seed_img = atlazer.create_subatlas_img(seed)
                seed_masker = NiftiLabelsMasker(seed_img, standardize='zscore')
                seed_ts = np.squeeze(seed_masker.fit_transform(bold_img))
                seed_to_voxel_corr = np.dot(voxels_ts.T, seed_ts)/seed_ts.shape[0]
                seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_corr.mean(axis=-1).T)
                fname = '_'.join([subj,atlas,metric,seed])+'_seed_to_voxel_corr.nii.gz'
                nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))
                hf.create_dataset(seed+'_ts', data=seed_ts)
            hf.close()
    print('{} seed_to_voxel performed in {}s'.format(subj,int(time()-t0)))


# TODO: should refactor this function, only a few lines changed from the one above
def sphere_seed_to_voxel(subj, seeds, metrics, atlases=['Harrison2009'], args=None):
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
        brain_masker = NiftiMasker(smoothing_fwhm=args.brain_smoothing_fwhm, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, verbose=0)
        voxels_ts = brain_masker.fit_transform(bold_img)

        # extract seed timeseries and perform seed-to-voxel correlation
        for seed in seeds:
            seed_masker = NiftiSpheresMasker([np.array(seed_loc[seed])], radius=3.5, t_r=0.81, \
                                low_pass=0.1, high_pass=0.01, verbose=0)
            seed_ts = np.squeeze(seed_masker.fit_transform(bold_img))
            seed_to_voxel_corr = np.dot(voxels_ts.T, seed_ts)/seed_ts.shape[0]
            seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_corr)
            fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
            fname = '_'.join([subj,metric,fwhm,atlas,seed,'ns_sphere_seed_to_voxel_corr.nii.gz'])
            nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))
    print('{} seed_to_voxel correlation performed in {}s'.format(subj,int(time()-t0)))

# TODO: adapt to Tian parcellatin, atm works only for Harrison2009 preprocessing
def merge_LR_hemis(subjs, seeds, metrics, seed_type='sphere_seed_to_voxel', atlas='Harrison2009', args=None):
    """ merge the left and right correlation images for each seed in each subject """
    hemis = ['L', 'R']
    in_fnames = dict( ( ((seed,metric),[]) for seed,metric in itertools.product(seeds,metrics) ) )
    for metric in metrics:
        for i,seed in enumerate(seeds):
            for k,subj in enumerate(subjs):
                if 'control' in subj:
                    coh = 'controls'
                else:
                    coh = 'patients'

                fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
                fnames = [os.path.join(proj_dir, 'postprocessing', subj, '_'.join([subj,metric,fwhm,atlas,seed+hemi,'ns_sphere_seed_to_voxel_corr.nii.gz']))
                          for hemi in hemis]
                new_img = nilearn.image.mean_img(fnames)
                #fname = s+'_detrend_gsr_filtered_'+seed+'_sphere_seed_to_voxel_corr.nii'
                fname = '_'.join([subj,metric,fwhm,atlas,seed])+'_ns_sphere_seed_to_voxel_corr.nii'
                os.makedirs(os.path.join(args.in_dir, metric, fwhm, seed, coh), exist_ok=True)
                nib.save(new_img, os.path.join(args.in_dir, metric, fwhm, seed, coh, fname))
                in_fnames[(seed,metric)].append(os.path.join(args.in_dir, metric, fwhm, seed, coh, fname))
    print('Merged L-R hemishperes')
    return in_fnames


def prep_fsl_randomise(in_fnames, seeds, metrics, args):
    """ prepare 4D images for FSL randomise """
    gm_mask = datasets.load_mni152_gm_mask()
    masker = NiftiMasker(gm_mask)

    ind_mask = False  # whether to apply GM mask to individual 3D images separatetely or already combined in 4D image
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))

    for metric,seed in itertools.product(metrics,seeds):
        if ind_mask:
            imgs = None
            for in_fname in in_fnames[(seed,metric)]:
                img = load_img(in_fname)
                masker.fit(img)
                masker.generate_report()
                masked_data = masker.transform(img)
                masked_img = masker.inverse_transform(masked_data)
                if imgs==None:
                    imgs = img
                else:
                    imgs = nilearn.image.concat_imgs([imgs, masked_img], auto_resample=True)
        else:
            img = nilearn.image.concat_imgs(in_fnames[(seed,metric)], auto_resample=True, verbose=0)
            masker.fit(img)
            masker.generate_report()
            masked_data = masker.transform(img)
            imgs = masker.inverse_transform(masked_data) #nib.Nifti1Image(masked_data, img.affine, img.header)
        nib.save(imgs, os.path.join(args.in_dir, metric, fwhm, 'masked_resampled_pairedT4D_'+seed))

    # saving design matrix and contrast
    design_matrix = create_design_matrix(subjs)
    design_con = np.hstack((1, -1)).astype(int)
    np.savetxt(os.path.join(args.in_dir, metric, fwhm, 'design_mat'), design_matrix, fmt='%i')
    np.savetxt(os.path.join(args.in_dir, metric, fwhm, 'design_con'), design_con, fmt='%i', newline=' ')


def unzip_correlation_maps(subjs, metrics, atlases, seeds, args):
    """ extract .nii files from .nii.gz and put them in place for analysis with SPM (not used if only analysing with nilearn) """
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    [os.makedirs(os.path.join(args.in_dir, metric, fwhm, seed, coh), exist_ok=True) \
            for metric,seed,coh in itertools.product(metrics,seeds,cohorts)]

    print('Unzipping seed-based correlation maps for use in SPM...')

    for subj,metric,atlas,seed in itertools.product(subjs,metrics,atlases,seeds):
        fname = '_'.join([subj,metric,fwhm,atlas,seed])+'_ns_sphere_seed_to_voxel_corr.nii.gz'
        infile = os.path.join(proj_dir, 'postprocessing', subj, fname)
        if 'control' in subj:
            coh = 'controls'
        else:
            coh = 'patients'
        with gzip.open(infile, 'rb') as f_in:
            with open(os.path.join(args.in_dir, metric, fwhm, seed, coh, fname[:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def get_subjs_after_scrubbing(subjs, metrics, min_time=5):
    scrub_key = 'scrubbed_length_min'
    scrub_thr = min_time
    proc_dir = 'post-fmriprep-fix'
    d_dir = deriv_dir #lukeH_deriv_dir

    revoked = []
    for subj,metric in itertools.product(subjs, metrics):
        fname = 'fmripop_'+metric+'_parameters.json'
        fpath = os.path.join(d_dir, proc_dir, subj, 'func', fname)
        with open(fpath, 'r') as f:
            f_proc = json.load(f)
            if f_proc[scrub_key] < scrub_thr:
                print("{} has less than {:.2f} min of data left after scrubbing, removing it..".format(subj, f_proc[scrub_key]))
                revoked.append(subj)

    rev_inds = [np.where(s==subjs)[0][0] for s in revoked]
    subjs = subjs.drop(rev_inds)
    return subjs, revoked


def create_local_sphere_within_cluster(vois, rois, metrics, args=None, sphere_radius=3.5):
    """ create a sphere VOIs of given radius within cluster VOIs (for DCM analysis) """
    max_locals = dict( ( ( (roi,metric) , {'controls':[], 'patients':[]}) for roi,metric in itertools.product(rois, metrics) ) )
    for metric in metrics:
        for roi,voi in zip(roi,vois):
            for subj in subjs:
                if 'control' in subj:
                    coh = 'controls'
                else:
                    coh = 'patients'

                fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
                corr_file = '_'.join([subj,metric,fwhm,atlas,roi,'ns_sphere_roi_to_voxel_corr.nii'])
                corr_fname = os.path.join(corr_path, roi, coh, corr_file)
                img = load_img(corr_fname)
                mask = load_img(proj_dir+'/postprocessing/SPM/rois_and_rois/'+voi+'.nii')
                mask = nilearn.image.new_img_like(img, mask.get_fdata())
                new_data = np.abs(img.get_fdata()) * mask.get_fdata()
                local_max = new_data.max()
                max_coords = np.where(new_data>=local_max)
                local_max_coords = nilearn.image.coord_transform(max_coords[0], max_coords[1], max_coords[2], img.affine)
                new_mask = nltools.create_sphere(local_max_coords, radius=sphere_radius)
                out_path = os.path.join(proj_dir, 'postprocessing', subj, 'spm', 'masks')
                os.makedirs(out_path, exist_ok=True)
                fname = os.path.join(out_path, 'local_'+voi+'_'+metric+'.nii')
                nib.save(new_mask, fname)
                max_locals[roi,metric][coh].append(new_mask)
    return max_locals


def resample_masks(masks):
    """ resample all given masks to the affine of the first in list """
    ref_mask = masks[0]
    out_masks = [ref_mask]
    for mask in masks[1:]:
        out_masks.append(resample_to_img(mask, ref_mask, interpolation='nearest'))
    return out_masks

def mask_imgs(flist, masks, args=None):
    """ mask input images using intersection of template masks and pre-compuetd within-groups union mask """
    # mask images to improve SNR
    t_mask = time()
    if args.use_gm_mask:
        gm_mask = datasets.load_mni152_gm_mask()
        masks.append(binarize_img(gm_mask))
    if args.use_fspt_mask: ## not sure it works fine
        fspt_mask = load_img(os.path.join(proj_dir, 'utils', 'Larger_FrStrPalThal_schaefer400_tianS4MNI_lps_mni.nii'), dtype=np.float64)
        masks.append(binarize_img(fspt_mask))
    if args.use_cortical_mask:
        ctx_mask = load_img(os.path.join(proj_dir, 'utils', 'schaefer_cortical.nii'), dtype=np.float64)
        masks.append(binarize_img(ctx_mask))
    if masks != []:
        masks = resample_masks(masks)
        mask = nilearn.masking.intersect_masks(masks, threshold=1, connected=False) # thr=1 : intersection; thr=0 : union
        masker = NiftiMasker(mask)
        masker.fit(imgs=list(flist))
        #masker.generate_report() # use for debug
        masked_data = masker.transform(imgs=list(flist))
        imgs = masker.inverse_transform(masked_data)
        imgs = list(iter_img(imgs))  # 4D to list of 3D
    else:
        imgs = list(flist)
        masker=None
    print('Masking took {:.2f}s'.format(time()-t_mask))
    return imgs, masker

def perform_second_level_analysis(seed, metric, design_matrix, cohorts=['controls', 'patients'], args=None, masks=[]):
    """ Perform second level analysis based on seed-to-voxel correlation maps """
    # naming convention is file system
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))

    # get images path
    con_flist = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'controls', '*'))
    pat_flist = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'patients', '*'))
    flist = np.hstack([con_flist, pat_flist])

    # remove revoked subjects
    if args.revoked != []:
        flist = [l for l in flist if ~np.any([s in l for s in revoked])]

    imgs, masker = mask_imgs(flist, masks=masks, args=args)

    # perform analysis
    t_glm = time()
    glm = SecondLevelModel(mask_img=masker)
    glm.fit(imgs, design_matrix=design_matrix)
    print('GLM fitting took {:.2f}s'.format(time()-t_glm))

    contrasts = dict()
    t0 = time()
    contrasts['within_con'] = glm.compute_contrast(np.array([1, 0]), output_type='all')
    t1 = time()
    contrasts['within_pat'] = glm.compute_contrast(np.array([0, 1]), output_type='all')
    t2 =  time()
    contrasts['between'] = glm.compute_contrast(np.array([1, -1]), output_type='all')
    print('within groups and between group contrasts took {:.2f}, {:.2f} and {:.2f}s'.format(t1-t0, t2-t1, time()-t2))
    n_voxels = np.sum(nilearn.image.get_data(glm.masker_.mask_img_))
    params = glm.get_params()
    return contrasts, n_voxels, params

def threshold_contrast(contrast, height_control='fpr', alpha=0.001, cluster_threshold=10):
    """ cluster threshold contrast at alpha with height_control method for multiple comparisons """
    thresholded_img, thresh = threshold_stats_img(
        contrast, alpha=alpha, height_control=height_control, cluster_threshold=cluster_threshold)
    cluster_table = get_clusters_table(
        contrast, stat_threshold=thresh, cluster_threshold=cluster_threshold,
        two_sided=True, min_distance=5.0)
    return thresholded_img, thresh, cluster_table

def create_within_group_mask(subroi_glm_results):
    """ create within group masks to use for between group contrasts to improve SNR """
    con_img, con_thr, c_table = threshold_contrast(subroi_glm_results['first_pass', 'contrasts']['within_con']['z_score'], cluster_threshold=100)
    con_mask = binarize_img(con_img, threshold=con_thr)
    pat_img, pat_thr, c_table = threshold_contrast(subroi_glm_results['first_pass', 'contrasts']['within_pat']['z_score'], cluster_threshold=100)
    pat_mask = binarize_img(pat_img, threshold=pat_thr)
    mask = nilearn.masking.intersect_masks([con_mask, pat_mask], threshold=0, connected=False)
    return mask, con_mask, pat_mask


def run_second_level(subjs, metrics, subrois, args):
    """ Run second level analysis """
    design_matrix = create_design_matrix(subjs)

    glm_results = dict()
    for metric,subroi in itertools.product(metrics,subrois):
        print('Starting 2nd level analysis for '+subroi+' subroi.')
        t0 = time()
        glm_results[subroi] = dict()

        t_fp = time()
        contrasts, n_voxels, params = perform_second_level_analysis(subroi, metric, design_matrix, args=args)
        glm_results[subroi]['first_pass','contrasts'], glm_results[subroi]['n_voxels'], glm_results[subroi]['first_pass','params'] = contrasts, n_voxels, params
        print('{} first pass in {:.2f}s'.format(subroi,time()-t_fp))

        t_wmask = time()
        within_group_mask, con_mask, pat_mask = create_within_group_mask(glm_results[subroi])
        print('created within groups mask in {:.2f}s'.format(time()-t_wmask))

        t_sp = time()
        contrasts, n_voxels, params = perform_second_level_analysis(subroi, metric, design_matrix, args=args, masks=[within_group_mask])
        glm_results[subroi]['second_pass','contrasts'], glm_results[subroi]['n_voxels'], glm_results[subroi]['second_pass','params'] = contrasts, n_voxels, params
        print('{} second pass in {:.2f}s'.format(subroi,time()-t_sp))

        # Correcting the p-values for multiple testing and taking negative logarithm
        #neg_log_pval = nilearn.image.math_img("-np.log10(np.maximum(1, img * {}))"
        #                .format(str(glm_results[subroi]['n_voxels'])),
        #                img=glm_results[subroi]['contrasts']['between']['p_value'])
        #glm_results[subroi]['neg_log_pval'] = neg_log_pval

        t_thr = time()
        for pss in ['first_pass', 'second_pass']:
            glm_results[subroi][(pss,'fpr',0.001,'thresholded_img')], \
                glm_results[subroi][(pss,'fpr',0.001,'thresh')], \
                glm_results[subroi][(pss,'fpr',0.001,'cluster_table')] = threshold_contrast( \
                                glm_results[subroi][pss,'contrasts']['between']['z_score'])

            print(' '.join([subroi,pss,'clusters at p<0.001 uncorrected:']))
            print(glm_results[subroi][(pss,'fpr',0.001,'cluster_table')])

            glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresholded_img')], \
                glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresh')], \
                glm_results[subroi][(pss,'fdr',args.fdr_threshold,'cluster_table')] = threshold_contrast( \
                                glm_results[subroi][pss,'contrasts']['between']['z_score'], height_control='fdr', alpha=args.fdr_threshold)

            print(' '.join([subroi,pss,'clusters at p<{:.2f} FDR corrected:'.format(args.fdr_threshold)]))
            print(glm_results[subroi][(pss,'fdr',args.fdr_threshold,'cluster_table')])

        print('Thresholding and clustering took {:.2f}s'.format(time()-t_thr))

        if args.plot_figs:
            t_plt = time()
            for pss in ['first_pass', 'second_pass']:
                fig = plt.figure(figsize=[16,4])
                ax1 = plt.subplot(1,2,1)
                plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fpr',0.001,'thresh')],
                                axes=ax1, title='_'.join([pss,subroi,'contrast_fpr0001']))
                ax2 = plt.subplot(1,2,2)
                plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresh')],
                                axes=ax2, title='_'.join([pss,subroi,'contrast_fdr{:03d}'.format(int(args.fdr_threshold*100))]))
                print('{} plotting took {:.2f}s'.format(subroi,time()-t_plt))

                if args.save_figs:
                    plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fpr',0.001,'thresh')],
                    output_file=os.path.join(args.out_dir,subroi+'_'+pss+'_contrast_fpr0001.pdf'))

        print('Finished 2nd level analysis for '+subroi+' ROI in {:.2f}s'.format(time()-t0))

    # savings
    if args.save_outputs:
        suffix = '_'+metric
        if args.min_time_after_scrubbing!=None:
            suffix += '_minLength'+str(int(args.min_time_after_scrubbing*10))
        if args.use_fspt_mask:
            suffix += '_fsptMask'
        if args.use_cortical_mask:
            suffix += '_corticalMask'
        today = datetime.datetime.now().strftime("%Y%m%d")
        suffix += '_'+today
        with gzip.open(os.path.join(args.out_dir,'glm_results'+suffix+'.pkl.gz'), 'wb') as of:
            pickle.dump(glm_results, of)

    return glm_results


def compute_voi_corr(subjs, seeds = ['Acc', 'dPut', 'vPut'], vois = ['lvPFC_R', 'lPFC_R', 'dPFC_L'], args=None):
    """ compute correlation between seed and VOI for each pathway, to extract p-values, effect size, etc. """
    dfs = []
    fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        for subj in subjs:
            if 'control' in subj:
                cohort = 'controls'
            else:
                cohort = 'patients'
            for seed,voi in zip(seeds, vois):
                # load correlation map
                fname = '_'.join([subj, metric, fwhm, atlas, seed, 'ns_sphere_seed_to_voxel_corr.nii'])
                corr_map = load_img(os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/seed_not_smoothed',
                                    metric, fwhm, seed, cohort, fname))
                # load voi mask
                #voi_mask = load_img(os.path.join(proj_dir, 'postprocessing', subj, 'spm/masks', 'local_'+'_'.join([voi, metric])+'.nii'))
                voi_mask = load_img(os.path.join(proj_dir, 'postprocessing', 'SPM', 'seeds_and_rois', voi+'_sphere3mm.nii.gz'))
                voi_mask = resample_to_img(voi_mask, corr_map, interpolation='nearest')

                # extract correlations
                voi_corr = corr_map.get_fdata().copy() * voi_mask.get_fdata().copy()
                avg_corr = np.mean(voi_corr[voi_corr!=0])
                df_line = {'subj':subj, 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 'cohort':cohort, 'pathway':'_'.join([seed,voi]), 'corr':avg_corr}
                dfs.append(df_line)
    df_voi_corr = pd.DataFrame(dfs)
    return df_voi_corr


def plot_voi_corr(df_voi_corr, seeds = ['Acc', 'dPut', 'vPut'], vois = ['lvPFC_R', 'lPFC_R', 'dPFC_L'], args=None):
    """ violinplots of FC in pahtways """
    colors = ['lightgrey', 'darkgrey']
    sbn.set_palette(colors)
    plt.rcParams.update({'font.size': 20, 'axes.linewidth':2})
    ylim = [-0.15, 0.3]
    fig = plt.figure(figsize=[12,6])
    df_voi_corr['corr'] = df_voi_corr['corr'] / 880.
    df_voi_corr['corr'].loc[df_voi_corr['corr']>1] = 1
    df_voi_corr['corr'].loc[df_voi_corr['corr']<-1] = -1
    #plt.subplot(1,2,1)
    #sbn.violinplot(data=df_voi_corr, y='corr', x='pathway', hue='cohort', orient='v', split=True, scale_hue=True,
    #               inner='quartile', dodge=True, width=0.8, cut=0)
    for i,(seed,voi) in enumerate(zip(seeds, vois)):
      ax = plt.subplot(1,len(seeds),i+1)
      sbn.barplot(data=df_voi_corr[df_voi_corr['pathway']=='_'.join([seed,voi])], y='corr', x='pathway', hue='cohort', orient='v')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2)
      if i==len(seeds)-1:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
      else:
        ax.get_legend().set_visible(False)
      plt.tight_layout()

    if args.save_figs:
        figname = 'seed_to_voi_FC_3pathways_sphere3mm.svg'
        plt.savefig(os.path.join(proj_dir, 'img', figname))
        #plt.savefig(os.path.join('/home/sebastin/tmp/', figname))


def print_voi_stats(df_voi_corr, seeds = ['Acc', 'dPut', 'vPut'], vois = ['lvPFC_R', 'lPFC_R', 'dPFC_L'], args=None):
    """ print seed to VOI stats """
    print('Seed to VOI statistics:\n-------------------------')
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
        out = dict()
        for seed,voi in zip(seeds, vois):
            key = '_'.join([seed, voi])
            df_con = df_voi_corr.loc[ (df_voi_corr['cohort']=='controls')
                                    & (df_voi_corr['atlas']==atlas)
                                    & (df_voi_corr['metric']==metric)
                                    & (df_voi_corr['pathway']==key) ]
            df_pat = df_voi_corr.loc[ (df_voi_corr['cohort']=='patients')
                                    & (df_voi_corr['atlas']==atlas)
                                    & (df_voi_corr['metric']==metric)
                                    & (df_voi_corr['pathway']==key) ]
            t,p = scipy.stats.ttest_ind(df_con['corr'], df_pat['corr'])
            d = cohen_d(df_con['corr'], df_pat['corr'])
            print("{} {} {} {}   T={:.3f}   p={:.3f}   cohen's d={:.2f}".format(atlas,metric,fwhm,key,t,p,d))


def compute_non_parametric_within_groups_mask(con_flist, pat_flist, design_matrix, masks, args):
    """ reconstruct within-group masks using non-parametric inference """
    # controls
    imgs, masker = mask_imgs(con_flist, masks=masks, args=args)
    neg_log_pvals_within_con = non_parametric_inference(list(np.sort(con_flist)),
                                 design_matrix=pd.DataFrame(np.ones([len(con_flist),1])),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_within_group, mask=masker, n_jobs=10, verbose=1)
    within_con_mask = binarize_img(neg_log_pvals_within_con, threshold=1.3) # -log(p)>1.3 corresponds to p<0.05 (p are already bonferroni corrected)
    # patients
    imgs, masker = mask_imgs(pat_flist, masks=masks, args=args)
    neg_log_pvals_within_pat = non_parametric_inference(list(np.sort(pat_flist)),
                                 design_matrix=pd.DataFrame(np.ones([len(pat_flist),1])),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_within_group, mask=masker, n_jobs=10, verbose=1)
    within_pat_mask = binarize_img(neg_log_pvals_within_pat, threshold=1.3) # -log(p)>1.3 corresponds to p<0.05 (p are already bonferroni corrected)
    within_groups_mask = nilearn.masking.intersect_masks([within_con_mask, within_pat_mask], threshold=0, connected=False)
    return within_groups_mask


def non_parametric_analysis(subjs, seed, metric, pre_metric, masks=[], args=None):
    """ Performs a non-parametric inference """
    post_metric = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/', pre_metric, metric, post_metric)

    # create imgs file list
    con_flist = glob.glob(os.path.join(in_dir, seed, 'controls', '*'))
    pat_flist = glob.glob(os.path.join(in_dir, seed, 'patients', '*'))
    pat_flist = [pf for pf in pat_flist if 'sub-patient16' not in pf]
    flist = np.hstack([con_flist, pat_flist])

    design_matrix = create_design_matrix(subjs)

    if args.use_SPM_mask:
        # template masking (& SPM within-group masks) need to be set manually via commenting others
        mask = os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/', metric, seed, 'scrub_out_1/'+seed+'_fl_0001unc_005fwe.nii.gz') # within-group
        #fspt_mask = binarize_img(os.path.join(proj_dir, 'utils', 'Larger_FrStrPalThal_schaefer100_tianS1MNI_lps_mni.nii.gz')) # fronto-striato-pallido-thalamic (fspt) mask
        #mask = os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/', metric, seed, 'scrub_out_1/'+seed+'_fl_0001unc_and_FrStrPalThal_mask.nii') # fspt & within-group
        #gm_mask = os.path.join(proj_dir, 'utils', 'schaefer_cortical.nii.gz') # gray matter only
    else:
        # create within-group masks non-parametrically
        mask = compute_non_parametric_within_groups_mask(con_flist, pat_flist, design_matrix, masks, args)

    # between-groups non-parametric inference
    neg_log_pvals_permuted_ols_unmasked = \
        non_parametric_inference(list(np.sort(flist)),
                                 design_matrix=pd.DataFrame(design_matrix['con'] - design_matrix['pat']),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_between_group, mask=mask, n_jobs=10, verbose=1)

    return neg_log_pvals_permuted_ols_unmasked, mask

def plot_non_param_maps(neg_log_pvals, seed, suffix, args=None):
    """ plot and save outputs of the non-parametric analysis """
    if args.plot_figs:
        plot_stat_map(neg_log_pvals, threshold=0.2, colorbar=True, title=' '.join([seed,suffix]), draw_cross=False) #cut_coords=[-24,55,34]
    if args.save_figs:
        plot_stat_map(neg_log_pvals, threshold=0.2, colorbar=True, title=' '.join([seed,suffix]), draw_cross=False,
                      output_file=os.path.join(proj_dir, 'img', '_'.join([seed,suffix])+'.pdf')) #cut_coords=[-24,55,34]


def compute_FC_within_masks(subjs, np_results, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ compute FC within masks used for the between-group analysis """
    dfs = []
    fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        for subj in subjs:
            if 'control' in subj:
                cohort = 'controls'
            else:
                cohort = 'patients'
            for seed in seeds:
                # load correlation map
                fname = '_'.join([subj, metric, fwhm, atlas, seed, 'ns_sphere_seed_to_voxel_corr.nii'])
                corr_map = load_img(os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/seed_not_smoothed',
                                    metric, fwhm, seed, cohort, fname))
                voi_mask = resample_to_img(np_results[seed]['within_group_mask'], corr_map, interpolation='nearest')

                # extract correlations
                voi_corr = corr_map.get_fdata().copy() * voi_mask.get_fdata().copy()
                for corr in np.ravel(voi_corr[voi_corr!=0]):
                    df_line = {'subj':subj, 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 'cohort':cohort, 'seed':seed, 'corr':corr}
                    dfs.append(df_line)
    df_mask_corr = pd.DataFrame(dfs)
    return df_mask_corr

def plot_within_mask_corr(df_mask_corr, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ bar plots of FC in pahtways """
    colors = ['lightgrey', 'darkgrey']
    sbn.set_palette(colors)
    plt.rcParams.update({'font.size': 20, 'axes.linewidth':2})
    ylim = [-0.15, 0.3]
    fig = plt.figure(figsize=[12,6])
    df_mask_corr['corr'] = df_mask_corr['corr'] / 880.
    df_mask_corr['corr'].loc[df_mask_corr['corr']>1] = 1
    df_mask_corr['corr'].loc[df_mask_corr['corr']<-1] = -1

    for i,seed in enumerate(seeds):
      ax = plt.subplot(1,len(seeds),i+1)
      #sbn.barplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v')
      #sbn.swarmplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v', dodge=True, size=0.1)
      sbn.violinplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v', split=True, scale_hue=True,
                     inner='quartile', dodge=True, width=0.8, cut=1)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2)
      if i==len(seeds)-1:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
      else:
        ax.get_legend().set_visible(False)
      plt.tight_layout()

    if args.save_figs:
        figname = 'seed_to_mask_corr_3seeds.svg'
        plt.savefig(os.path.join(proj_dir, 'img', figname))



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
    parser.add_argument('--use_gm_mask', default=False, action='store_true', help='use a gray matter mask to reduce the space of the second level analysis')
    parser.add_argument('--use_fspt_mask', default=False, action='store_true', help='use a fronto-striato-pallido-thalamic mask to reduce the space of the second level analysis')
    parser.add_argument('--use_cortical_mask', default=False, action='store_true', help='use a cortical mask to reduce the space of the second level analysis')
    parser.add_argument('--unzip_corr_maps', default=False, action='store_true', help='unzip correlation maps for use in SPM (not necessary if only nilearn analysis)')
    parser.add_argument('--min_time_after_scrubbing', default=None, type=float, action='store', help='minimum time (in minutes) needed per subject needed to be part of the analysis (after scrubbing (None=keep all subjects))')
    parser.add_argument('--prep_fsl_randomise', default=False, action='store_true', help='Prepare 4D images for running FSL randomise')
    parser.add_argument('--create_sphere_within_cluster', default=False, action='store_true', help='export sphere around peak within VOI cluster in prep for DCM analysis')
    parser.add_argument('--brain_smoothing_fwhm', default=8., type=none_or_float, action='store', help='brain smoothing FWHM (default 8mm as in Harrison 2009)')
    parser.add_argument('--fdr_threshold', type=float, default=0.05, action='store', help="cluster level threshold")
    parser.add_argument('--compute_voi_corr', default=False, action='store_true', help="compute seed to VOI correlation and print stats")
    parser.add_argument('--non_parametric_analysis', default=False, action='store_true', help="compute between group analysis using non-parametric inference")
    parser.add_argument('--use_SPM_mask', default=False, action='store_true', help="use within-group masks generated from SPM (one-tailed)")
    parser.add_argument('--two_sided_within_group', default=False, action='store_true', help="use two-tailed test to recreate within-group mask with parametric inference")
    parser.add_argument('--two_sided_between_group', default=False, action='store_true', help="use two-tailed test for between-group analysis with parametric inference")
    parser.add_argument('--n_perm', type=int, default=5000, action='store', help="number of permutation for non-parametric analysis")
    parser.add_argument('--within_mask_corr', default=False, action='store_true', help="compute FC within group masks and plot")
    args = parser.parse_args()

    if args.subj!=None:
        subjs = [args.subj]
    else:
        subjs = pd.read_table(os.path.join(code_dir, 'subject_list_all.txt'), names=['name'])['name']

    # options
    atlases= ['Harrison2009'] #['schaefer100_tianS1', 'schaefer200_tianS2', 'schaefer400_tianS4'] #schaefer400_harrison2009
    #metrics = ['detrend_filtered', 'detrend_gsr_filtered']
    pre_metric = 'seed_not_smoothed' #'unscrubbed_seed_not_smoothed'
    metrics = ['detrend_gsr_filtered_scrubFD05'] #'detrend_gsr_smooth-6mm', 'detrend_gsr_filtered_scrubFD06'

    args.atlases = atlases
    args.pre_metric = pre_metric
    args.metrics = metrics

    #TODO: in_dir must be tailored to the atlas. ATM everything is put in Harrison2009 folder
    args.in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/', args.seed_type+'Rep', pre_metric)

    seeds = list(seed_loc.keys()) #['AccL', 'AccR', 'dCaudL', 'dCaudR', 'dPutL', 'dPutR', 'vPutL', 'vPutR', 'vCaudSupL', 'vCaudSupR', 'drPutL', 'drPutR']
    subrois = np.unique([seed[:-1] for seed in seeds])#['Acc', 'dCaud', 'dPut', 'vPut', 'drPut']


    seedfunc = {'Harrison2009':sphere_seed_to_voxel,
            'TianS4':seed_to_voxel}

    # First remove subjects without enough data
    if args.min_time_after_scrubbing != None:
        subjs, revoked = get_subjs_after_scrubbing(subjs, metrics, min_time=args.min_time_after_scrubbing)
    else:
        revoked=[]
    args.revoked=revoked

    # Then process data
    if args.compute_seed_corr:
        for atlas in atlases:
            Parallel(n_jobs=args.n_jobs)(delayed(seedfunc[args.seed_type])(subj,seeds,metrics,atlases,args) for subj in subjs)

    if args.unzip_corr_maps:
        unzip_correlation_maps(subjs, metrics, atlases, seeds, args)

    if args.merge_LR_hemis:
        in_fnames = merge_LR_hemis(subjs, subrois, metrics, seed_type=str(seedfunc[args.seed_type]), args=args)

        # can only prep fsl with in_fnames from merge_LR_hemis
        if args.prep_fsl_randomise:
            prep_fsl_randomise(in_fnames, subrois, metrics, args)

    if args.run_second_level:
        out_dir = os.path.join(proj_dir, 'postprocessing', 'glm', pre_metric)
        os.makedirs(out_dir, exist_ok=True)
        args.out_dir = out_dir
        glm_results = run_second_level(subjs, metrics, subrois, args)

    if args.compute_voi_corr:
        df_voi_corr = compute_voi_corr(subjs, args=args)
        print_voi_stats(df_voi_corr, args=args)
        plot_voi_corr(df_voi_corr, args=args)

    if args.non_parametric_analysis:
        suffix = 'non_parametric'
        if args.two_sided_within_group:
            suffix += '_within2tailed'
        if args.two_sided_between_group:
            suffix += '_between2tailed'

        np_results = dict()
        for seed,metric in itertools.product(subrois, metrics):
            neg_log_pvals, mask = non_parametric_analysis(subjs, seed, metric, pre_metric, args=args)
            plot_non_param_maps(neg_log_pvals, seed, suffix, args)
            np_results[seed] = {'neg_log_pvals':neg_log_pvals, 'within_group_mask':mask}

        if args.save_outputs:
            with gzip.open(os.path.join(proj_dir, 'postprocessing', 'non_parametric', suffix+'.pkl.gz'), 'wb') as f:
                pickle.dump(np_results,f)

    if args.within_mask_corr:
        df_mask_corr = compute_FC_within_masks(subjs, np_results, args=args)
        if args.plot_figs:
            plot_within_mask_corr(df_mask_corr, args=args)
