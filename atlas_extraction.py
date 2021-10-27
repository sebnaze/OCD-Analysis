# Extract timeseries of brain regions based on atlas and 
# fMRI preprocessing method.
# 
# Original author: Luke Hearne
# Modified by: Sebastien Naze
#
# 2021 - Clinical Brain Networks - QIMR Berghofer  
###########################################################
import h5py
from joblib import Parallel, delayed
import os
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
from time import time

# global variables
# paths
# change this when operating from lucky vs. hpc
working_path = '/home/sebastin/working/lab_lucac/'
lukeH_proj_dir = working_path+'lukeH/projects/OCDbaseline/'
sebN_proj_dir = working_path+'sebastiN/projects/OCDbaseline/'
lukeH_deriv_dir = lukeH_proj_dir+'data/derivatives/post-fmriprep-fix/'
sebN_deriv_dir = sebN_proj_dir+'data/derivatives/post-fmriprep-fix/'
parc_dir = working_path+'shared/parcellations/qsirecon_atlases_with_subcortex/'
#scratch_dir = proj_dir+'data/scratch/nilearn/'

# task
task_list = ['rest']

# preprocessed image space
img_space = 'MNI152NLin2009cAsym'

# denoising label (no smoothing needed)
denoise_label = {
    'rest': ['detrend_filtered_scrub', 'detrend_gsr_filtered_scrub']}

# parcellation nifti files used in timeseries extraction
ts_parc_dict = {'Schaefer100_TianS1': parc_dir+'schaefer100_tianS1MNI_lps_mni.nii.gz',
                'Schaefer200_TianS2': parc_dir+'schaefer200_tianS2MNI_lps_mni.nii.gz',
                'Schaefer400_TianS4': parc_dir+'schaefer400_tianS4MNI_lps_mni.nii.gz'
                }

# lists of parcellations to create fc matrices from
# these will match the above dict in name
#con_cortical_parcs = ['Schaefer2018_100_7', 'Schaefer2018_200_7',
#                      'Schaefer2018_300_7', 'Schaefer2018_400_7']
#con_subcortical_parcs = ['Tian_S1', 'Tian_S2', 'Tian_S3', 'Tian_S4']
parcs = ['Schaefer100_TianS1', 'Schaefer200_TianS2', 'Schaefer400_TianS4']

def extract_timeseries(subj):
    # Extracts atlas based timeseries using nilearn
    for task in task_list:
        for denoise in denoise_label[task]:
            for parc in ts_parc_dict.keys():
                print('\t', subj, '\t', task, ':', denoise, '\t', parc)

                # get subject / task / denoise specific BOLD nifti filename
                bold_file = (lukeH_deriv_dir+subj+'/func/'+subj+'_task-'+task
                             + '_space-'+img_space+'_desc-'+denoise+'.nii.gz')

                # use nilearn to extract timeseries
                masker = NiftiLabelsMasker(ts_parc_dict[parc])
                time_series = masker.fit_transform(bold_file)

                # save timeseries out as h5py file
                out_file = (sebN_deriv_dir+subj+'/timeseries/'+subj+'_task-'
                            + task+'_atlas-'+parc+'_desc-'+denoise+'.h5')
                hf = h5py.File(out_file, 'w')
                hf.create_dataset(parc, data=time_series)
                hf.close()


def generate_fc(subj):
    # generate the tian-schaefer connectivity matrices
    for task in task_list:
        for denoise in denoise_label[task]:
            for parc in parcs:

                # load the cortical timeseries
                in_file = (sebN_deriv_dir+subj+'/timeseries/'+subj+'_task-'
                           + task+'_atlas-'+parc+'_desc-'+denoise+'.h5')
                hf = h5py.File(in_file, 'r')
                time_series = hf[parc][:]
                hf.close()

                # perform fc (here, correlation only)
                fc = np.corrcoef(time_series.T)

                # save out
                out_file = (sebN_deriv_dir+subj+'/fc/'+subj+'_task-'+task
                            + '_atlas-'+parc+'_desc-correlation-'+denoise+'.h5')
                hf = h5py.File(out_file, 'w')
                hf.create_dataset(name='fc', data=fc)
                hf.close()

def process_subj(subj):
    for newdir in ['timeseries', 'fc']:
        os.makedirs(os.path.join(sebN_deriv_dir, subj, newdir), exist_ok=True)
    start = time()
    print(subj)
    extract_timeseries(subj)
    generate_fc(subj)
    finish = time()
    print(subj, ' time elapsed:', finish - start)


if __name__ == "__main__":
    # loop through everyone and run:
    subj_list = list(np.loadtxt('../subject_list.txt', dtype='str'))
    Parallel(n_jobs=20)(delayed(process_subj)(subj) for subj in subj_list)
        
