import argparse
import bct
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
from nilearn.image import load_img
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import numpy as np
import os
import pickle
import pandas as pd
import scipy
from scipy.io import loadmat
import sklearn
from sklearn.decomposition import PCA
import statsmodels
from statsmodels.stats import multitest
import sys
import time
from time import time

proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'

sys.path.insert(0, os.path.join(code_dir))
import qsiprep_analysis
importlib.reload(qsiprep_analysis)
import atlaser
importlib.reload(atlaser)
from atlaser import Atlaser

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)
subjs = pd.read_table(os.path.join(code_dir, 'subject_list_all.txt'), names=['name'])['name']


xls_fname = 'P2253_Data_Master-File.xlsx' #'P2253_OCD_Data_Pre-Post-Only.xlsx' #'P2253_YBOCS.xlsx'


def create_dataframes(args):
    """ load XLS master file and currate into controls and patients pandas dataframes  """
    xls = pd.read_excel(os.path.join(proj_dir, 'data', xls_fname), sheet_name=['OCD Patients', 'Healthy Controls'])

    df_pat = xls['OCD Patients'][['Participant_ID', 'Pre/Post/6mnth', 'Age', 'Gender(F=1,M=2)', 'Handedness(R=1,L=2)', 'YBOCS_Total', 'OBQ_Total', 'HAMA_Total', 'MADRS_Total', 'OCIR_Total', 'Anx_total', 'Dep_Total', 'FSIQ-4_Comp_Score']]
    df_pat = df_pat[df_pat['Pre/Post/6mnth']=='Pre'][['Participant_ID', 'Age', 'Gender(F=1,M=2)', 'Handedness(R=1,L=2)', 'YBOCS_Total', 'OBQ_Total', 'HAMA_Total', 'MADRS_Total', 'OCIR_Total', 'Anx_total', 'Dep_Total', 'FSIQ-4_Comp_Score']]

    df_con = xls['Healthy Controls'][['Participant_ID', 'Age', 'Gender(F=1;M=2)', 'Handedness(R=1,L=2)', 'YBOCS_Total', 'OBQ_Total', 'HAMA_Total', 'MADRS_Total', 'OCIR_Total', 'Anx_total', 'Dep_Total', 'FSIQ-4_Comp_Score']]
    df_con['Gender(F=1;M=2)'] = df_con['Gender(F=1;M=2)'].replace([0], 2)

    # sort alphabetically by subject ID
    df_con['subj'] = ['sub-control{:2s}'.format(s[-2:]) for s in df_con.Participant_ID]
    df_pat['subj'] = ['sub-patient{:2s}'.format(s.split('_')[0][-2:]) for s in df_pat.Participant_ID]
    df_con.rename(columns={'Gender(F=1;M=2)':'Gender(F=1,M=2)'}, inplace=True)
    df_pat.sort_values(by=['subj'], inplace=True)
    df_con.sort_values(by=['subj'], inplace=True)

    # savings
    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'df_con.pkl'), 'wb') as f:
            pickle.dump(df_con, f)
        with open(os.path.join(proj_dir, 'postprocessing', 'df_pat.pkl'), 'wb') as f:
            pickle.dump(df_pat, f)

    return df_con.reset_index(drop=True), df_pat.reset_index(drop=True)



def print_demographics(df_con, df_pat, revoked=['sub-patient16', 'sub-patient35'], \
                            fields=['Age', 'Gender(F=1,M=2)', 'Handedness(R=1,L=2)', \
                                    'YBOCS_Total', 'OBQ_Total', 'HAMA_Total', \
                                    'MADRS_Total', 'OCIR_Total', 'FSIQ-4_Comp_Score']):
    """ display demographic data for controls and patients """
    df_con.drop(index=np.where(df_con.subj.isin(revoked))[0], inplace=True)
    df_pat.drop(index=np.where(df_pat.subj.isin(revoked))[0], inplace=True)
    print('CONTROLS: n={}'.format(len(df_con)))
    print(df_con[fields].aggregate(['mean', 'std']))

    print('\n\nPATIENTS: n={}'.format(len(df_pat)))
    print(df_pat[fields].aggregate(['mean', 'std']))

    print('\n\n\n{:20} {:40} {:20}'.format('', 'Controls', 'Patients'))
    for field in fields:
        t,p = scipy.stats.ttest_ind(df_con[field], df_pat[field])
        print('\n{:20} {:.2f} ({:.2f}) {:20} {:.2f} ({:.2f}) \t\t p={:.4f}'.format(\
                field, df_con[field].mean(), df_con[field].std(), \
                '', df_pat[field].mean(), df_pat[field].std(), p))

def get_dcm_results():
    """ import pathway weigth from individual subjects DCMs """
    # store effective connectivity in a dataframe

    f = h5py.File(os.path.join(proj_dir, 'postprocessing', 'DCM', 'AccOFC_PutPFC_vPutdPFC', 'GCM.mat'), 'r');
    GCM = f['GCM']
    tmp = []
    for i in range(np.shape(GCM)[0]):
        subj = subjs[i]
        A = f[GCM[i][0]]['Ep']['A'][:]
        tmp.append({'subj':subj, 'Acc-OFC':A[0,1], 'OFC-Acc':A[1,0],
                                 'Put-PFC':A[2,3], 'PFC-Put':A[3,2],
                                 'vPut-dPFC':A[4,5], 'dPFC-vPut':A[5,4]})
    """
    # start with Acc-OFC
    f = h5py.File(os.path.join(proj_dir, 'postprocessing', 'DCM', 'AccOFC', 'GCM.mat'), 'r');
    GCM = f['GCM']
    tmp = []
    for i in range(np.shape(GCM)[0]):
        subj = subjs[i]
        A = f[GCM[i][0]]['Ep']['A'][:]
        tmp.append({'subj':subj, 'Acc-OFC':A[0,1], 'OFC-Acc':A[1,0]})

    # Then Put-PFC
    f = h5py.File(os.path.join(proj_dir, 'postprocessing', 'DCM', 'PutPFC', 'GCM.mat'), 'r');
    GCM = f['GCM']
    if (np.shape(GCM)[0] != len(tmp)):
        error('Number of subjects in GCM of Put-PFC looks different than Acc-OFC')
    for i in range(np.shape(GCM)[0]):
        subj = subjs[i]
        A = f[GCM[i][0]]['Ep']['A'][:]
        tmp[i]['Put-PFC'] = A[0,1]
        tmp[i]['PFC-Put'] = A[1,0]

    # Then vPut-dPFC
    f = h5py.File(os.path.join(proj_dir, 'postprocessing', 'DCM', 'vPutdPFC', 'GCM.mat'), 'r');
    GCM = f['GCM']
    if (np.shape(GCM)[0] != len(tmp)/2):
        error('Number of subjects in GCM of vPut-dPFC looks different than Acc-OFC')
    for i in range(np.shape(GCM)[0]):
        subj = subjs[i]
        A = f[GCM[i][0]]['Ep']['A'][:]
        tmp[i]['vPut-dPFC'] = A[0,1]
        tmp[i]['dPFC-vPut'] = A[1,0]
    """
    df_dcm = pd.DataFrame.from_dict(tmp)
    return df_dcm

def print_ybocs_dcm_relation(df):
    print('\n\nClinical relation to DCM weights:')
    behavs = ['YBOCS_Total', 'OBQ_Total', 'HAMA_Total', 'MADRS_Total', 'OCIR_Total']
    checklist_5dims = [] #TODO if deemed necessary
    for i,e in enumerate(['Acc-OFC', 'OFC-Acc', 'Put-PFC', 'PFC-Put', 'vPut-dPFC', 'dPFC-vPut']):
        for j,y in enumerate(np.concatenate([behavs, checklist_5dims])):
            r,p = scipy.stats.pearsonr(df[e], df[y])
            print('{:20}  {:25}  r={:.3f}  p={:.3f}'.format(e, y, r, p))

def compute_fc(args, seeds=['AccR', 'dPutR', 'vPutL'], vois=['OFC', 'PFC', 'dPFC_L']):
    """ compute correlation coefficient between VOI clusters from SPM/DCM """
    data = []
    for i,subj in enumerate(subjs):
        row = [subj]
        for seed,voi in zip(seeds,vois):
            seed_ts = scipy.io.loadmat(os.path.join(proj_dir, 'postprocessing', subj, 'spm/cluster', 'VOI_'+seed+'_1.mat'))['Y'].ravel()
            voi_ts = scipy.io.loadmat(os.path.join(proj_dir, 'postprocessing', subj, 'spm/cluster', 'VOI_'+voi+'_1.mat'))['Y'].ravel()
            r,p = scipy.stats.pearsonr(seed_ts, voi_ts)
            row.append(r)
        data.append(row)
        print("{} FC computed".format(subj))
    df = pd.DataFrame(data, columns=['subj', 'AccOFC', 'PutPFC', 'vPutdPFC'])
    return df

def print_ybocs_fc_relation(df):
    print('\n\nClinical relation to FC in striato-frontal systems:')
    behavs = ['YBOCS_Total', 'OBQ_Total', 'HAMA_Total', 'MADRS_Total', 'OCIR_Total']
    checklist_5dims = [] #TODO if deemed necessary
    for i,e in enumerate(['AccOFC', 'PutPFC', 'vPutdPFC']):
        for j,y in enumerate(np.concatenate([behavs, checklist_5dims])):
            r,p = scipy.stats.pearsonr(df[e], df[y])
            print('{:20}  {:25}  r={:.3f}  p={:.3f}'.format(e, y, r, p))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--print_demographics', default=False, action='store_true', help='plot demographic *table*')
    parser.add_argument('--print_ybocs_dcm', default=False, action='store_true', help='Y-BOCS to DCM weight relation')
    parser.add_argument('--print_ybocs_fc', default=False, action='store_true', help='Y-BOCS to SPM FC relation')
    args = parser.parse_args()

    revoked=['sub-patient14', 'sub-patient15', 'sub-patient16', 'sub-patient29', 'sub-patient35', 'sub-patient51']

    df_con, df_pat = create_dataframes(args)

    if args.print_demographics:
        print_demographics(df_con, df_pat, revoked)

    # relation to FC/SPM
    if args.print_ybocs_fc:
        df_fc = compute_fc(args)
        print_ybocs_fc_relation(df_pat.merge(df_fc, how='inner'))

    # relation to DCM
    if args.print_ybocs_dcm:
        df_dcm = get_dcm_results()
        print_ybocs_dcm_relation(df_pat.merge(df_dcm, how='inner'))