################################################################################
# Structure-Function analysis
#
# Author: Sebastien Naze
# QIMR Berghofer
# 2021
################################################################################

import argparse
import bct
import h5py
import itertools
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
import sklearn
import statsmodels
from statsmodels.stats import multitest
import sys

proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'

sys.path.insert(0, os.path.join(code_dir))
import qsiprep_analysis
import atlaser

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)
subjs = pd.read_table(os.path.join(code_dir, 'subject_list.txt'), names=['name'])['name']

cohorts = ['controls', 'patients']

# rois of the FrStrThal Atlases to exclude from analysis
excl_rois = { 'fspt':[], \
              'Fr':['Thal', 'Pal', 'Put', 'Caud', 'Acc'], \
              'StrTh':['PFC', 'OFC', 'Fr', 'FEF', 'ACC', 'Cing', 'PrC', 'ParMed'] }

# set here which edges to analyze based on the atlas
atlas_types = {'ocdOFClPFC_ocdAccdPut': { 'atlases' : ['ocdOFClPFC_ocdAccdPut'],
                                          'subrois' : [['AccSPM_R'], ['PutSPM_R']],
                                          'rois' : {'AccSPM_R': ['OFC'], 'PutSPM_R':['PFC']},
                                          'file_suffix' : {'SC': '_ocd', 'FC':'_ocd'} },#{'SC': '_harrison2009', 'FC':'_harrison2009_fwhm8'} },
               'schaefer400_tianS4':    { 'atlases' : ['schaefer400_tianS4'],
                                          'subrois' : None,
                                          'rois' : None, # None will compute SC-FC for all edges!
                                          'file_suffix' : {'SC':'', 'FC':''} },
               'schaefer400_harrison2009': { 'atlases' : ['schaefer400_harrison2009'],
                                             'subrois' : [['Acc'], ['dPut']],
                                             'rois' : {'Acc': ['Right_ContB_PFClv'], 'dPut':['Right_ContA_PFCl']},
                                             'file_suffix' : {'SC':'_ocd', 'FC':'_ocd'} } }

# plotting utils
plt_utl = {'controls':'blue',
           'patients':'orange'}

def inv_normal(x):
    """ rank-based inverse gaussian transformation """
    rank = scipy.stats.rankdata(x)
    p = rank / (len(rank)+1)
    y = scipy.stats.norm.ppf(p,0,1)
    return y.reshape(x.shape)

def mutual_info(x, y, bins=32, base=None):
    """ computes mutual information between connectvity matrices x and y """
    c_x = np.histogram(x, bins)[0]
    Hx = scipy.stats.entropy(c_x, base=base)
    c_y = np.histogram(y, bins)[0]
    Hy = scipy.stats.entropy(c_y, base=base)
    c_xy = np.histogram2d(x, y, bins)[0]
    Hxy = scipy.stats.entropy(c_xy.flatten(), base=base)
    mi = Hx + Hy - Hxy
    return mi

def compute_corr_mi(scs, fcs, zscore=False, keep_non_significant=True):
    """ compute correlation and mutual information between m x m x n structural (scs) and functional (fcs) matrices"""
    if fcs.shape[-1]!=scs.shape[-1] :
        print('ERROR: Not the same number of subjects in conns_sc and conns_fc!')
        return None
    nz_inds = scs.nonzero()
    new_scs = inv_normal(scs[nz_inds])
    scs[nz_inds] = new_scs
    rs = {'r':np.array([]), 'p':np.array([]), 'mi':np.array([]), 'scs':np.array([]), 'fcs':np.array([]), \
          'n':0, 'subjs_inds':np.array([], dtype=int)}
    for s in range(fcs.shape[-1]):
        sc = scs[:,:,s].flatten()
        fc = fcs[:,:,s].flatten()
        if zscore:
            sc = scipy.stats.zscore(sc)
            fc = scipy.stats.zscore(fc)
        if (len(sc)<2 or len(fc)<2):
            r,p = (np.NaN, np.NaN)
        else:
            r,p = scipy.stats.pearsonr(sc, fc)
        rs['r'] = np.append(rs['r'], r)
        rs['p'] = np.append(rs['p'], p)
        if keep_non_significant:
            rs['scs'] = np.append(rs['scs'], sc)
            rs['fcs'] = np.append(rs['fcs'], fc)
            rs['n'] += 1
            rs['subjs_inds'] = np.append(rs['subjs_inds'], int(s))
        else:
            if p<=0.05:
                rs['scs'] = np.append(rs['scs'], sc)
                rs['fcs'] = np.append(rs['fcs'], fc)
                rs['n'] += 1
                rs['subjs_inds'] = np.append(rs['subjs_inds'], int(s))
        mi = mutual_info(sc, fc, base=10)
        rs['mi'] = np.append(rs['mi'], mi)
    return rs

def scfc_corr(subnet, atlases, sc_metrics, fc_metrics, conns_sc, conns_fc, cohorts=['controls', 'patients'], \
              row_rois=None, col_rois=None, sc_threshold=None, zscore=False, keep_non_significant=True):
    """ Compute structure-function relation using pearson correlation and mutual information.
        (roi input is a list of ROI names (i.e. Acc) to perform `row-wise` analysis if needed) """
    corrs = dict( ((atlas,scm,fcm,coh),None) for atlas,scm,fcm,coh in itertools.product(atlases, sc_metrics, fc_metrics, cohorts) )
    for atlas, scm, fcm, coh in itertools.product(atlases, sc_metrics, fc_metrics, cohorts):
        atlazer = atlaser.Atlaser(atlas)

        # sort out mask indexing
        if (subnet != 'wholebrain'):
            node_inds, _ = qsiprep_analysis.get_fspt_Fr_node_ids(atlas, subctx=excl_rois[subnet]) #TODO: refactor get_fspt_Fr_node_ids to more generic
        else:
            node_inds = atlazer.node_ids.astype(int)

        # threshold SC
        if (sc_threshold != None):
            c, _ = qsiprep_analysis.threshold_connectivity(conns_sc[(atlas,scm,coh)][np.ix_(node_inds-1, node_inds-1)], \
                    quantile=sc_threshold)
        else:
            c = conns_sc[(atlas,scm,coh)][np.ix_(node_inds-1, node_inds-1)]

        # sort out ROI indexing after maksing
        if (row_rois!=None):
            row_rois_node_inds = atlazer.get_rois_node_indices(row_rois)
        else:
            row_rois_node_inds = node_inds
        if (col_rois!=None):
            col_rois_node_inds = atlazer.get_rois_node_indices(col_rois)
        else:
            col_rois_node_inds = node_inds

        row_ind = np.array([i for i,n in enumerate(node_inds) if n in row_rois_node_inds])
        col_ind = np.array([i for i,n in enumerate(node_inds) if n in col_rois_node_inds])
        if ((row_ind.size == 0) | (col_ind.size == 0)):
            print('ROI {} or {} not in atlas {} subnet {}, no output will be generated for this subnet'.format(
                row_rois, col_rois, atlas, subnet))
            break;

        # get matrices and compute correlation and MI
        scs = c[np.ix_(row_ind,col_ind)]
        fcs = conns_fc[atlas,fcm,coh][np.ix_(row_rois_node_inds-1,col_rois_node_inds-1)].copy()
        rs = compute_corr_mi(scs, fcs, zscore=zscore, keep_non_significant=keep_non_significant)
        corrs[atlas,scm,fcm,coh] = rs
    return corrs


def plot_scfc_distrib(corrs, scm='count_nosift', fcm='detrend_filtered', bins=10):
    """ Plot structure-function correlation/MI distribution and t-test results for each atlas """
    plt.figure(figsize=[16,10])
    gs = plt.GridSpec(2,3)
    for i,atlas in enumerate(atlases):
        plt.subplot(gs[0,i])
        plt.hist(corrs[atlas,scm,fcm,'controls']['r'], bins=bins, alpha=0.5)
        plt.hist(corrs[atlas,scm,fcm,'patients']['r'], bins=bins, alpha=0.5)
        plt.legend(['controls', 'patients'])
        t,p = scipy.stats.ttest_ind(corrs[atlas,scm,fcm,'controls']['r'], corrs[atlas,scm,fcm,'patients']['r'])
        plt.title('{}    t={:.2f}  p={:.2f}'.format(atlas,t,p))
        plt.xlabel('correlation')
        plt.ylabel('n_subjs')

        plt.subplot(gs[1,i])
        plt.hist(corrs[atlas,scm,fcm,'controls']['mi'], bins=bins, alpha=0.5)
        plt.hist(corrs[atlas,scm,fcm,'patients']['mi'], bins=bins, alpha=0.5)
        plt.legend(['controls', 'patients'])
        t,p = scipy.stats.ttest_ind(corrs[atlas,scm,fcm,'controls']['mi'], corrs[atlas,scm,fcm,'patients']['mi'])
        plt.title('{}    t={:.2f}  p={:.2f}'.format(atlas,t,p))
        plt.xlabel('mutual information')
        plt.ylabel('n_subjs')
    plt.show(block=False)

def plot_scfc_pvals(outp_scfc, subnet, subroi, atlases, sc_metrics, fc_metrics):
    """ plot bar plots of p-values """
    plt.figure(figsize=[16,4])
    for i,atlas in enumerate(atlases):
        plt.subplot(1,3,i+1)
        cnt=1
        for scm,fcm in itertools.product(sc_metrics, fc_metrics):
            plt.bar(cnt, outp_scfc[subnet,subroi,atlas,scm,fcm]['p'], label='{}-{}'.format(scm.split('_')[1], fcm.split('_')[1][:3]), alpha=0.7)
            cnt+=1
        plt.ylim([0,1])
        plt.axhline(y=0.05, linestyle='--', c='red')
        plt.legend()
        plt.title('{} - {} - {}'.format(subnet,subroi,atlas))
    plt.show(block=False)


def linregr(x,y):
    """ Perform linear regression between X and Y variables. Returns X, Y, and predicted Y.  """
    X = np.reshape(x, [-1,1])
    Y = np.reshape(y, [-1,1])
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(X,Y)
    Yp = lr.predict(X)
    return X,Y,Yp

def plot_scfc_corr(corrs, subnet, atlas, sc_metrics, fc_metrics):
    """ Plot SC w.r.t. FC with corelation coef, p-value and linear regression """
    n_sc = len(sc_metrics)
    n_fc = len(fc_metrics)
    ncols = n_sc + n_fc
    fig = plt.figure(figsize=[16,4])
    for i,scm in enumerate(sc_metrics):
        for j,fcm in enumerate(fc_metrics):
            ax = plt.subplot(1,ncols,i*n_sc+j+1)
            stats = np.array([])
            for coh in cohorts:
                rs = corrs[atlas,scm,fcm,coh]
                X, Y, Yp = linregr(rs['scs'], rs['fcs'])
                r, p = scipy.stats.pearsonr(rs['scs'], rs['fcs'])
                stats = np.concatenate([stats, [r,p,rs['n']]])
                ax.plot(X, Y, '.', markersize=5, color=plt_utl[coh])
                ax.plot(X, Yp, '-', linewidth=2, color=plt_utl[coh])
            titl = '{} - {} \n {} - {} \n'.format(subnet, atlas, scm, fcm) \
                 + 'r_con={:.3f}, p_con={:.3f}, n_con={}\nr_pat={:.3f}, p_pat={:.3f}, n_pat={}'.format(*stats)
            ax.set_title(titl)
    plt.show(block=False)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--output_suffix', type=str, default='_ocd', action='store', help='output suffix for saving correlations')
    parser.add_argument('--sc_threshold', type=float, default=None, action='store', help='threshold applied to structutal connectome [0-1]')
    parser.add_argument('--atlas_type', type=str, default='ocdOFClPFC_ocdAccdPut', action='store', help='VOI atlas, can be ocdOFClPFC_ocdAccdPut (default, i.e. Acc-OFC dPut-PFC), schaefer400_harrison2009, or schaefer400_tianS4')
    args = parser.parse_args()

    # options
    atlases = ['ocdOFClPFC_ocdAccdPut'] #['schaefer100_tianS1', 'schaefer200_tianS2', 'schaefer400_tianS4']
    fc_metrics = ['detrend_gsr_filtered', 'detrend_filtered']
    sc_metrics = ['count_sift'] #['count_sift', 'count_nosift']

    # load data
    with open(os.path.join(proj_dir, 'postprocessing',
                            'conns_SC'+atlas_types[args.atlas_type]['file_suffix']['SC']+'.pkl'), 'rb') as scf:
        conns_sc = pickle.load(scf)
    with open(os.path.join(proj_dir, 'postprocessing',
                            'conns_FC'+atlas_types[args.atlas_type]['file_suffix']['FC']+'.pkl'), 'rb') as fcf:
        conns_fc = pickle.load(fcf)

    # remove patients 28 and 55 from FC and SC (since failed)
    pt_inds, = np.where(['patient' in subj for subj in subjs])
    con_inds, = np.where(['control' in subj for subj in subjs])
    pt28_idx, = np.where(['patient28' in subj for subj in subjs[pt_inds]])
    for k,conn in conns_fc.items():
        if k[2]=='patients':
            #conns_fc[k] = np.delete(conn[:,:,:-1], pt28_idx, axis=-1)  <-- pt55 not processed
            conns_fc[k] = np.delete(conn, pt28_idx, axis=-1)

    # create summary dataframe
    scfc_summary_df = pd.DataFrame()
    for i,subj in enumerate(subjs):
        if ( ('patient28' in subj) | ('patient55' in subj) ):
            subjs.pop(i)
    #scfc_summary_df['subj'] = subjs


    # extract correlation and MI for each sub-network
    subnets = ['wholebrain', 'fspt', 'Fr', 'StrTh']
    subrois = atlas_types[args.atlas_type]['subrois']
    subnet_corrs = dict()
    for subnet, subroi in itertools.product(subnets,subrois):
        row_rois = np.array(subroi).flatten()
        col_rois = np.array([atlas_types[args.atlas_type]['rois'][roi] for roi in row_rois]).flatten()
        sbr = '_'.join(subroi)
        subnet_corrs[subnet,sbr] = scfc_corr(subnet, atlases, sc_metrics, fc_metrics, conns_sc, conns_fc,
                                         row_rois=row_rois, col_rois=col_rois, sc_threshold=args.sc_threshold)

    # save in summary dataframe
    for subnet,subroi,atlas,scm,fcm in itertools.product(subnets,subrois,atlases,sc_metrics,fc_metrics):
        if ((subnet_corrs[subnet,sbr][atlas,scm,fcm,'controls'] != None) and (subnet_corrs[subnet,sbr][atlas,scm,fcm,'patients'] != None)):
            sbr = '_'.join(subroi)
            df_ = pd.DataFrame()
            df_['subj'] = subjs.copy()
            df_['subnet'] = np.repeat(subnet, len(subjs))
            df_['subroi'] = np.repeat(sbr, len(subjs))
            df_['atlas'] = np.repeat(atlas, len(subjs))
            df_['sc_metric'] = np.repeat(scm, len(subjs))
            df_['fc_metric'] = np.repeat(fcm, len(subjs))
            df_['sc'] = np.repeat(np.NaN, len(subjs))
            df_['fc'] = np.repeat(np.NaN, len(subjs))
            inds = np.concatenate([subnet_corrs[subnet,sbr][atlas,scm,fcm,'controls']['subjs_inds'],
                                   subnet_corrs[subnet,sbr][atlas,scm,fcm,'patients']['subjs_inds']+len(con_inds)])
            df_['sc'].iloc[inds] = np.concatenate([subnet_corrs[subnet,sbr][atlas,scm,fcm,coh]['scs'] for coh in cohorts])
            df_['fc'].iloc[inds] = np.concatenate([subnet_corrs[subnet,sbr][atlas,scm,fcm,coh]['fcs'] for coh in cohorts])
            df_['r'] = np.concatenate([subnet_corrs[subnet,sbr][atlas,scm,fcm,coh]['r'] for coh in cohorts])
            df_['cohort'] = np.concatenate([np.repeat('controls', len(con_inds)), np.repeat('patients', len(pt_inds)-2)]) #<-- removed pt28 and 55
            scfc_summary_df = scfc_summary_df.append(df_, ignore_index=True)

    # save SC-FC correlations and MI
    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'corrs_SCFC'+args.output_suffix+'.pkl'), 'wb') as pf:
            pickle.dump(subnet_corrs, pf)
        with open(os.path.join(proj_dir, 'postprocessing', 'scfc_summary_df'+args.output_suffix+'.pkl'), 'wb') as pf:
            pickle.dump(scfc_summary_df, pf)


    # extract stats
    outp_scfc = dict( ( (subnet,'_'.join(subroi),atlas,scm,fcm),None) for subnet,subroi,atlas,scm,fcm in itertools.product(subnets, subrois, atlases, sc_metrics, fc_metrics) )
    p_min = [1., [None,None,None,None]] # [p, key]
    excl_subnets = []
    for subroi,subnet,atlas,scm,fcm in itertools.product(subrois, subnets, atlases, sc_metrics, fc_metrics):
        sbr = '_'.join(subroi)
        if ( (subnet_corrs[subnet,sbr][atlas,scm,fcm,'controls'] != None) and (subnet_corrs[subnet,sbr][atlas,scm,fcm,'patients'] != None) ):
            t,p = scipy.stats.ttest_ind(subnet_corrs[subnet,sbr][atlas,scm,fcm,'controls']['r'], subnet_corrs[subnet,sbr][atlas,scm,fcm,'patients']['r'], permutations=1000)
            outp_scfc[subnet,sbr,atlas,scm,fcm] = {'t':t, 'p':p}
            if (p < p_min[0]):
                p_min = [p, [subnet,atlas,scm,fcm]]
        else:
            excl_subnets.append(subnet)

    # plot stats
    for subnet,subroi in itertools.product(subnets, subrois):
        sbr = '_'.join(subroi)
        if subnet not in excl_subnets:
            plot_scfc_pvals(outp_scfc, subnet, sbr, atlases, sc_metrics, fc_metrics)#p_min[1][2], p_min[1][3])
            for atlas in atlases:
                plot_scfc_corr(subnet_corrs[subnet,sbr], subnet, atlas, sc_metrics, fc_metrics)
