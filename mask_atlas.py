#
# Mask Atlas
#
# Filter existing parcellation based on ROI names and location (optional)
#
# Author: Sebastien Naze
# -----------------------------------------------------------------------

import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
from nilearn import image
import numpy as np
import os
import pandas as pd
import scipy

# default global variable
atlas_name = 'schaefer400_harrison2009'
atlas_suffix = '.nii.gz' #'MNI_lps_mni.nii.gz'
new_atlas_name = 'FrStrPalThal_' + atlas_name
proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline/'
#atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'
atlas_dir = os.path.join(proj_dir, 'utils')
out_dir = os.path.join(proj_dir, 'utils')
atlas_cfg = pd.read_json(os.path.join(atlas_dir, 'atlas_config.json'))
atlas_img = image.load_img(os.path.join(atlas_dir, atlas_name+atlas_suffix))

atlas_data = atlas_img.get_fdata()

def get_roi_node_indices(roi):
    """ Get node indices of ROI based on its name """
    roi_node_ids = np.array([i for i,label in enumerate(atlas_cfg[atlas_name]['node_names']) if roi in label]).flatten()
    roi_node_ids = np.array(atlas_cfg[atlas_name]['node_ids'])[roi_node_ids.astype(int)]
    return roi_node_ids

def get_rois_node_indices(rois):
    """ Get node indices of ROIs based on their names"""
    node_ids = np.concatenate([get_roi_node_indices(roi=reg) for reg in rois])
    return node_ids

def get_roi_atlas_indices(data, roi='Thal'):
    """ Get atlas data indices (i,j,k) of ROI """
    roi_node_ids = np.array([i for i,l in enumerate(atlas_cfg[atlas_name]['node_names']) if roi in l])
    roi_node_ids = np.array(atlas_cfg[atlas_name]['node_ids'])[roi_node_ids]
    roi_atlas_ids = np.where(np.any([atlas_data==roi_idx for roi_idx in roi_node_ids], axis=0))
    return np.array(roi_atlas_ids).T

def get_roi_names(roi_node_ids):
    """ Get ROI names from their node indices """
    cfg_ids = [np.where(atlas_cfg[atlas_name]['node_ids']==idx)[0][0] for idx in roi_node_ids]
    names = [atlas_cfg[atlas_name]['node_names'][i] for i in cfg_ids]
    return names

# filter Frontal and sub-cortical ROIs
ctx = ['PFC', 'OFC', 'Fr', 'FEF', 'ACC', 'Cing', 'PrC', 'ParMed']
subctx = ['Acc', 'Put', 'Caud']
#ctx = ['Left_SalVentAttnA_FrMed_1', 'Right_SalVentAttnB_PFCl_1', 'Right_ContB_PFClv_1']
#subctx = ['KJjhfewbfkf'] # whatever hash code to discard
ctx_node_ids = get_rois_node_indices(ctx) #np.concatenate([get_roi_node_indices(roi=reg) for reg in ctx])
subctx_node_ids = get_rois_node_indices(subctx) #np.concatenate([get_roi_node_indices(roi=reg) for reg in subctx])

# discard regions which are not in ROI list
fspt_atlas_data = atlas_data.copy()
fspt_atlas_data[~np.any([atlas_data==roi for roi in np.concatenate([ctx_node_ids, subctx_node_ids])], axis=0)] = 0

def save_new_atlas(new_atlas_data):
    """ Save a new atlas """
    new_atlas_img = nib.Nifti1Image(new_atlas_data, atlas_img.affine, atlas_img.header)
    fname = new_atlas_name + atlas_suffix
    nib.save(new_atlas_img, os.path.join(out_dir, fname))
    print('New atlas {} saved in {}'.format(fname, out_dir))
    return fname


# save new atlas
new_atlas_fname = save_new_atlas(fspt_atlas_data)

# transform indices in MNI coordinates
M = atlas_img.affine[:3,:3]
abc = atlas_img.affine[:3,3]
def f(i, j, k):
    """ Return X, Y, Z coordinates for i, j, k """
    return M.dot([i, j, k]) + abc

def get_coords(data, ids):
    """ Get MNI coordinates of atlas data from atlas indices """
    coords = []
    for i in ids:
        x, y, z = f(*i)
        coords.append([x, y, z])
    return np.array(coords)

# get thalamic coordinates
#thal_atlas_ids = get_roi_atlas_indices(atlas_data, roi='Thal')
#coords = get_coords(atlas_data, thal_atlas_ids)

most_posterior_coord = -30 #coords[:,1].min()

def filter_posterior_regions(orig_atlas_data, new_atlas_data, most_posterior_coord, in_node_ids, except_node_ids=[]):
    print('Filter any region which is posterior to {}mm'.format(most_posterior_coord))

    # get indices of labeled voxels
    new_atlas_ids = np.where(new_atlas_data!=0)
    new_atlas_ids = np.array(new_atlas_ids).T
    #print(new_atlas_ids.shape)

    # get indices of voxels which are too posterior
    new_atlas_coords = get_coords(orig_atlas_data, new_atlas_ids)
    too_post_ids = new_atlas_ids[ [i for i,y in enumerate(new_atlas_coords[:,1]) if (y < most_posterior_coord)] ]
    too_post_ids = tuple(map(tuple,too_post_ids))

    # get ROI's indices and names that are too posterior
    roi_ids = []
    for t in too_post_ids:
        roi_ids.append(orig_atlas_data[t])
    roi_ids = np.unique(roi_ids).astype(int)

    # remove these ROIs from new atlas
    newer_atlas_data = new_atlas_data.copy()
    newer_atlas_data[np.any([orig_atlas_data==roi for roi in roi_ids if roi not in except_node_ids], axis=0)] = 0

    # remove these ROIs node ids
    def remove_roi_node_ids_from_other_node_ids_list(other_node_ids, roi_ids):
        """ remove these ROIs node ids """
        discard = []
        for roi in roi_ids:
            for i,reg in enumerate(other_node_ids):
                if roi==reg:
                    discard.append(i)
        return np.delete(other_node_ids,discard)
    in_node_ids = remove_roi_node_ids_from_other_node_ids_list(in_node_ids, roi_ids)
    newer_node_ids = sorted(np.concatenate([in_node_ids, except_node_ids]))
    newer_node_names = get_roi_names(newer_node_ids)
    #print(*list(zip(node_ids,node_names)), sep='\n')

    return newer_atlas_data, newer_node_ids, newer_node_names

new_atlas_data, new_node_ids, new_node_names = filter_posterior_regions(atlas_data, fspt_atlas_data, most_posterior_coord, in_node_ids=ctx_node_ids, except_node_ids=subctx_node_ids)

# save new atlas
new_atlas_fname = save_new_atlas(new_atlas_data)


# and save new atlas config
new_atlas_cfg = { atlas_name : {'file':new_atlas_fname, 'node_names':new_node_names, 'node_ids':list(np.array(new_node_ids).astype(str))} }
json_fname = 'atlas_cfg_'+new_atlas_name+'.json'
with open(os.path.join(out_dir,json_fname), 'w') as jf:
    json.dump(new_atlas_cfg, jf, indent=2)
