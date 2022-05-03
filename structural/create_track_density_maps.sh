#!/bin/bash

module load mrtrix3/3.0_RC3

proj_dir=/mnt/lustre/working/lab_lucac/sebastiN/projects/OCDbaseline/
#proj_dir=/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline/

mapfile -t subj_array < ${proj_dir}docs/code/subject_list_all.txt

for subj in ${subj_array[@]}; do

  out_dir=${proj_dir}postprocessing/${subj}/

  mkdir -p ${out_dir}

  declare -a atlases=("AccR_dPutR_vPutL_lvPFC_lPFC_dPFC_sphere6-12mm")

  for atlas in ${atlases[@]}
  do
      echo ${subj} ${atlas}
      template_file=${out_dir}${subj}_detrend_gsr_filtered_scrubFD05_brainFWHM8mm_Harrison2009_AccL_ns_sphere_seed_to_voxel_corr.nii.gz

      #tckedit ${out_dir}${atlas}_sift_* ${out_dir}${atlas}_sift.tck
      tckmap -template ${template_file} ${out_dir}${atlas}_sift_1-4.tck ${out_dir}${atlas}_sift_AccOFC.nii.gz
      tckmap -template ${template_file} ${out_dir}${atlas}_sift_2-5.tck ${out_dir}${atlas}_sift_PutPFC.nii.gz
      tckmap -template ${template_file} ${out_dir}${atlas}_sift_3-6.tck ${out_dir}${atlas}_sift_vPutdPFC.nii.gz

      tckmap -template ${template_file} ${out_dir}${atlas}_sift_1-4.tck ${out_dir}${atlas}_nosift_AccOFC.nii.gz
      tckmap -template ${template_file} ${out_dir}${atlas}_sift_2-5.tck ${out_dir}${atlas}_nosift_PutPFC.nii.gz
      tckmap -template ${template_file} ${out_dir}${atlas}_sift_3-6.tck ${out_dir}${atlas}_nosift_vPutdPFC.nii.gz

  done
done
