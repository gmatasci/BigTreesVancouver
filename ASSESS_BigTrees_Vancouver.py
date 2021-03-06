"""
Project Name: BigTreesVan
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: ASSESS_BigTrees_Vancouver.py
Objective: Assessment of tree detection algorithm: detection rate, stem location and attributes.
"""

## IMPORT MODULES ---------------------------------------------------------------

from __future__ import division  ## to allow floating point divisions
import os
import sys
import warnings
import glob
import time
import arcpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
from simpledbf import Dbf5

from Functions_BigTrees_Vancouver import*

if __name__ == '__main__':

    mpl.rc('image', interpolation='none')   ## to change ugly default behavior of image/imshow plots

## PARAMETERS ----------------------------------------------------------------

    PARAMS = {}

    PARAMS['base_dir'] = r'D:\Research\ANALYSES\BigTreesVan'

    PARAMS['dataset_name'] = 'Vancouver_500m_tiles'
    PARAMS['experiment_name'] = 'step_0p3m_mindist_8_mask_5m'

    PARAMS['ref_trees_path'] = r'E:\BigTreesVan_data\GroundTruth\reference_trees.shp'

    PARAMS['wkg_dir'] = os.path.join(PARAMS['base_dir'], 'wkg')
    PARAMS['exp_dir'] = os.path.join(PARAMS['wkg_dir'], PARAMS['dataset_name'], PARAMS['experiment_name'])

    PARAMS['ht_filter_colname'] = 'appr_ht'   ## name of column containing the approximate height for reference trees (extracted by overlaying CHM and tree location)
    PARAMS['tree_ht_thresh'] = 30   ## height threshold in meters below which we remove reference trees

    PARAMS['pred_ht_colname'] = 'Tree_ht'  ## name of column containing the predicted height for all trees
    PARAMS['obs_ht_colname'] = 'avg_ht'    ## name of column containing the observed height for reference trees
    PARAMS['pred_diam_colname'] = 'Crown_dm'  ## name of column containing the predicted crown diameter for all trees
    PARAMS['obs_diam_colname'] = 'cr_diam'   ## name of column containing the observed crown diameter for reference trees



## START ---------------------------------------------------------------------


    print(python_info())

    print('ASSESS_BigTrees_Vancouver.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    start_time = tic()

    ## Save parameters
    params_filename = 'PARAMS_ASSESS_%s_%s_predHtThresh%d.json' % (PARAMS['dataset_name'], PARAMS['experiment_name'], PARAMS['tree_ht_thresh'])
    with open(os.path.join(PARAMS['exp_dir'], params_filename), 'w') as fp:
        json.dump(PARAMS, fp)

    arcpy.CheckOutExtension("3D")
    arcpy.CheckOutExtension("spatial")
    arcpy.env.overwriteOutput = True

    arcgis_temp_dir = os.path.join(PARAMS['exp_dir'], 'arcgis_temp')
    tile_segmented_dir = os.path.join(PARAMS['exp_dir'], 'tiles_segmented')

    ## List file paths of all tiles
    file_key = os.path.join(tile_segmented_dir, '*_segments_polyg.shp')
    segments_paths = glob.glob(file_key)

    FN_tree_IDs = TP_tree_IDs = pd.Series(dtype=np.int)
    error_df = pd.DataFrame(columns={'seg_ID', 'TREE_ID', 'error_XY', 'error_Z', 'error_crown_dm'})

    ## Loop over the segments tiles
    for i_tile, _ in enumerate(segments_paths):

        ## Get tile name
        filename = segments_paths[i_tile].split('\\')[-1]
        tile_name = filename.split('_')[0]+'_'+filename.split('_')[1]
        print('tile %s (%d/%d)' % (tile_name, i_tile+1, len(segments_paths)))

        result = arcpy.GetCount_management(segments_paths[i_tile])
        count = int(result.getOutput(0))
        if count == 0:
            continue

        ## Check which points fall into tile boundary
        temp_bounding_box_path = os.path.join(arcgis_temp_dir, "bounding_box.shp")
        arcpy.MinimumBoundingGeometry_management(segments_paths[i_tile], temp_bounding_box_path, "ENVELOPE", "ALL")

        arcpy.SpatialJoin_analysis(temp_bounding_box_path, PARAMS['ref_trees_path'],
                                   os.path.join(arcgis_temp_dir, "ref_trees_tile_boundary_join.shp"),
                                   "JOIN_ONE_TO_MANY", "KEEP_COMMON", "", "CONTAINS")

        ## If tile does not overlap any reference tree skip to next tile
        if arcpy.management.GetCount(os.path.join(arcgis_temp_dir, "ref_trees_tile_boundary_join.shp"))[0] == "0":
            continue

        ## Spatial join to check which reference trees are contained in each segment
        arcpy.SpatialJoin_analysis(segments_paths[i_tile], PARAMS['ref_trees_path'], os.path.join(arcgis_temp_dir, "ref_trees_segments_join.shp"),
                                   "JOIN_ONE_TO_MANY", "KEEP_COMMON", "", "CONTAINS")

        tile_join_df = Dbf5(os.path.join(arcgis_temp_dir, 'ref_trees_tile_boundary_join.dbf')).to_dataframe()
        drop_bool = (tile_join_df[PARAMS['obs_ht_colname']] == 0) & (tile_join_df[PARAMS['ht_filter_colname']] < PARAMS['tree_ht_thresh'])  ## to be dropped a tree has to have no measured height value (avg_ht) while being under a certain threshold in filtering height retieved based on the raw CHM
        tile_join_df = tile_join_df.drop(tile_join_df[drop_bool].index)

        ## Read attribute table and remove reference trees whose height is below a threshold
        seg_join_df = Dbf5(os.path.join(arcgis_temp_dir, 'ref_trees_segments_join.dbf')).to_dataframe()
        drop_bool = (seg_join_df[PARAMS['obs_ht_colname']] == 0) & (seg_join_df[PARAMS['ht_filter_colname']] < PARAMS['tree_ht_thresh'])  ## to be dropped a tree has to have no measured height value (avg_ht) while being under a certain threshold in filtering height retieved based on the raw CHM
        seg_join_df = seg_join_df.drop(seg_join_df[drop_bool].index)

        ## Extend FN Series with Tree_IDs of trees not found within the segments
        FN_tree_IDs = FN_tree_IDs.append(tile_join_df.loc[~tile_join_df['TREE_ID'].isin(seg_join_df['TREE_ID'])]['TREE_ID'])

        ## Remove remaining TREE_ID duplicates in the tree-segment pairs
        if seg_join_df.shape[0] != len(np.unique(seg_join_df['TREE_ID'])):
            seg_join_df = seg_join_df.drop_duplicates(subset='TREE_ID')

        ## Initialize error_df_tile df to contain error values for this tile
        unique_seg_IDs = np.unique(seg_join_df["ID"])
        error_df_tile = pd.DataFrame({'seg_ID': unique_seg_IDs})
        error_df_tile['TREE_ID'] = error_df_tile['error_XY'] = error_df_tile['error_Z'] = error_df_tile['error_crown_dm'] = ''

        ## Loop over all unique segment IDs
        for id in unique_seg_IDs:
            candidates = seg_join_df.loc[seg_join_df['ID'] == id]
            distance_XY = np.linalg.norm(candidates[['X_coord', 'Y_coord']].values - candidates[['POINT_X', 'POINT_Y']].values, axis=1)
            matched_tree = candidates.iloc[np.argmin(distance_XY)]
            TP_tree_IDs = TP_tree_IDs.append(pd.Series(matched_tree['TREE_ID']))  ## extend TP tree ID Series with the ID of the matched tree

            ## Fill matched tree ID and distance between it and the treetop
            error_df_tile.loc[error_df_tile['seg_ID']==id, ['TREE_ID']] = matched_tree['TREE_ID']
            error_df_tile.loc[error_df_tile['seg_ID']==id, ['error_XY']] = np.amin(distance_XY)

            ## Compute tree height error
            obs_Z = matched_tree[[PARAMS['obs_ht_colname']]].values
            pred_Z = matched_tree[[PARAMS['pred_ht_colname']]].values
            if obs_Z == 0:
                error_df_tile.loc[error_df_tile['seg_ID']==id, ['error_Z']] = np.NaN
            else:
                error_df_tile.loc[error_df_tile['seg_ID']==id, ['error_Z']] = pred_Z - obs_Z  ## error as predicted - observed

            ## Compute crown diameter error
            obs_diam = matched_tree[[PARAMS['obs_diam_colname']]].values
            pred_diam = matched_tree[[PARAMS['pred_diam_colname']]].values
            if obs_diam == 0:
                error_df_tile.loc[error_df_tile['seg_ID'] == id, ['error_crown_dm']] = np.NaN
            else:
                error_df_tile.loc[
                    error_df_tile['seg_ID'] == id, ['error_crown_dm']] = pred_diam - obs_diam  ## error as predicted - observed

            FN_tree_IDs = FN_tree_IDs.append(candidates[candidates['TREE_ID']!=matched_tree['TREE_ID']]['TREE_ID'])   ## increment FN count by the number of extra trees (after the 1st one) matched to a given segment

        error_df = error_df.append(error_df_tile)

    error_df = error_df.drop_duplicates(subset='TREE_ID')   ## to remove duplicates potentially created by reference trees that appear in multiple overlapping tiles
    TP_tree_IDs = TP_tree_IDs.drop_duplicates()
    FN_tree_IDs = FN_tree_IDs.drop_duplicates()

    ## Compute accuracy metrics
    RES = {}
    RES['TPR'] = len(TP_tree_IDs) / (len(TP_tree_IDs) + len(FN_tree_IDs))
    RES['FNR'] = len(FN_tree_IDs) / (len(TP_tree_IDs) + len(FN_tree_IDs))
    RES['avg_XYdist'] = error_df['error_XY'].mean(skipna=True)
    RES['Z_RMSE'] = np.sqrt((error_df['error_Z']**2).mean(skipna=True))
    RES['Crown_dm_RMSE'] = np.sqrt((error_df['error_crown_dm']**2).mean())
    RES['Z_bias'] = error_df['error_Z'].mean(skipna=True)
    RES['Crown_dm_bias'] = error_df['error_crown_dm'].mean()

    RES['n_rates'] = len(TP_tree_IDs) + len(FN_tree_IDs)
    RES['n_XY'] = sum(~error_df['error_XY'].isnull())
    RES['n_Z'] = sum(~error_df['error_Z'].isnull())
    RES['n_Crown_dm'] = sum(~error_df['error_crown_dm'].isnull())

    ## Save RES dictionary to a json file
    res_filename = 'RES_ASSESS_%s_%s_predHtThresh%d.json' % (PARAMS['dataset_name'], PARAMS['experiment_name'], PARAMS['tree_ht_thresh'])
    with open(os.path.join(PARAMS['exp_dir'], res_filename), 'w') as fp:
        json.dump(RES, fp)

    print('Main segmentation results:\n\n '
          'Detection rate (n=%d): TPR = %.3f, FNR = %.3f \n\n '
          'Average X, Y distance (n=%d) [m] = %2.2f \n\n '
          'Treetop height (n=%d): RMSE [m] = %2.2f, bias [m] = %2.2f \n\n '
          'Crown diameter (n=%d): RMSE [m] = %2.2f, bias [m] = %2.2f \n\n '
          % (RES['n_rates'], RES['TPR'], RES['FNR'],
             RES['n_XY'], RES['avg_XYdist'],
             RES['n_Z'], RES['Z_RMSE'], RES['Z_bias'],
             RES['n_Crown_dm'], RES['Crown_dm_RMSE'], RES['Crown_dm_bias']))

    print('Total ' + toc(start_time))




