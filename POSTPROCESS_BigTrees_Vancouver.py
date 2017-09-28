"""
Project Name: BigTreesVan
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: POSTPROCESS_BigTrees_Vancouver.py
Objective: Merge the tiles with the segmented crowns into a single shapefile, clean the dataset and plot interesting data
"""

## TO DO -------------------------------------------------------------------

## STILL TO DO:

# Prior to actual run:
# - reset all parameters to full run values

## SOLVED:
# - DeleteIdentical_management to be run on Shape and not on attributes (even with Tree_ht and Crown_dm we have identical entries bc a raster is the source)
# - check missing tile at Burrard -- original las file was not properly classified (missing high vegetation)
# - merge still having duplicate polygons -- should be good with XY_tolerance 1.2 m
# - change to merge classified tiles --


## IMPORT MODULES ---------------------------------------------------------------

import os
import glob
import time
import arcpy
import numpy as np
from simpledbf import Dbf5
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from Functions_BigTrees_Vancouver import*

if __name__ == '__main__':


## PARAMETERS ----------------------------------------------------------------

    PARAMS = {}

    PARAMS['base_dir'] = r'D:\Research\ANALYSES\BigTreesVan'

    PARAMS['dataset_name'] = 'Vancouver_500m_tiles'
    PARAMS['experiment_name'] = 'step_0p3m_mindist_8_mask_5m'

    PARAMS['wkg_dir'] = os.path.join(PARAMS['base_dir'], 'wkg')
    PARAMS['exp_dir'] = os.path.join(PARAMS['wkg_dir'], PARAMS['dataset_name'], PARAMS['experiment_name'])

    PARAMS['output_mxd'] = os.path.join(PARAMS['exp_dir'], r'VanBigTrees_'+PARAMS['dataset_name']+'_'+PARAMS['experiment_name']+'.mxd')   ## path to final mxd to save results

    PARAMS['XY_tolerance'] = "1.5 Meters"   ## increase spatial tolerance to deal with small shift in the tiles when mergind and deleting duplicate polygons

    PARAMS['max_tree_ht'] = 66   ## segments higher than this threshold are not trees (cranes, towers, trees on top of buildings, etc.)
    PARAMS['min_crown_dm'] = 2   ## segments with crown diameter lower than this threshold are not trees (cranes, towers, trees on top of buildings, etc.)
    PARAMS['X_coord_artifact'] = 498000   ## x coordinate (beginning of easternmost tile) past which there are artifacts due to missing ground point in las files in E:\BigTreesVan_data\LiDAR\CoV\Classified_LiDAR\las

    PARAMS['merged_seg_filename'] = 'TreeCrowns_Vancouver.shp'

    PARAMS['tile_segmented_file_key'] = os.path.join(PARAMS['exp_dir'], 'tiles_classified_FINAL', '*_segments_classif.shp')

    PARAMS['source_lyr_path'] = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'conif_decid.lyr')

    PARAMS['final_dir'] = os.path.join(PARAMS['exp_dir'], 'FINAL_Outputs')

    PARAMS['fig_dir'] = os.path.join(PARAMS['exp_dir'], 'Figures')

## START ---------------------------------------------------------------------

    print(python_info())

    print('POSTPROCESS_BigTrees_Vancouver.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    start_time = tic()

    arcpy.CheckOutExtension("3D")
    arcpy.CheckOutExtension("spatial")
    arcpy.env.overwriteOutput = True
    arcpy.env.XYTolerance = PARAMS['XY_tolerance']

    arcgis_temp_dir = os.path.join(PARAMS['exp_dir'], 'arcgis_temp')
    dirs = [arcgis_temp_dir, PARAMS['final_dir'], PARAMS['fig_dir']]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    merged_seg_path = os.path.join(PARAMS['final_dir'], PARAMS['merged_seg_filename'])

    ## List file paths of segmented tiles
    segments_paths = glob.glob(PARAMS['tile_segmented_file_key'])

    ## Merge all listed tiles and remove duplicates in overlapping regions
    arcpy.Merge_management(segments_paths, merged_seg_path)
    arcpy.MakeFeatureLayer_management(merged_seg_path, "merged_seg_lyr")  ## make a layer from the feature class
    arcpy.DeleteIdentical_management(in_dataset="merged_seg_lyr", fields=["Shape"], xy_tolerance=PARAMS['XY_tolerance'])

    ## Clean shapefile by removing aberrant polygons
    arcpy.SelectLayerByAttribute_management(in_layer_or_view="merged_seg_lyr",
                                            where_clause=' "X_coord" >= %d OR "Tree_ht" >= %d OR "Crown_dm" < %d '
                                                                                            % (PARAMS['X_coord_artifact']-15, PARAMS['max_tree_ht'], PARAMS['min_crown_dm'], ))   ## also remove maximum expected crown radius
    arcpy.DeleteFeatures_management("merged_seg_lyr")

    ## Get MXD and dataframe
    mxd = arcpy.mapping.MapDocument(PARAMS['output_mxd'])
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    mxd.activeView = df.name
    mxd.title = df.name

    ## Define and add layers
    source_lyr = arcpy.mapping.Layer(PARAMS['source_lyr_path'])
    layer_to_add = arcpy.mapping.Layer(merged_seg_path)
    arcpy.mapping.AddLayer(df, layer_to_add, "TOP")

    ## Applying symbology
    layer_to_update = arcpy.mapping.ListLayers(mxd, layer_to_add.name, df)[0]  ## redefine layer_to_update as an arcpy object
    arcpy.ApplySymbologyFromLayer_management(layer_to_update, source_lyr)  ## first apply the symbology with ApplySymbologyFromLayer_management (needs a path to the layer providing the symbology)
    arcpy.mapping.UpdateLayer(df, layer_to_update, source_lyr, True)  ## update symbology by using the one of source_lyr (has to be an arpy object and not just the path)

    ## Saving final MXD in same file
    mxd.save()

    ## Plot interesting data
    full_df = Dbf5(merged_seg_path.replace('shp', 'dbf')).to_dataframe()

    for type in ['coniferous', 'deciduous']:

        if type == 'coniferous':
            label = 'Y'
        else:
            label = 'N'

        tree_ht = full_df.loc[full_df['Pred_Conif']==label][['Tree_ht']].values
        crown_dm = full_df.loc[full_df['Pred_Conif']==label][['Crown_dm']].values  # the histogram of the data

        ## Histogram of tree heights
        fig = plt.figure()
        plt.hist(tree_ht, 50, facecolor='green', alpha=0.75)
        plt.grid(True)
        plt.xlabel('Tree height [m]')
        plt.ylabel('count')
        plt.savefig(os.path.join(PARAMS['fig_dir'], 'Hist_%s_tree_ht.pdf' % type))

        ## Histogram of crown diameters
        fig = plt.figure()
        plt.hist(crown_dm, 50, facecolor='red', alpha=0.75)
        plt.grid(True)
        plt.xlabel('Crown diameter [m]')
        plt.ylabel('count')
        plt.savefig(os.path.join(PARAMS['fig_dir'], 'Hist_%s_crown_dm.pdf' % type))

        ## Calculate the point density
        tree_ht_subs = tree_ht[::10]
        crown_dm_subs = crown_dm[::10]
        xy = np.hstack([tree_ht_subs, crown_dm_subs]).transpose()
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = crown_dm_subs[idx], tree_ht_subs[idx], z[idx]

        ## Scatterplot of tree height vs. crown diameter
        fig = plt.figure()
        plt.scatter(x, y, c=z, s=50, edgecolor='', alpha=0.75)
        plt.grid(True)
        plt.xlabel('Crown diameter [m]')
        plt.ylabel('Tree height [m]')
        plt.savefig(os.path.join(PARAMS['fig_dir'], 'Scatter_%s_ht_dm.pdf' % type))

    print('Total ' + toc(start_time))