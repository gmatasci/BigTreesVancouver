"""
Project Name: BigTreesVan
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: POSTPROCESS_BigTrees_Vancouver.py
Objective:
"""

## TO DO -------------------------------------------------------------------

## STILL TO DO:
# - change to merge classified tiles

# Prior to actual run:
# - reset all parameters to full run values

## SOLVED:
# - DeleteIdentical_management to be run on Shape and not on attributes (even with Tree_ht and Crown_dm we have identical entries bc a raster is the source)
# - check missing tile at Burrard -- original las file was not properly classified (missing high vegetation)
# - merge still having duplicate polygons -- should be good with XY_tolerance 1.2 m


## IMPORT MODULES ---------------------------------------------------------------

import os
import glob
import time
import arcpy

from Functions_BigTrees_Vancouver import*

if __name__ == '__main__':


## PARAMETERS ----------------------------------------------------------------

    PARAMS = {}

    PARAMS['base_dir'] = r'D:\Research\ANALYSES\BigTreesVan'

    PARAMS['dataset_name'] = 'Vancouver_500m_tiles'
    PARAMS['experiment_name'] = 'step_0p3m_mindist_8'

    # PARAMS['dataset_name'] = 'Tune_alg_8tiles'
    # PARAMS['experiment_name'] = 'step_0p3m_radius_8_nofill'

    PARAMS['wkg_dir'] = os.path.join(PARAMS['base_dir'], 'wkg')
    PARAMS['exp_dir'] = os.path.join(PARAMS['wkg_dir'], PARAMS['dataset_name'], PARAMS['experiment_name'])

    PARAMS['output_mxd'] = os.path.join(PARAMS['exp_dir'], r'VanBigTrees_'+PARAMS['dataset_name']+'_'+PARAMS['experiment_name']+'.mxd')   ## path to final mxd to save results

    PARAMS['XY_tolerance'] = "1.2 Meters"   ## increase spatial tolerance to deal with small shift in the tiles when mergind and deleting duplicate polygons

    PARAMS['X_coord_artifact'] = 498000   ## x coordinate (beginning of easternmost tile) past which there are artifacts due to missing ground point in las files in E:\BigTreesVan_data\LiDAR\CoV\Classified_LiDAR\las
    PARAMS['merged_seg_filename'] = 'TreeCrowns_merged.shp'

    # PARAMS['source_lyr_path'] = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'conif_decid.lyr')
    PARAMS['source_lyr_path'] = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'segments_polyg.lyr')


## START ---------------------------------------------------------------------

    print(python_info())

    print('POSTPROCESS_BigTrees_Vancouver.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    start_time = tic()

    arcpy.CheckOutExtension("3D")
    arcpy.CheckOutExtension("spatial")
    arcpy.env.overwriteOutput = True
    arcpy.env.XYTolerance = PARAMS['XY_tolerance']

    arcgis_temp_dir = os.path.join(PARAMS['exp_dir'], 'arcgis_temp')
    tile_segmented_dir = os.path.join(PARAMS['exp_dir'], 'tiles_segmented')
    final_dir = os.path.join(PARAMS['exp_dir'], 'Final_Outputs')
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    merged_seg_path = os.path.join(final_dir, PARAMS['merged_seg_filename'])

    ## List file paths of segmented tiles
    file_key = os.path.join(tile_segmented_dir, '*_segments_polyg.shp')
    segments_paths = glob.glob(file_key)


    # ## Merge all listed tiles and remove duplicates in overlapping regions
    arcpy.Merge_management(segments_paths, merged_seg_path)
    arcpy.MakeFeatureLayer_management(merged_seg_path, "merged_seg_lyr")  ## make a layer from the feature class
    arcpy.DeleteIdentical_management(in_dataset="merged_seg_lyr", fields=["Shape"], xy_tolerance=PARAMS['XY_tolerance'])
    # arcpy.DeleteIdentical_management(in_dataset=merged_seg_path, fields=["Shape_area", "Tree_ht", "X_coord"])  ## Select and delete artifact polygons from final layer

    arcpy.SelectLayerByAttribute_management(in_layer_or_view="merged_seg_lyr", where_clause=' "X_coord" >= %d ' % (PARAMS['X_coord_artifact']-15))   ## also remove maximum expected crown radius
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

    print('Total ' + toc(start_time))