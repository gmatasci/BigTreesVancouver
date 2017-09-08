"""
Project Name: BigTreesVan
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: MAIN_BigTrees_Vancouver.py
Objective: Build high-res CHM from LiDAR for the city of Vancouver, detect treetops, segment crowns and extract attributes
"""

## TO DO -------------------------------------------------------------------

## STILL TO DO:
# - treetop finder varying with height (from R?)
# - watershed segmentation with markers: python vs OTB
# - grid search on key parameters

# Prior to actual run:
# - reset all parameters to full run values

## SOLVED:
# - treetops cannot be closer than a certain distance -- corner_peaks() handles min_distance_peaks properly (peak_local_max() does not)
# - save PARAMS object in wkg folder -- done as JSON file
# - put in a folder that does not get deleted the shps that are source for lyrs files -- folder in "D:\Research\ANALYSES\BigTreesVan\mxds\lyrs\source_layers"
# - lasgrid and las2dem create raster tiles with a difference of 1 row/column -- they still do but using a common layers extent we can clip rasters to a common extent
# - add final layers to mxd with symbology
# - remove small single pixel segments with some majority vote filter and update unique labels. To check which ones are with:
        # unique, counts = np.unique(segments_arr, return_counts=True)
        # unique_counts = pd.DataFrame({'lab':unique, 'counts':counts})
        # srt_unique_counts = unique_counts.sort('counts') -- not happening anymore
# - lastile producing weird number of tiles -- when running on whole Vancouver is ok
# - check usage of all cores -- works for lastools functions with -cores option
# - how to stitch back tiles? -- not needed, as results will be provided tiled. Could be done by simple merge as now the tiles no longer contain border segments. One possibility was with http://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/remove-raster-segment-tiling-artifacts.htm
# - remove tile buffers? (in lastools or in arcpy w extent.Xmin+buffer, etc.) -- not needed, plus final tiles with segments cannot be unbuffered



## IMPORT MODULES ---------------------------------------------------------------


from __future__ import division  # to allow floating point divisions
import os
import sys
import glob
import time
import logging
import shutil     # to remove folders and their contents
import gc       # to manually run garbage collection
import arcpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.morphology import *
from skimage.feature import *
from skimage.filters import rank
from scipy.ndimage.morphology import binary_fill_holes
import pandas as pd
import json
import multiprocessing as mp
from functools import partial

from Functions_BigTrees_Vancouver import*
from Functions_LAStools import*

if __name__ == '__main__':

    mpl.rc('image', interpolation='none')   ## to change ugly default behavior of image/imshow plots

## PARAMETERS ----------------------------------------------------------------

    PARAMS = {}

    # PARAMS['lidar_processing'] = True
    PARAMS['lidar_processing'] = False

    # PARAMS['raster_processing'] = True
    PARAMS['raster_processing'] = False

    PARAMS['build_CHMs'] = True
    # PARAMS['build_CHMs'] = False

    PARAMS['segment_crowns'] = True
    # PARAMS['segment_crowns'] = False

    PARAMS['build_MXD'] = True
    # PARAMS['build_MXD'] = False

    PARAMS['write_files'] = True
    # PARAMS['write_files'] = False

    # PARAMS['plot_figures'] = True
    PARAMS['plot_figures'] = False

    PARAMS['base_dir'] = r'D:\Research\ANALYSES\BigTreesVan'

    PARAMS['experiment_name'] = '0p3m_PF_CHM_minDistpeaks_7'
    # PARAMS['experiment_name'] = '0p3m_PF_CHM'

    PARAMS['dataset_name'] = 'tune_alg_10tiles'
    # PARAMS['dataset_name'] = '1000m_1tile_QE'

    #### TO LAZ and remove unzipped las
    # PARAMS['data_dir'] = r'E:\BigTreesVan_data\LiDAR\CoV\Classified_LiDAR'
    #### TO LAZ and remove unzipped las
    PARAMS['data_dir'] = os.path.join(PARAMS['base_dir'], r'wkg\trial_data_'+PARAMS['dataset_name'])

    PARAMS['wkg_dir'] = os.path.join(PARAMS['base_dir'], 'wkg')
    PARAMS['exp_dir'] = os.path.join(PARAMS['wkg_dir'], PARAMS['dataset_name'], PARAMS['experiment_name'])

    PARAMS['input_mxd'] = os.path.join(PARAMS['base_dir'], 'mxds', r'VanBigTrees_empty.mxd')  ## path to empty mxd that has to have a scale bar (will adapt automatically) and bookmarks already set
    PARAMS['output_mxd'] = os.path.join(PARAMS['base_dir'], 'mxds', r'VanBigTrees_'+PARAMS['dataset_name']+'_'+PARAMS['experiment_name']+'.mxd')   ## path to final mxd to save results

    PARAMS['nr_cores'] = 32    ## number of cores to be used by lastools functions

    PARAMS['tile_size'] = 500      ## tile size for lidar processing with lastools (half the tile size of tiles received from City of Vancouver)

    PARAMS['tile_buffer'] = 30      ## tile buffer to avoid boundary problems (we expect no tree crown to be bigger than this)

    PARAMS['step'] = 0.3  ## 0.3, pixel size of raster layers (DSM, CHM, masks, etc.)

    PARAMS['cell_size'] = 2     ## lasnoise parameter to remove powerlines: size in meters of each voxel of the 3x3 voxel neighborhood
    PARAMS['isolated'] = 50     ## lasnoise parameter to remove powerlines: remove points that have less neighboring points than this threshold in the 3x3 voxel neighborhood

    PARAMS['DEM_type'] = 'PF_DSM'  ## las2dem parameter defining the type of DSM created
    PARAMS['freeze_dist'] = 0.8   ## las2dem spike-free CHM parameter
    PARAMS['subcircle'] = 0.2   ## lasgrid parameter in vegetation mask to thicken point by adding a discrete ring of 8 new points that form a circle with radius "subcircle"
    PARAMS['fill'] = 0          ## lasgrid parameter in vegetation mask to fill voids in the grid with a square search radius "fill" number of pixels

    PARAMS['ht_thresh'] = 2   ## height threshold to filter low vegetation and other structures from masked DSM

    PARAMS['disk_radius'] = 5  ## 5, structuring element for morphological operations (opening(), reconstruction()) on vegetation mask and chm: radius of 5 means a 5*2+1 diameter disk

    PARAMS['min_distance_peaks'] = 7   ## minimum separation in pixels between treetops in peak_local_max() and corner_peaks(), i.e. 9 pix = 9*0.3 = 2.7 m @ 0.3 m spatial resolution

    PARAMS['tree_ht_thresh'] = 10   ## height threshold in meters below which we remove segmented trees
    PARAMS['crown_dm_thresh'] = 40  ## crown diameter threshold in meters above which the polygon is no more a tree but background instead

    PARAMS['gr_names_keys'] = {'VegMaskRaw': 'veg_mask_raw',
                               'VegMaskFilled': 'veg_mask_filled',
                               'CHMtreeTops': 'chm_raw',
                               'CHMsegments': 'chm_reconstr',
                               'TreeCrown': 'segments_polyg',
                               'TreeTop': 'tree_tops'}   ## MXD group names and corresponding file key for glob.glob()


## START ---------------------------------------------------------------------


    print(python_info())

    print('MAIN_BigTrees_Vancouver.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    print('Dataset: %s \nExperiment: %s' % (PARAMS['dataset_name'], PARAMS['experiment_name']))

    start_time = tic()

    params_filename = 'PARAMS_MAIN_%s_%s.json' % (PARAMS['dataset_name'], PARAMS['experiment_name'])
    with open(os.path.join(PARAMS['exp_dir'], params_filename), 'w') as fp:
        json.dump(PARAMS, fp)

    arcpy.CheckOutExtension("3D")
    arcpy.CheckOutExtension("spatial")
    arcpy.env.overwriteOutput = True

    arcgis_temp_dir = os.path.join(PARAMS['exp_dir'], 'arcgis_temp')
    tile_las_dir = os.path.join(PARAMS['data_dir'], 'las')
    tile_laz_dir = os.path.join(PARAMS['data_dir'], 'laz')
    tile_denoised_dir = os.path.join(PARAMS['exp_dir'], 'tiles_denoised')
    tile_ht_norm_dir = os.path.join(PARAMS['exp_dir'], 'tiles_ht_norm')
    tile_dem_dir = os.path.join(PARAMS['exp_dir'], 'tiles_dem')
    tile_mask_dir = os.path.join(PARAMS['exp_dir'], 'tiles_mask')
    tile_chm_treetops_dir = os.path.join(PARAMS['exp_dir'], 'tiles_chm_treetops')
    tile_segmented_dir = os.path.join(PARAMS['exp_dir'], 'tiles_segmented')

    dirs = [arcgis_temp_dir,
            tile_las_dir,
            tile_laz_dir,
            tile_denoised_dir,
            tile_ht_norm_dir,
            tile_dem_dir,
            tile_mask_dir,
            tile_chm_treetops_dir,
            tile_segmented_dir
            ]

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    arcpy.env.workspace = arcgis_temp_dir  # set temporary files directory (used when calling Reclassify(), etc.), could use arcpy.env.workspace = "in_memory"


## TO PRE-TILE A SMALL AREA MIMICKING THE BIG DATASET ------------------------------------------

# ## Create an unbuffered tiling from the original data (one original tile for testing):
# lastile(in_dir=tile_las_dir, in_ext='las', out_dir=tile_las_dir, out_ext='las', tile_size=200,
#        tile_buffer=0, nr_cores=PARAMS['nr_cores'], verbose=True)


    if PARAMS['lidar_processing']:

        print("Lidar processing")

## DEFINE TILE-BASED PREPROCESSING ---------------------------------------------------------------

        ## Print infos about las/laz file
        # lasinfo(in_dir=tile_las_dir, file_name='CoV_4840E_54570N', in_ext='las', verbose=True)

        ## To access to the points from neighbouring tiles.
        lasindex(in_dir=tile_las_dir, in_ext='las', nr_cores=PARAMS['nr_cores'], verbose=True)

        ## Create buffered tiles
        lastile(in_dir=tile_las_dir, in_ext='las', out_dir=tile_laz_dir, out_ext='laz', tile_size=PARAMS['tile_size'],
               tile_buffer=PARAMS['tile_buffer'], nr_cores=PARAMS['nr_cores'], verbose=True)


## DTM/DSM AND VEGETATION MASK COMPUTATION ----------------------------------------------------------------

        ## Remove isolated points (powerlines)
        lasnoise(in_dir=tile_laz_dir, in_ext='laz', cell_size=PARAMS['cell_size'], isolated=PARAMS['isolated'],
                 out_dir=tile_denoised_dir, nr_cores=PARAMS['nr_cores'], verbose=True)

        ## Height normalized point cloud on already buffered tiles
        lasheight(in_dir=tile_denoised_dir, in_ext='laz', out_dir=tile_ht_norm_dir, replace_z=True,
                  nr_cores=PARAMS['nr_cores'], verbose=True)

        ## Pit-free CHM (DSM from height normalized point cloud)
        las2dem(in_dir=tile_ht_norm_dir, in_ext='laz', DEM_type='DSM', step=PARAMS['step'],
                freeze_dist=PARAMS['freeze_dist'], out_dir=tile_dem_dir, hillshaded=False,
                nr_cores=PARAMS['nr_cores'], verbose=True)

        ## Lasgrid in classif mode to keep high and mid vegetation (no class 4 in Vancouver data: mid vegetation)
        lasgrid(in_dir=tile_ht_norm_dir, key='*', in_ext='laz', out_dir=tile_mask_dir, out_name='_'+PARAMS['gr_names_keys']['VegMaskRaw'],
                step=PARAMS['step'], action='mask', classes_to_keep='4 5',
                subcircle=PARAMS['subcircle'], fill=PARAMS['fill'], nr_MB_mem=2000,
                nr_cores=PARAMS['nr_cores'], verbose=True)



## RASTER PROCESSING ----------------------------------------------------------------

    if PARAMS['raster_processing']:

        print("Raster processing")

        ## List file paths of all the DSM and vegetation mask tiles
        file_key = os.path.join(tile_dem_dir, '*'+PARAMS['DEM_type']+'*.asc')
        dsm_paths = glob.glob(file_key)
        file_key = os.path.join(tile_mask_dir, '*_'+PARAMS['gr_names_keys']['VegMaskRaw']+'.asc')
        mask_paths = glob.glob(file_key)

        ## Stop if different tiles have been creeated
        if len(dsm_paths) != len(mask_paths):
            sys.exit("Number of DSM tiles differs from nr of mask tiles")

        ## Loop over the tiles
        for i_tile, _ in enumerate(dsm_paths):

            ## Read DSM and vegetation mask and related infos
            dsm = arcpy.Raster(dsm_paths[i_tile])
            mask = arcpy.Raster(mask_paths[i_tile])
            tile_name = dsm.name.split('_')[0]+'_'+dsm.name.split('_')[1]  ## root to rename all tile-based layers

            print('tile %s (%d/%d) ...' % (tile_name, i_tile+1, len(dsm_paths)))

            chm_raw_path = os.path.join(tile_chm_treetops_dir, tile_name + '_' + PARAMS['gr_names_keys']['CHMtreeTops'] + '.asc')

            if PARAMS['build_CHMs']:

                # if os.path.exists(chm_raw_path):
                #     continue

                _, spatial_info = raster_2_array(dsm)   ## read reference spatial_info to be changed later on based on common extent

                ## Build common extent between DSM and mask (las2dem and lasgrid somehow produce rasters w different extents, although pixel grids overlap)
                common_extent = dsm.extent
                common_extent.XMin = np.ceil(max(dsm.extent.XMin, mask.extent.XMin))
                common_extent.YMin = np.ceil(max(dsm.extent.YMin, mask.extent.YMin))
                common_extent.XMax = np.floor(min(dsm.extent.XMax, mask.extent.XMax))
                common_extent.YMax = np.floor(min(dsm.extent.YMax, mask.extent.YMax))

                ## Check if mask is empty and skip tile if true
                if (common_extent.XMin >= common_extent.XMax) | (common_extent.YMin >= common_extent.YMax):
                    continue

                ## Update spatial_info based on common extent
                spatial_info['lower_left'].X = common_extent.XMin
                spatial_info['lower_left'].Y = common_extent.YMin

                ## Clip rasters to common extent to allow processing as numpy arrays
                dsm_clipped_path = os.path.join(arcgis_temp_dir, dsm.name.replace('.asc', '_clipped.dat'))      ## has to be saved in an actual directory (in_memory/ did not work)
                arcpy.Clip_management(dsm, str(common_extent), dsm_clipped_path)    ## str(extent) creates an 'Envelope' object
                dsm_clipped = arcpy.Raster(dsm_clipped_path)
                mask_clipped_path = os.path.join(arcgis_temp_dir, mask.name.replace('.asc', '_clipped.dat'))
                arcpy.Clip_management(mask, str(common_extent), mask_clipped_path)
                mask_clipped = arcpy.Raster(mask_clipped_path)

                ## Convert clipped rasters to array
                dsm_arr, _ = raster_2_array(dsm_clipped)
                arcpy.Delete_management(dsm_clipped_path)
                mask_arr, _ = raster_2_array(mask_clipped)
                arcpy.Delete_management(mask_clipped_path)

                ## Convert non-vegetation (no data) pixels to 0
                mask_arr[mask_arr==-9999] = 0

                ## Denoise vegetation mask with morphological opening
                mask_open_arr = opening(mask_arr, selem=disk(PARAMS['disk_radius']))

                ## Fill holes in vegetation mask
                mask_open_filled_arr = binary_fill_holes(mask_open_arr).astype(int)

                ## Build raw CHM (the one height values must be read from) by masking out non-vegetation DSM pixels and pixels lower than a height threshold
                chm_raw_arr = dsm_arr * mask_open_filled_arr
                chm_raw_arr[chm_raw_arr < PARAMS['ht_thresh']] = 0   ## sets to 0 pixels lower than threshold and -9999 at the borders

                if PARAMS['plot_figures']:
                    plt.figure(), plt.title('Veg. mask'), plt.imshow(mask_arr), plt.colorbar()
                    plt.figure(), plt.title('Veg. mask opened'), plt.imshow(mask_open_arr), plt.colorbar()
                    plt.figure(), plt.title('Veg. mask opened and filled'), plt.imshow(mask_open_filled_arr), plt.colorbar()
                    plt.figure(), plt.title('CHM raw'), plt.imshow(chm_raw_arr), plt.colorbar()

                if PARAMS['write_files']:
                    chm_raw = array_2_raster(chm_raw_arr, spatial_info)
                    arcpy.RasterToASCII_conversion(chm_raw, chm_raw_path)
                    mask_open_filled_towrite_arr = np.copy(mask_open_filled_arr)
                    mask_open_filled_towrite_arr[mask_open_filled_towrite_arr == 0] = -9999
                    mask_open_filled_towrite = array_2_raster(mask_open_filled_towrite_arr, spatial_info)
                    arcpy.RasterToASCII_conversion(mask_open_filled_towrite, os.path.join(tile_mask_dir, tile_name + '_' + PARAMS['gr_names_keys']['VegMaskFilled'] + '.asc'))

            else:

                if os.path.exists(chm_raw_path):
                    chm_raw = arcpy.Raster(chm_raw_path)
                    chm_raw_arr, spatial_info = raster_2_array(chm_raw)

            #### TREETOP DETECTION -------------------------------------------------------------------

            if PARAMS['segment_crowns']:

                ## Build integer CHM for faster image processing
                chm_raw_arr_int = chm_raw_arr * 100  ## multiply by 100 to keep 2 decimal when converting to int for rank.median()
                chm_raw_arr_int = chm_raw_arr_int.astype(np.uint16)
                chm_median_arr = rank.median(chm_raw_arr_int, disk(2))  ## denoises image to avoid high values for trees whose treetop is close to high buildings

                ## Compute opening image
                chm_open_arr = opening(chm_median_arr, selem=disk(PARAMS['disk_radius']))

                ## Compute opening by reconstruction image with opening image as seed and CHM as mask
                chm_reconstr_arr = reconstruction(seed=chm_open_arr, mask=chm_median_arr, method='dilation', selem=disk(PARAMS['disk_radius']))
                if PARAMS['write_files']:
                    chm_reconstr = array_2_raster(chm_reconstr_arr, spatial_info)
                    arcpy.RasterToASCII_conversion(chm_reconstr, os.path.join(tile_chm_treetops_dir, tile_name+'_'+PARAMS['gr_names_keys']['CHMsegments']+'.asc'))

                ## Detect local maxima regions on opening by reconstruction CHM: results in flat tree tops and no noise
                local_max_flat_arr = peak_local_max(chm_reconstr_arr, min_distance=PARAMS['min_distance_peaks'], indices=False)  ## with indices=False, the output is a boolean matrix

                ## Detect local maxima on median filtered CHM: results in true tree tops but with noise
                local_max_raw_arr = corner_peaks(chm_median_arr, min_distance=PARAMS['min_distance_peaks'], indices=False)  ## corner_peaks() respects the min_distance criterion, instead peak_local_max() does not

                ## Keep only true treetops in within flat tops
                local_max_arr = local_max_flat_arr * local_max_raw_arr

                if PARAMS['plot_figures']:
                    plt.figure(), plt.title('CHM median filtered'), plt.imshow(chm_median_arr), plt.colorbar()
                    plt.figure(), plt.title('CHM opened'), plt.imshow(chm_open_arr), plt.colorbar()
                    plt.figure(), plt.title('CHM opened by reconstr.'), plt.imshow(chm_reconstr_arr), plt.colorbar()
                    plt.figure(), plt.title('Local max flat'), plt.imshow(local_max_flat_arr), plt.colorbar()
                    plt.figure(), plt.title('Local max raw'), plt.imshow(dilation(local_max_raw_arr, selem=disk(2))), plt.colorbar()
                    plt.figure(), plt.title('Local max'), plt.imshow(dilation(local_max_arr, selem=disk(2))), plt.colorbar()

                if PARAMS['write_files']:
                    local_max_visual = array_2_raster(dilation(local_max_arr, selem=disk(2)).astype(int), spatial_info)
                    arcpy.RasterToASCII_conversion(local_max_visual, os.path.join(tile_chm_treetops_dir, tile_name+'_tree_tops.asc'))

                ## Watershed segmentation
                chm_reconstr_arr_int = chm_reconstr_arr.astype(np.uint16)  ## recast as integer the input to watershed() has to be an int image
                gradient_arr = rank.gradient(chm_reconstr_arr_int, selem=disk(1))     ## compute gradient image with very thin edges (disk(1))
                markers_arr = local_max_arr.astype(int)
                markers_arr[local_max_arr] = np.arange(1, sum(sum(local_max_arr)) + 1)  # equivalent to the following but without repeated labels for contiguous pixels: ndi.label(local_max_arr.astype(int))[0]  ## assign unique labels to each marker to start the segmentation (same labels will be assigned to segments)
                markers_dilat_arr = dilation(markers_arr, selem=disk(1))  ## thicken markers with a disk of size 2 to avoid problems if local max is at the edge
                segments_arr = watershed(image=gradient_arr, markers=markers_dilat_arr, mask=mask_open_filled_arr)

                if PARAMS['plot_figures']:
                    plt.figure(), plt.title('CHM reconstr'), plt.imshow(chm_reconstr_arr_int), plt.colorbar()
                    plt.figure(), plt.title('Markers'), plt.imshow(dilation(markers_arr), cmap='prism'), plt.colorbar()
                    plt.figure(), plt.title('Gradient'), plt.imshow(gradient_arr), plt.colorbar()
                    plt.figure(), plt.title('Segments'), plt.imshow(segments_arr, cmap='prism'), plt.colorbar()

                ## Convert raster segments to shp
                segments = array_2_raster(segments_arr, spatial_info)
                segment_raster_path = os.path.join(tile_segmented_dir, tile_name + '_segments_img.asc')
                arcpy.RasterToASCII_conversion(segments, segment_raster_path)
                segments_shp_path = os.path.join(tile_segmented_dir, tile_name + '_' + PARAMS['gr_names_keys']['TreeCrown'] + '.shp')
                arcpy.RasterToPolygon_conversion(segments, segments_shp_path, 'NO_SIMPLIFY')

                ## Delete segments with gridcode = 0 (small border polygons)
                uc = arcpy.UpdateCursor(segments_shp_path, "gridcode=0", arcpy.Describe(segments_shp_path).spatialReference)
                for row in uc:
                    uc.deleteRow(row)

                ## Assign projection
                arcpy.DefineProjection_management(segments_shp_path, arcpy.SpatialReference("NAD 1983 UTM Zone 10N"))

                ## Compute area for each tree crown
                arcpy.AddField_management(segments_shp_path, "Shape_area", "DOUBLE")
                exp = "!SHAPE.AREA@SQUAREMETERS!"
                arcpy.CalculateField_management(segments_shp_path, "Shape_area", exp, "PYTHON_9.3")

                ## Compute segment centroids
                arcpy.AddField_management(segments_shp_path, 'X_coord', "DOUBLE", field_precision=11, field_scale=3)  ## if precision is greater than 6, we have to use "DOUBLE"
                arcpy.AddField_management(segments_shp_path, 'Y_coord', "DOUBLE", field_precision=11, field_scale=3)
                arcpy.CalculateField_management(segments_shp_path, 'X_coord', "!SHAPE.CENTROID.X!", "PYTHON_9.3")
                arcpy.CalculateField_management(segments_shp_path, 'Y_coord', "!SHAPE.CENTROID.Y!", "PYTHON_9.3")

                ## Read attribute table to pandas and add crown diameter
                attr_table_df = pd.DataFrame.from_records(arcpy.da.FeatureClassToNumPyArray(segments_shp_path, ('ID', 'GRIDCODE', 'Shape_area')))
                attr_table_df['Crown_dm'] = np.sqrt(attr_table_df['Shape_area']/np.pi) * 2   ## diameter in m of a circle of same are of the current polygon

                ## Retrieve height of treetop within crown
                seg_label_tree_ht_df = pd.DataFrame({'GRIDCODE':markers_arr[local_max_arr], 'Tree_ht':chm_raw_arr[local_max_arr]})  ## Read Tree heigth values from CHM with real values (chm_raw_arr), local_max_arr is the boolean matrix with treetops
                attr_table_df = pd.merge(attr_table_df, seg_label_tree_ht_df, on='GRIDCODE', how='inner')         ## the order of polygons and pixels is already the same for safety reasons we merge the two df based on the segment label

                ## Add empty fields to attribute table
                fields = ['Tree_ht', 'Crown_dm']
                for field in fields:
                    arcpy.AddField_management(segments_shp_path, field, "FLOAT", field_precision=5, field_scale=2)

                ## Update the two new fields of attribute table
                if attr_table_df.shape[0] > 0:
                    with arcpy.da.UpdateCursor(segments_shp_path, fields) as cursor:   ## Create update cursor for feature class
                        for irow, row in enumerate(cursor):
                            for icol, field in enumerate(fields):
                                row[icol] = attr_table_df[field][irow]
                            cursor.updateRow(row)

                ## Create temporary shps: frame.shp will be the result
                temp_bounding_box_path = os.path.join(arcgis_temp_dir, "bounding_box.shp")
                temp_bounding_box_small_path = os.path.join(arcgis_temp_dir, "bounding_box_small.shp")
                temp_frame_path = os.path.join(arcgis_temp_dir, "frame.shp")

                ## Create difference of tile bounding box and inwards buffered tile bounding box
                arcpy.MakeFeatureLayer_management(segments_shp_path, "seg_lyr")  ## make a layer from the feature class
                arcpy.MinimumBoundingGeometry_management("seg_lyr", temp_bounding_box_path, "ENVELOPE", "ALL")
                arcpy.Buffer_analysis(temp_bounding_box_path, temp_bounding_box_small_path, -2)
                arcpy.SymDiff_analysis(temp_bounding_box_path, temp_bounding_box_small_path, temp_frame_path)

                ## Select polygons intersecting this frame to remove boundary segments
                arcpy.SelectLayerByLocation_management(in_layer="seg_lyr", overlap_type="INTERSECT",
                                                       select_features=temp_frame_path)

                ## Use a SQL query to select segments larger than a certain threshold to remove background polygon
                ## and segments associated with trees smaller than a threshold
                arcpy.SelectLayerByAttribute_management(in_layer_or_view="seg_lyr",
                                                        selection_type="ADD_TO_SELECTION",
                                                        where_clause=' "Crown_dm" >= %d OR "Tree_ht" < %d ' % (PARAMS[
                                                            'crown_dm_thresh'], PARAMS['tree_ht_thresh']))

                ## Delete selected polygons from final layer
                arcpy.DeleteFeatures_management("seg_lyr")


#### LOAD LAYERS INTO MXD PROJECT -------------------------------------------------------------------

    if PARAMS['build_MXD']:

        print("Loading layers into MXD")

        ## Get MXD and dataframe
        mxd = arcpy.mapping.MapDocument(PARAMS['input_mxd'])
        df = arcpy.mapping.ListDataFrames(mxd)[0]  ## assuming there is only 1 df (called Layer, the default)
        mxd.activeView = df.name
        mxd.title = df.name

        ## Loop over the tiles in tile_chm_treetops_dir
        layer_paths = glob.glob(os.path.join(tile_mask_dir, '*_'+PARAMS['gr_names_keys']['VegMaskFilled']+'.asc'))   ## we do not plot VegMaskRaw anymore
        layer_paths.extend(glob.glob(os.path.join(tile_chm_treetops_dir, '*_'+PARAMS['gr_names_keys']['CHMtreeTops']+'.asc')))
        layer_paths.extend(glob.glob(os.path.join(tile_chm_treetops_dir, '*_'+PARAMS['gr_names_keys']['CHMsegments']+'.asc')))
        layer_paths.extend(glob.glob(os.path.join(tile_chm_treetops_dir,'*_'+PARAMS['gr_names_keys']['TreeTop']+'.asc')))
        layer_paths.extend(glob.glob(os.path.join(tile_segmented_dir, '*_'+PARAMS['gr_names_keys']['TreeCrown']+'.shp')))
        for i_layer, layer_path in enumerate(layer_paths):

            layer_to_add = arcpy.mapping.Layer(layer_path)
            arcpy.mapping.AddLayer(df, layer_to_add, "TOP")

            if PARAMS['gr_names_keys']['TreeTop'] in layer_path:
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'detected_treetops.lyr')
            elif PARAMS['gr_names_keys']['CHMtreeTops'] in layer_path:
                arcpy.BuildPyramids_management(layer_path, skip_existing="SKIP_EXISTING")  ## do not create pyramids if they already exist
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'chm_2_60.lyr')
            elif PARAMS['gr_names_keys']['CHMsegments'] in layer_path:
                arcpy.BuildPyramids_management(layer_path, skip_existing="SKIP_EXISTING")  ## do not create pyramids if they already exist
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'chm_segments_200_6000.lyr')
            elif PARAMS['gr_names_keys']['VegMaskRaw'] in layer_path or PARAMS['gr_names_keys']['VegMaskFilled'] in layer_path:
                arcpy.BuildPyramids_management(layer_path, skip_existing="SKIP_EXISTING")  ## do not create pyramids if they already exist
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'vegetation_mask.lyr')
            elif PARAMS['gr_names_keys']['TreeCrown'] in layer_path:
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'segments_polyg.lyr')

            source_lyr = arcpy.mapping.Layer(source_lyr_path)
            layer_to_update = arcpy.mapping.ListLayers(mxd, layer_to_add.name, df)[0]  ## redefine layer_to_update as an arcpy object
            arcpy.ApplySymbologyFromLayer_management(layer_to_update, source_lyr)  ## first apply the symbology with ApplySymbologyFromLayer_management (needs a path to the layer providing the symbology)
            arcpy.mapping.UpdateLayer(df, layer_to_update, source_lyr, True)  ## update symbology by using the one of source_lyr (has to be an arpy object and not just the path)

        ## Grouping layers
        positions = ['TOP'] * len(PARAMS['gr_names_keys'])
        grouping_dict = {n:[PARAMS['gr_names_keys'][n], positions[i]] for i, n in enumerate(PARAMS['gr_names_keys'])}
        group_arcmap_layers(mxd, df, os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'NewGroupLayer.lyr'), grouping_dict)

        ## Saving final MXD
        mxd.saveACopy(PARAMS['output_mxd'])

    print('Total ' + toc(start_time))



## MAYBE TO USE

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# import rpy2.robjects as robjects

# robjects.r('''
#         library(ForestTools)
#         library(raster)
#         aa <- c(0, 4, 6)
#         ''')
#
# not working
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

## PARAMETERS OF OTHER UNUSED FUNCTIONS
# PARAMS['filter_above'] = 120         ## parameter for lasheight() to remove points above a threshold in m
# PARAMS['filter_below'] = -2          ## parameter for lasheight() to remove points below a threshold in m
# PARAMS['landscape'] = '-city'          ## parameter for lasground_new() defining type of environment (controls the "-step" parameter)
# PARAMS['spike_down'] = 3           ## parameter for lasground_new() defining a threshold to filter downspikes from the TIN
# PARAMS['beam_radius'] = 0.1   ## specify the radius of the laser beam in pit-free CHM generation
# PARAMS['pit_free_levels'] = np.array([0, 2, 5, 10, 15, 20, 25, 30])  ## successive height levels to build pit-free CHM

## DIRECTORIES OF OTHER UNUSED FUNCTIONS
# tile_denoised_dir = os.path.join(PARAMS['temp_dir'], 'tile_denoised_dir')
# tile_ground_dir = os.path.join(PARAMS['temp_dir'], 'tiles_ground')
# tile_classif_dir = os.path.join(PARAMS['temp_dir'], 'tiles_classif')
# tile_ht_filtered_dir = os.path.join(PARAMS['temp_dir'], 'tile_ht_filtered_dir')


## Get polygon centroid
# cursor = arcpy.da.SearchCursor(segments_shp_path, "SHAPE@XY")
# centroid_coords = []
# for feature in cursor:
#     centroid_coords.append(feature[0])
# centroid_coords_df = pd.DataFrame(centroid_coords)

## Write centroids to point shp
# centroids_shp_path = os.path.join(tile_segmented_dir, tile_name + '_segments_centroid.shp')
# point = arcpy.Point()
# pointGeometryList = []
# for pt in centroid_coords:
#     point.X = pt[0]
#     point.Y = pt[1]
#     pointGeometry = arcpy.PointGeometry(point)
#     pointGeometryList.append(pointGeometry)
# arcpy.CopyFeatures_management(pointGeometryList, centroids_shp_path)


# ## Workaround to save updated attribute table without touching the shp
# segment_dbf_path = segments_shp_path.replace('.shp', '.dbf')
# segment_dbf_path_temp = segments_shp_path.replace('.shp', '_temp.dbf')  ## temporary file with a different name wrt the shp
# if os.path.exists(segment_dbf_path):        ## remove old dbf to allow renaming back to the same name once updated
#     os.remove(segment_dbf_path)
# arcpy.da.NumPyArrayToTable(attr_table_df.to_records(index=False), segment_dbf_path_temp)  ## save with temporary name otherwise "RuntimeError: Number of shapes does not match the number of table records"
# os.rename(segment_dbf_path_temp, segment_dbf_path)      ## rename dbf back to original name to match shp name
# # os.remove(segment_dbf_path_temp.replace('dbf', 'cpg'))


## To plot the difference between CHMs after reconstruction, supposed to be markers for watershed or even treetops
# local_max_mask = chm_arr == chm_reconstr_arr
# plt.figure()
# plt.imshow(local_max_mask)
# plt.colorbar()


# ## Use lasgrid as a mosaicking tool: action='mosaic' takes the highest pixel value, to avoid zero pixels at boundary of DSMs ending up in mosaic
# if PARAMS['write_files']:
#     lasgrid(in_dir=tile_dem_dir, key='*PF_DSM*', out_dir=PARAMS['wkg_dir'],
#             out_name=('PF_CHM_fdist%0.2f' % PARAMS['freeze_dist']).replace('.', 'p'),
#             step=PARAMS['step'], action='mosaic', nr_MB_mem=2000, in_ext='asc', verbose=True)
#     filelist = glob.glob(os.path.join(tile_dem_dir, '*.asc'))
#     for f in filelist:
#         os.remove(f)

# ## DTM
# las2dem(in_dir=tile_laz_dir, in_ext='laz', DEM_type='DTM', step=PARAMS['step'],
#         out_dir=tile_dem_dir, hillshaded=False,
#         buffer_mode=PARAMS['buffer_mode_las2dem'],
#         nr_cores=PARAMS['nr_cores'], verbose=True)

# if PARAMS['write_files']:
#     lasgrid(in_dir=tile_dem_dir, key='*DTM*', out_dir=PARAMS['wkg_dir'], out_name='DTM_full',
#             step=PARAMS['step'], action='mosaic', nr_MB_mem=2000, in_ext='asc', verbose=True)
#     filelist = glob.glob(os.path.join(tile_dem_dir, '*.asc'))
#     for f in filelist:
#         os.remove(f)

## TO CLASSIFY POINT CLOUD (NOT NEEDED AS ALREADY PROVIDED AS CLASSIFIED) -------------------------------

## Zip las files to laz
# laszip(in_dir=tile_las_dir, in_ext='laz', out_dir=tile_laz_dir, verbose=True)

## Remove isolated points
# lasnoise(in_dir=tile_laz_dir, in_ext='laz', cell_size=PARAMS['cell_size'], isolated=PARAMS['isolated'],
#          out_dir=tile_denoised_dir, nr_cores=PARAMS['nr_cores'], verbose=False)

# Find the bare-earth points in all tiles (buffer is automatically added, so newly saved tile is bigger)
# lasground_new(in_dir=tile_denoised_dir, in_ext='laz', out_dir=tile_ground_dir,
#               buffer_mode=PARAMS['buffer_mode'], tile_buffer=PARAMS['tile_buffer'],
#               landscape=PARAMS['landscape'], spike_down=PARAMS['spike_down'], nr_cores=PARAMS['nr_cores'],
#               verbose=True)

## Remove low and high outliers that are often just noise (e.g. clouds or birds)
## By default lasheight uses the points classified as ground to construct a TIN and then calculates
# lasheight(in_dir=tile_ground_dir, in_ext='laz', out_dir=tile_ht_filtered_dir,
#           above=PARAMS['filter_above'], below=PARAMS['filter_below'], nr_cores=PARAMS['nr_cores'],
#           buffer_mode=PARAMS['buffer_mode'], verbose=True)

## Identify buildings and trees in all denoised tiles
# lasclassify(in_dir=tile_ht_filtered_dir, in_ext='laz', out_dir=tile_classif_dir,
#             nr_cores=PARAMS['nr_cores'], buffer_mode=PARAMS['buffer_mode'], verbose=True)

# Reuse lastile to remove the buffer from the classified tiles:
# Not needed as we never actually added buffers to the tiles
# lastile(in_dir=tile_classif_dir, in_ext='laz', out_dir=tile_chm_treetops_dir,
#         tile_buffer='remove', nr_cores=PARAMS['nr_cores'], verbose=True)


# #%% CHM WITH PITS ---------------------------------------------------------------------
#
# ## CHM simply as DSM - DTM with numpy
#
# dsm_rast = arcpy.Raster(tile_dem_path + '/' + 'dsm.asc')
# dtm_rast = arcpy.Raster(tile_dem_path + '/' + 'dtm.asc')
#
# dsc = arcpy.Describe(dsm_rast)
# ll=arcpy.Point(dsc.Extent.XMin,dsc.Extent.YMin)  # store lower left corner to ensure the saved ascii is put the right place
#
# dsm_arr = arcpy.RasterToNumPyArray(dsm_rast)
# dtm_arr = arcpy.RasterToNumPyArray(dtm_rast)
#
# chm_arr = dsm_arr - dtm_arr
#
# chm_arr[np.logical_or(chm_arr > 100, chm_arr < 0)] = 0   # replace implausible value with zeros
#
# chm_rast = arcpy.NumPyArrayToRaster(chm_arr, x_cell_size=PARAMS['step'], lower_left_corner=ll)
# arcpy.RasterToASCII_conversion(chm_rast, tile_chm_path + '/' + 'chm_numpy.asc')

## CHM directly from points via lasheight

## Normalize points wrt ground (height value written as z, replacing absolute a.s.l. elevation)
# print('lasheight - chm')
# cmd = '%s/lasheight -i %s/*.laz ' \
#       '-replace_z ' \
#       '-odir %s -odix _height_norm -olaz -cores %d' \
#       % (PARAMS['lastools_path'], tile_classif_path, tile_chm_path, PARAMS['nr_cores'])   # tile_classif_path as input folder to still have buffers to compute meaningful TINs
# os.system(cmd)

# # ## Reuse lastile to remove the buffer from the height normalized tiles
# # print('lastile - remove buffer from height normalized points')
# # cmd = '%s/lastile -i %s/*.laz ' \
# #       '-remove_buffer ' \
# #       '-olaz -odir %s -odix _no_buff -cores %d' \
# #       % (PARAMS['lastools_path'], tile_chm_path, tile_chm_path, PARAMS['nr_cores'])
# # os.system(cmd)
#
# ## NOT WORKING: weird raster written
# #
# # print('lasgrid - chm with pits')
# #
# # cmd = '%s/lasgrid -i %s/*_no_buff.laz -merged ' \    # uses height normalized points without buffers
# #       '-highest -step %.3f -set_min_max 0 10 -fill 2 ' \
# #       '-o %s/chm.asc' \
# #       % (PARAMS['lastools_path'], tile_chm_path, PARAMS['step'], tile_chm_path)
# # os.system(cmd)
# ## NOT WORKING

## PIT-FREE CHM  ---------------------------------------------------------------------
#
# sub_step = PARAMS['step'] / 2  # sub-resolution for the beam file (recommended as about half the step)
# kill = PARAMS['step'] * 3  # desired triangle kill (recommended as about 3 times the step)
#
# print('lasgrid chm without pits')
#
# ## First, we thin the point cloud (height normalized points WITH buffers) by keeping only the highest return
# ## (splatted into a circle to account for the laser beam diameter -subcircle) per grid cell of size -step
# cmd = '%s/lasthin -i %s/*.laz ' \
#       '-highest -step %.3f -subcircle %.3f ' \
#       '-odir %s -odix _beam -olaz -cores %d' \
#       % (PARAMS['lastools_path'], tile_chm_path, sub_step, PARAMS['beam_radius'], temp_path, PARAMS['nr_cores'])
# os.system(cmd)
#
# ## Then, we create a series of CHMs only using points above a set of thresholds PARAMS['pit_free_levels']
# ## (assigned to -drop_z_below with level) by considering only points within the tile WITHOUT buffer (-use_tile_bb)
# for level in PARAMS['pit_free_levels']:
#     cmd = '%s/las2dem -i %s/*.laz -use_tile_bb ' \
#           '-drop_z_below %d -step %.3f -kill %.3f ' \
#           '-odir %s -odix _%d -oasc -cores %d' \
#           % (PARAMS['lastools_path'], temp_path, level, PARAMS['step'], kill, temp_path, level, PARAMS['nr_cores'])
#     os.system(cmd)
#
# ## Then, we merge the partial CHMs together into a pit-free CHM by keeping the maximum value across all rasters
# os.chdir(temp_path)
# tiles = glob.glob('*.laz')
# tile_names = [s.strip('.laz') for s in tiles]
# for tile_name in tile_names:
#     cmd = '%s/lasgrid -i %s/%s_*.asc -merged ' \
#               '-step %.3f -highest ' \
#               '-o %s/%s_chm_pit_free.asc' \
#               % (PARAMS['lastools_path'], temp_path, tile_name, PARAMS['step'], tile_chm_path, tile_name)
#     os.system(cmd)
#
#
# # shutil.rmtree(temp_path.replace('/', '\\'))
#
# ## Finally, we stitch together the tiles
# cmd = '%s/las2dem -i %s/*_chm_pit_free.asc -merged ' \
#       '-step %.3f ' \
#       '-o %s/chm_pit_free.asc' \
#       % (PARAMS['lastools_path'], tile_chm_path, PARAMS['step'], tile_chm_path)
# os.system(cmd)


## DSMs AND HILLSHADES -----------------------------------------------------------------------

## Use actual LiDAR ground points from final tiles (i.e. those without buffer) to create merged raster
## DTMs ('-keep_class 2' to retain only ground points) and DSMs ('-first_only' to retain first returns only)
## as well as corresponding hillshades in ESRI ASCII format (*.asc)

# ## DSM
# las2dem(in_dir=tile_ht_norm_dir, in_ext='laz', DEM_type='DSM', step=PARAMS['step'],
#         out_dir=tile_dem_dir, hillshaded=False, buffer_mode=PARAMS['buffer_mode_las2dem'],
#         nr_cores=PARAMS['nr_cores'], verbose=True)
# lasgrid(in_dir=tile_dem_dir, key='*DSM*', out_dir=PARAMS['wkg_dir'], out_name='DSM_full',
#         step=PARAMS['step'], nr_MB_mem=2000, in_ext='asc', verbose=True)
# filelist = glob.glob(os.path.join(tile_dem_dir, '*.asc'))
# for f in filelist:
#     os.remove(f)
#
# ## Hillshaded DSM
# las2dem(in_dir=tile_ht_norm_dir, in_ext='laz', DEM_type='DSM', step=PARAMS['step'],
#         out_dir=tile_dem_dir, hillshaded=True, buffer_mode=PARAMS['buffer_mode_las2dem'],
#         nr_cores=PARAMS['nr_cores'], verbose=True)
# lasgrid(in_dir=tile_dem_dir, key='*DSM*', out_dir=PARAMS['wkg_dir'], out_name='HDSM_full',
#         step=PARAMS['step'], nr_MB_mem=2000, in_ext='asc', verbose=True)
#
#
# for fdist in PARAMS['freeze_dist']:
#
#     print('Freeze distance: %0.2f' % fdist)
#
#     ## Hillshaded Pit-free DSM
#     las2dem(in_dir=tile_ht_norm_dir, in_ext='laz', DEM_type='PF_DSM', step=PARAMS['step'],
#             freeze_dist=fdist, out_dir=tile_dem_dir, hillshaded=True, buffer_mode=PARAMS['buffer_mode_las2dem'],
#             nr_cores=PARAMS['nr_cores'], verbose=True)
#     lasgrid(in_dir=tile_dem_dir, key='*PF_DSM*', out_dir=PARAMS['wkg_dir'],
#             out_name=('Hillsh_PF_DSM_full_fdist%0.2f' % fdist).replace('.', 'p'),
#             step=PARAMS['step'], nr_MB_mem=2000, in_ext='asc', verbose=True)
#     filelist = glob.glob(os.path.join(tile_dem_dir, '*.asc'))
#     for f in filelist:
#         os.remove(f)
#
#     ## Pit-free DSM_Veg
#     las2dem(in_dir=tile_ht_norm_dir, in_ext='laz', DEM_type='PF_DSM_Veg', step=PARAMS['step'],
#             freeze_dist=fdist, out_dir=tile_dem_dir, hillshaded=False, buffer_mode=PARAMS['buffer_mode_las2dem'],
#             nr_cores=PARAMS['nr_cores'], verbose=True)
#     lasgrid(in_dir=tile_dem_dir, key='*PF_DSM_Veg*', out_dir=PARAMS['wkg_dir'],
#             out_name=('PF_DSM_Veg_full_fdist%0.2f' % fdist).replace('.', 'p'), step=PARAMS['step'], nr_MB_mem=2000, in_ext='asc', verbose=True)
#     filelist = glob.glob(os.path.join(tile_dem_dir, '*.asc'))
#     for f in filelist:
#         os.remove(f)
#
#     ## Hillshaded Pit-free DSM_Veg
#     las2dem(in_dir=tile_ht_norm_dir, in_ext='laz', DEM_type='PF_DSM_Veg', step=PARAMS['step'],
#             freeze_dist=fdist, out_dir=tile_dem_dir, hillshaded=True, buffer_mode=PARAMS['buffer_mode_las2dem'],
#             nr_cores=PARAMS['nr_cores'], verbose=True)
#     lasgrid(in_dir=tile_dem_dir, key='*PF_DSM_Veg*', out_dir=PARAMS['wkg_dir'],
#             out_name=('Hillsh_PF_DSM_Veg_full_fdist%0.2f' % fdist).replace('.', 'p'), step=PARAMS['step'],
#             nr_MB_mem=2000, in_ext='asc', verbose=True)
#     filelist = glob.glob(os.path.join(tile_dem_dir, '*.asc'))
#     for f in filelist:
#         os.remove(f)
#
# ## Hillshaded DTM
# las2dem(in_dir=tile_classif_dir, in_ext='laz', DEM_type='DTM', step=PARAMS['step'],
#         out_dir=tile_hillsh_dir, hillshaded=True, buffer_mode=PARAMS['buffer_mode_las2dem'],
#         nr_cores=PARAMS['nr_cores'], verbose=True)
# lasgrid(in_dir=tile_hillsh_dir, key='*DTM*', out_dir=PARAMS['wkg_dir'], out_name='HDTM_full',
#         step=PARAMS['step'], nr_MB_mem=2000, in_ext='asc', verbose=True)

# ## Hillshaded DSM_Veg
# las2dem(in_dir=tile_classif_dir, in_ext='laz', DEM_type='DSM_Veg', step=PARAMS['step'],
#         out_dir=tile_hillsh_dir, hillshaded=True, buffer_mode=PARAMS['buffer_mode_las2dem'],
#         nr_cores=PARAMS['nr_cores'], verbose=True)
# lasgrid(in_dir=tile_hillsh_dir, key='*DSM_Veg*', out_dir=PARAMS['wkg_dir'], out_name='HDSM_Veg_full',
#         step=PARAMS['step'], nr_MB_mem=2000, in_ext='asc', verbose=True)

