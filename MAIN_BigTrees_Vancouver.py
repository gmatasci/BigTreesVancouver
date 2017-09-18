"""
Project Name: BigTreesVan
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: MAIN_BigTrees_Vancouver.py
Objective: Build high-res CHM from LiDAR for the city of Vancouver, detect treetops, segment crowns and extract attributes
"""

## TO DO -------------------------------------------------------------------

## STILL TO DO:

# - grid search on key parameters

# Prior to actual run:
# - check # mask_filled
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
# - watershed segmentation with markers: python vs OTB -- not done as we keep tiles (no need for tile-merging utility of OTB) and as it seems OTB cannot do segmentation with markers
# - treetop finder varying with height (from R?) -- too many parameters to fit
# - the tip of tall trees is classified as "unclassified", so vegetation mask has holes -- run LAStools classification with lasclassify
# - running gradient on mean CHM or on raw CHM does not improve results: same elongated objects are possible (mean CHM bc is very similar to open by reconstruction mean chm, raw CHM bc too many details)

## IMPORT MODULES ---------------------------------------------------------------


from __future__ import division  # to allow floating point divisions
import os
import sys
import glob
import shutil
import time
import arcpy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.morphology import *
from skimage.segmentation import *
from skimage.feature import *
from skimage.filters import rank
import json

from Functions_BigTrees_Vancouver import*
from Functions_LAStools import*

if __name__ == '__main__':

    mpl.rc('image', interpolation='none')   ## to change ugly default behavior of image/imshow plots

## PARAMETERS ----------------------------------------------------------------

    PARAMS = {}

    PARAMS['lidar_processing'] = True
    # PARAMS['lidar_processing'] = False

    PARAMS['raster_processing'] = True
    # PARAMS['raster_processing'] = False

    PARAMS['build_CHMs'] = True
    # PARAMS['build_CHMs'] = False

    PARAMS['segment_crowns'] = True
    # PARAMS['segment_crowns'] = False

    PARAMS['build_MXD'] = True
    # PARAMS['build_MXD'] = False

    # PARAMS['layers_in_MXD'] = 'all'
    PARAMS['layers_in_MXD'] = 'essentials'

    PARAMS['write_files'] = True
    # PARAMS['write_files'] = False

    # PARAMS['plot_figures'] = True
    PARAMS['plot_figures'] = False

    PARAMS['base_dir'] = r'D:\Research\ANALYSES\BigTreesVan'

    # PARAMS['dataset_name'] = 'Vancouver_500m_tiles'
    # PARAMS['dataset_name'] = 'Tune_alg_8tiles'
    # PARAMS['dataset_name'] = 'Tune_alg_1subtile_QE'
    PARAMS['dataset_name'] = 'Tune_alg_1tile_QE'

    PARAMS['experiment_name'] = 'step_0p3m_mindist_8_FINAL'

    #### TODO TO LAZ and remove unzipped las
    # PARAMS['data_dir'] = r'E:\BigTreesVan_data\LiDAR\CoV\Classified_LiDAR'
    #### TODO LAZ and remove unzipped las
    PARAMS['data_dir'] = os.path.join(PARAMS['base_dir'], r'wkg\trial_data_'+PARAMS['dataset_name'])

    PARAMS['wkg_dir'] = os.path.join(PARAMS['base_dir'], 'wkg')
    PARAMS['exp_dir'] = os.path.join(PARAMS['wkg_dir'], PARAMS['dataset_name'], PARAMS['experiment_name'])

    PARAMS['input_mxd'] = os.path.join(PARAMS['base_dir'], 'mxds', r'VanBigTrees_empty.mxd')  ## path to empty mxd that has to have a scale bar (will adapt automatically) and bookmarks already set
    PARAMS['output_mxd'] = os.path.join(PARAMS['exp_dir'], r'VanBigTrees_'+PARAMS['dataset_name']+'_'+PARAMS['experiment_name']+'.mxd')   ## path to final mxd to save results

    PARAMS['nr_cores'] = 32    ## number of cores to be used by lastools functions

    PARAMS['tile_size'] = 500      ## tile size for lidar processing with lastools (half the tile size of tiles received from City of Vancouver)

    PARAMS['tile_buffer'] = 30      ## tile buffer to avoid boundary problems (we expect no tree crown to be bigger than this)

    PARAMS['step'] = 0.3  ## 0.3, pixel size of raster layers (DSM, CHM, masks, etc.)

    PARAMS['cell_size'] = 2    ## lasnoise parameter to remove powerlines: size in meters of each voxel of the 3x3 voxel neighborhood
    PARAMS['isolated'] = 50     ## lasnoise parameter to remove powerlines: remove points that have less neighboring points than this threshold in the 3x3 voxel neighborhood

    PARAMS['DEM_type'] = 'PF_DSM'  ## las2dem parameter defining the type of DSM created
    PARAMS['freeze_dist'] = 0.8   ## las2dem spike-free CHM parameter
    PARAMS['veg_ht_thresh'] = 3   ## height threshold to filter low vegetation and other structures from masked DSM

    PARAMS['subcircle'] = 0.2   ## lasgrid parameter in vegetation mask to thicken point by adding a discrete ring of 8 new points that form a circle with radius "subcircle"

    PARAMS['disk_radius'] = 4  ## radius of structuring element for morphological operations (opening(), reconstruction()) on vegetation mask and chm: radius of 5 means a 5*2+1 diameter disk
    PARAMS['min_size_holes'] = 300  ## number of contiguous 0 pixels in the vegetation mask to be filled by remove_small_holes()

    PARAMS['min_distance_peaks'] = 8   ## minimum separation in pixels between treetops in peak_local_max() and corner_peaks(), i.e. 4 pix = 4*0.5 = 2 m @ 0.5 m spatial resolution

    PARAMS['compactness'] = 1   ## segment compactness parameter for watershed segmentation (between 0 and 1)

    PARAMS['tree_ht_thresh'] = 15   ## height threshold in meters below which we remove segmented trees
    PARAMS['crown_dm_thresh'] = 40  ## crown diameter threshold in meters above which the polygon is no more a tree but background instead

    PARAMS['gr_names_keys'] = {'VegMaskRaw': 'veg_mask_raw',
                               'VegMaskFilled': 'veg_mask_filled',
                               'CHMtreeTops': 'chm_raw',
                               'CHMsegments': 'chm_mean',
                               'TreeCrown': 'segments_polyg',
                               'TreeTop': 'tree_tops'}   ## MXD group names and corresponding file key for glob.glob()


## START ---------------------------------------------------------------------

    print(python_info())

    print('MAIN_BigTrees_Vancouver.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    print('Dataset: %s \nExperiment: %s' % (PARAMS['dataset_name'], PARAMS['experiment_name']))

    start_time = tic()

    arcpy.CheckOutExtension("3D")
    arcpy.CheckOutExtension("spatial")
    arcpy.env.overwriteOutput = True

    arcgis_temp_dir = os.path.join(PARAMS['exp_dir'], 'arcgis_temp')
    tile_las_dir = os.path.join(PARAMS['data_dir'], 'las')
    tile_laz_dir = os.path.join(PARAMS['data_dir'], 'laz')
    tile_denoised_dir = os.path.join(PARAMS['exp_dir'], 'tiles_denoised')
    tile_ht_norm_dir = os.path.join(PARAMS['exp_dir'], 'tiles_ht_norm')
    tile_ht_norm_classif_dir = os.path.join(PARAMS['exp_dir'], 'tile_ht_norm_classif_dir')
    tile_dem_dir = os.path.join(PARAMS['exp_dir'], 'tiles_dem')
    tile_mask_dir = os.path.join(PARAMS['exp_dir'], 'tiles_mask')
    tile_chm_treetops_dir = os.path.join(PARAMS['exp_dir'], 'tiles_chm_treetops')
    tile_segmented_dir = os.path.join(PARAMS['exp_dir'], 'tiles_segmented')

    dirs = [arcgis_temp_dir,
            tile_las_dir,
            tile_laz_dir,
            tile_denoised_dir,
            tile_ht_norm_dir,
            tile_ht_norm_classif_dir,
            tile_dem_dir,
            tile_mask_dir,
            tile_chm_treetops_dir,
            tile_segmented_dir
            ]

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    params_filename = 'PARAMS_MAIN_%s_%s.json' % (PARAMS['dataset_name'], PARAMS['experiment_name'])
    with open(os.path.join(PARAMS['exp_dir'], params_filename), 'w') as fp:
        json.dump(PARAMS, fp)

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

        lasclassify(in_dir=tile_ht_norm_dir, out_dir=tile_ht_norm_classif_dir, nr_cores=PARAMS['nr_cores'], verbose=True)

        ## Pit-free DSM (from height normalized point cloud)
        las2dem(in_dir=tile_ht_norm_classif_dir, in_ext='laz', DEM_type=PARAMS['DEM_type'], step=PARAMS['step'],
                freeze_dist=PARAMS['freeze_dist'], out_dir=tile_dem_dir, hillshaded=False,
                nr_cores=PARAMS['nr_cores'], verbose=True)

        ## For tiles in which Pit-free DSM fails, recompute it with default method using first return only
        files_to_recompute = [file for file in glob.glob(tile_dem_dir+'/*.asc') if os.path.getsize(file) == 0]
        if len(files_to_recompute) > 0:
            odix = ('_PF_DSM_%.2f' % PARAMS['freeze_dist']).replace('.', 'p')
            for file in files_to_recompute:
                tile_name = file.split('\\')[-1].split('_')[0]+'_'+file.split('\\')[-1].split('_')[1]  ## root to rename all tile-based layers
                ht_norm_tile = glob.glob(os.path.join(tile_ht_norm_classif_dir, tile_name+'*'))[0]
                cmd = 'las2dem -i %s ' \
                      '-first_only -extra_pass -step %.2f ' \
                      '-oasc -odir %s -odix %s' \
                      % (ht_norm_tile, PARAMS['step'], tile_dem_dir, odix)
                os.system(cmd)

        ## Lasgrid in classif mode to keep high and mid vegetation (no class 4 in Vancouver data: mid vegetation)
        lasgrid(in_dir=tile_ht_norm_classif_dir, key='*', in_ext='laz', out_dir=tile_mask_dir, out_name='_'+PARAMS['gr_names_keys']['VegMaskRaw'],
                step=PARAMS['step'], action='mask', classes_to_keep='4 5',
                subcircle=PARAMS['subcircle'], nr_MB_mem=2000,
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

                ## Remove all tile files in folders if tile does not contain vegetation
                if sum(sum(mask_arr)) == 0:
                    dirs = [tile_denoised_dir,
                            tile_ht_norm_dir,
                            tile_ht_norm_classif_dir,
                            tile_dem_dir,
                            tile_mask_dir,
                            tile_chm_treetops_dir
                            ]
                    for dir in dirs:
                        files_to_remove = glob.glob(os.path.join(dir, tile_name+'*'))
                        for file in files_to_remove:
                            try:
                                os.remove(file)   ## if it is a raster file that is being loaded by Arcpy this will fail...
                            except:
                                arcpy.Delete_management(file)  ## ...so delete it with Arcpy
                    continue

                ## Denoise vegetation mask with morphological opening
                mask_open_arr = binary_opening(mask_arr, selem=disk(PARAMS['disk_radius']))
                mask_filled_arr = remove_small_holes(mask_open_arr, min_size=PARAMS['min_size_holes']).astype(int)

                ## Build raw CHM (the one height values must be read from) by masking out non-vegetation DSM pixels and pixels lower than a height threshold
                chm_raw_arr = dsm_arr * mask_filled_arr

                chm_raw_arr[chm_raw_arr < PARAMS['veg_ht_thresh']] = 0   ## sets to 0 pixels lower than threshold and -9999 at the borders

                if PARAMS['plot_figures']:
                    plt.figure(), plt.title('Veg. mask'), plt.imshow(mask_arr), plt.colorbar()
                    plt.figure(), plt.title('Veg. mask opened'), plt.imshow(mask_open_arr), plt.colorbar()
                    plt.figure(), plt.title('Veg. mask filled'), plt.imshow(mask_filled_arr), plt.colorbar()
                    plt.figure(), plt.title('CHM raw'), plt.imshow(chm_raw_arr), plt.colorbar()

                if PARAMS['write_files']:
                    chm_raw = array_2_raster(chm_raw_arr, spatial_info)
                    arcpy.RasterToASCII_conversion(chm_raw, chm_raw_path)
                    mask_filled_towrite_arr = np.copy(mask_filled_arr)
                    mask_filled_towrite_arr[mask_filled_towrite_arr == 0] = -9999
                    mask_filled_towrite = array_2_raster(mask_filled_towrite_arr, spatial_info)
                    arcpy.RasterToASCII_conversion(mask_filled_towrite, os.path.join(tile_mask_dir, tile_name + '_' + PARAMS['gr_names_keys']['VegMaskFilled'] + '.asc'))

            else:

                if os.path.exists(chm_raw_path):
                    chm_raw = arcpy.Raster(chm_raw_path)
                    chm_raw_arr, spatial_info = raster_2_array(chm_raw)

            #### TREETOP DETECTION -------------------------------------------------------------------

            if PARAMS['segment_crowns']:

                ## Build integer CHM for faster image processing
                chm_raw_arr_int = chm_raw_arr * 100  ## multiply by 100 to keep 2 decimal when converting to int for rank.median()
                chm_raw_arr_int = chm_raw_arr_int.astype(np.uint16)

                ## Denoise CHM to avoid high values for trees whose treetop is close to high buildings
                chm_median_arr = rank.median(chm_raw_arr_int, disk(2))

                ## Smooth CHM
                chm_mean_arr = rank.mean(chm_median_arr, disk(1))

                if PARAMS['write_files']:
                    chm_mean = array_2_raster(chm_mean_arr, spatial_info)
                    arcpy.RasterToASCII_conversion(chm_mean, os.path.join(tile_chm_treetops_dir, tile_name+'_'+PARAMS['gr_names_keys']['CHMsegments']+'.asc'))

                ## Detect local maxima on median filtered CHM: results in true tree tops
                local_max_arr = corner_peaks(chm_mean_arr, min_distance=PARAMS['min_distance_peaks'], indices=False)  ## corner_peaks() respects the min_distance criterion, instead peak_local_max() does not

                if PARAMS['plot_figures']:
                    plt.figure(), plt.title('CHM raw'), plt.imshow(chm_raw_arr_int, vmin=0, vmax=3600), plt.colorbar()
                    plt.figure(), plt.title('CHM filtered'), plt.imshow(chm_median_arr, vmin=0, vmax=3600), plt.colorbar()
                    plt.figure(), plt.title('CHM filtered and smoothed'), plt.imshow(chm_mean_arr, vmin=0, vmax=3600), plt.colorbar()
                    plt.figure(), plt.title('Local max'), plt.imshow(dilation(local_max_arr, selem=disk(2))), plt.colorbar()

                if PARAMS['write_files']:
                    local_max_visual = array_2_raster(dilation(local_max_arr, selem=disk(2)).astype(int), spatial_info)
                    arcpy.RasterToASCII_conversion(local_max_visual, os.path.join(tile_chm_treetops_dir, tile_name+'_' + PARAMS['gr_names_keys']['TreeTop'] + '.asc'))

                ## Watershed segmentation
                chm_mean_arr_int = chm_mean_arr.astype(np.uint16)  ## recast as integer the input to watershed() has to be an int image
                gradient_arr = rank.gradient(chm_mean_arr_int, selem=disk(1))     ## compute gradient image with very thin edges (disk(1))
                markers_arr = local_max_arr.astype(int)
                markers_arr[local_max_arr] = np.arange(1, sum(sum(local_max_arr)) + 1)  ## equivalent to the following but without repeated labels for contiguous pixels: ndi.label(local_max_arr.astype(int))[0]  ## assign unique labels to each marker to start the segmentation (same labels will be assigned to segments)
                markers_dilat_arr = dilation(markers_arr, selem=disk(1))  ## thicken markers with a disk to avoid problems if local max is at the edge
                segments_arr = watershed(image=gradient_arr, markers=markers_dilat_arr, mask=mask_filled_arr, compactness=PARAMS['compactness'])   ## use vegetation mask as mask to avoid boundaries being drawn beyond vegetation limits

                if PARAMS['plot_figures']:
                    plt.figure(), plt.title('CHM mean'), plt.imshow(chm_mean_arr_int), plt.colorbar()
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


        if PARAMS['layers_in_MXD'] == 'essentials':
            groups = {k: PARAMS['gr_names_keys'][k] for k in ('CHMtreeTops', 'TreeTop', 'TreeCrown')}
            layer_paths = glob.glob(os.path.join(tile_chm_treetops_dir, '*_' + PARAMS['gr_names_keys']['CHMtreeTops'] + '.asc'))
            layer_paths.extend(glob.glob(os.path.join(tile_chm_treetops_dir, '*_' + PARAMS['gr_names_keys']['TreeTop'] + '.asc')))
            layer_paths.extend(glob.glob(os.path.join(tile_segmented_dir, '*_' + PARAMS['gr_names_keys']['TreeCrown'] + '.shp')))
        elif PARAMS['layers_in_MXD'] == 'all':
            groups = PARAMS['gr_names_keys']
            layer_paths = glob.glob(os.path.join(tile_mask_dir, '*_'+PARAMS['gr_names_keys']['VegMaskFilled']+'.asc'))
            layer_paths.extend(glob.glob(os.path.join(tile_mask_dir, '*_' + PARAMS['gr_names_keys']['VegMaskRaw'] + '.asc')))
            layer_paths.extend(glob.glob(os.path.join(tile_chm_treetops_dir, '*_'+PARAMS['gr_names_keys']['CHMtreeTops']+'.asc')))
            layer_paths.extend(glob.glob(os.path.join(tile_chm_treetops_dir, '*_'+PARAMS['gr_names_keys']['CHMsegments']+'.asc')))
            layer_paths.extend(glob.glob(os.path.join(tile_chm_treetops_dir,'*_'+PARAMS['gr_names_keys']['TreeTop']+'.asc')))
            layer_paths.extend(glob.glob(os.path.join(tile_segmented_dir, '*_'+PARAMS['gr_names_keys']['TreeCrown']+'.shp')))

        ## Loop over the tiles in tile_chm_treetops_dir
        for i_layer, layer_path in enumerate(layer_paths):

            layer_name = layer_path.split('\\')[-1]

            layer_to_add = arcpy.mapping.Layer(layer_path)
            arcpy.mapping.AddLayer(df, layer_to_add, "TOP")

            if PARAMS['gr_names_keys']['TreeTop'] in layer_name:
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'detected_treetops.lyr')
            elif PARAMS['gr_names_keys']['CHMtreeTops'] in layer_name:
                arcpy.BuildPyramids_management(layer_path, skip_existing="SKIP_EXISTING")  ## do not create pyramids if they already exist
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'chm_2_60.lyr')
            elif PARAMS['gr_names_keys']['CHMsegments'] in layer_name:
                arcpy.BuildPyramids_management(layer_path, skip_existing="SKIP_EXISTING")  ## do not create pyramids if they already exist
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'chm_segments_200_6000.lyr')
            elif PARAMS['gr_names_keys']['VegMaskRaw'] in layer_name or PARAMS['gr_names_keys']['VegMaskFilled'] in layer_name:
                arcpy.BuildPyramids_management(layer_path, skip_existing="SKIP_EXISTING")  ## do not create pyramids if they already exist
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'vegetation_mask.lyr')
            elif PARAMS['gr_names_keys']['TreeCrown'] in layer_name:
                source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'segments_polyg.lyr')

            source_lyr = arcpy.mapping.Layer(source_lyr_path)
            layer_to_update = arcpy.mapping.ListLayers(mxd, layer_to_add.name, df)[0]  ## redefine layer_to_update as an arcpy object
            arcpy.ApplySymbologyFromLayer_management(layer_to_update, source_lyr)  ## first apply the symbology with ApplySymbologyFromLayer_management (needs a path to the layer providing the symbology)
            arcpy.mapping.UpdateLayer(df, layer_to_update, source_lyr, True)  ## update symbology by using the one of source_lyr (has to be an arpy object and not just the path)

        ## Grouping layers
        positions = ['TOP'] * len(groups)
        grouping_dict = {n:[groups[n], positions[i]] for i, n in enumerate(groups)}
        group_arcmap_layers(mxd, df, os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'NewGroupLayer.lyr'), grouping_dict)

        ## Saving final MXD
        mxd.saveACopy(PARAMS['output_mxd'])

    print('Total ' + toc(start_time))