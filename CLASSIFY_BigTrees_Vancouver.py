"""
Project Name: BigTreesVan
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: CLASSIFY_BigTrees_Vancouver.py
Objective: Classify segmented tree crowns into deciduous or coniferous and assess classification accuracy
"""


## IMPORT MODULES ---------------------------------------------------------------


from __future__ import division  # to allow floating point divisions
import os
import sys
import glob
import time
import logging
import arcpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from simpledbf import Dbf5

from Functions_BigTrees_Vancouver import*
from Functions_LAStools import*

if __name__ == '__main__':

    mpl.rc('image', interpolation='none')   ## to change ugly default behavior of image/imshow plots

## PARAMETERS ----------------------------------------------------------------

    PARAMS = {}

    PARAMS['lidar_preprocessing'] = True  ## switch to run the sorting and indexing of the tiles

    PARAMS['build_dataset'] = False   ## switch to run the long loop reading the GT in each tile

    PARAMS['train_rf_model'] = False  ## switch to run the training of the model and save it, or, alternatively to load it from a file

    PARAMS['predict_on_tiles'] = True  ## switch to run the prediction with the RF model on all the tiles

    PARAMS['load_in_MXD'] = True  ## switch to load the final result into the MXD project

    PARAMS['base_dir'] = r'D:\Research\ANALYSES\BigTreesVan'

    PARAMS['dataset_name'] = 'Vancouver_500m_tiles'
    PARAMS['experiment_name'] = 'step_0p3m_mindist_8_mask_5m'

    PARAMS['wkg_dir'] = os.path.join(PARAMS['base_dir'], 'wkg')
    PARAMS['exp_dir'] = os.path.join(PARAMS['wkg_dir'], PARAMS['dataset_name'], PARAMS['experiment_name'])

    PARAMS['output_mxd'] = os.path.join(PARAMS['exp_dir'], r'VanBigTrees_'+PARAMS['dataset_name']+'_'+PARAMS['experiment_name']+'.mxd')   ## path to final mxd to save results

    PARAMS['ref_trees_path'] = r'E:\BigTreesVan_data\GroundTruth\reference_trees.shp'

    PARAMS['ht_filter_colname'] = 'Tree_ht'   ## column name to use for filtering based on the height values
    PARAMS['tree_ht_thresh'] = 15   ## height threshold in meters below which we remove the trees to be used in the classification

    PARAMS['train_pct'] = 0.70    ## percentage of samples being assigned to the training set

    PARAMS['ht_cutoff'] = 2   ## height cutoff in m to be used in lascanopy to compute lidar metrics
    PARAMS['bicentiles_upper'] = 99   ## centile of height to use as upper value in the computation of the bicentiles

    PARAMS['lidar_metrics'] = '-cov -dns -ske -kur -avg -std -qav -p 10 20 30 40 50 60 70 80 90 95 99 -b 10 20 30 40 50 60 70 80 90 95'   ## string with lascanopy commands to extract metrics
    PARAMS['to_normalize'] = ['avg', 'std', 'qav', 'p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p95']   ## list of CSV column names of metrics to be normalized (by the 99th centile)
    PARAMS['feature_names'] = ['cov', 'dns', 'ske', 'kur', 'norm_avg', 'norm_std', 'norm_qav', 'norm_p10', 'norm_p20',  'norm_p30', 'norm_p40', 'norm_p50', 'norm_p60', 'norm_p70', 'norm_p80', 'norm_p90', 'norm_p95', 'b10', 'b20',  'b30', 'b40', 'b50', 'b60', 'b70', 'b80', 'b90', 'b95']   ## final feature names to feed to the RF

    PARAMS['ntrees'] = 1000    ## number of RF trees
    PARAMS['rf_model_path'] = os.path.join(PARAMS['exp_dir'], 'rf_model_%d.pkl' % PARAMS['ntrees'])  ## path at which to save the RF model file

    PARAMS['nr_cores'] = 32   ## number of cores to be used by lastools functions and by RF


## START ---------------------------------------------------------------------


    print(python_info())

    print('CLASSIFY_BigTrees_Vancouver.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    start_time = tic()

    params_filename = 'PARAMS_CLASSIFY_%s_%s.json' % (PARAMS['dataset_name'], PARAMS['experiment_name'])
    with open(os.path.join(PARAMS['exp_dir'], params_filename), 'w') as fp:
        json.dump(PARAMS, fp)

    arcpy.CheckOutExtension("3D")
    arcpy.CheckOutExtension("spatial")
    arcpy.env.overwriteOutput = True

    arcgis_temp_dir = os.path.join(PARAMS['exp_dir'], 'arcgis_temp')
    tile_ht_norm_classif_dir = os.path.join(PARAMS['exp_dir'], 'tiles_ht_norm_classif')
    tile_ht_norm_classif_sorted_dir = os.path.join(PARAMS['exp_dir'], 'tiles_ht_norm_classif', 'sorted')
    tile_chm_treetops_dir = os.path.join(PARAMS['exp_dir'], 'tiles_chm_treetops')
    tile_segmented_dir = os.path.join(PARAMS['exp_dir'], 'tiles_segmented')
    tile_lidar_metrics_dir = os.path.join(PARAMS['exp_dir'], 'tiles_lidar_metrics')
    tile_classified_dir = os.path.join(PARAMS['exp_dir'], 'tiles_classified_FINAL')

    for dir in [tile_lidar_metrics_dir, tile_ht_norm_classif_sorted_dir, tile_classified_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    ## Sort and reindex lidar tiles to speed up massively the extraction of metrics with lascanopy
    if PARAMS['lidar_preprocessing']:
        lassort(in_dir=tile_ht_norm_classif_dir, out_dir=tile_ht_norm_classif_sorted_dir, in_ext='laz', nr_cores=PARAMS['nr_cores'], verbose=True)
        lasindex(in_dir=tile_ht_norm_classif_sorted_dir, in_ext='laz', nr_cores=PARAMS['nr_cores'], verbose=True)

    ## List file paths of all tiles
    file_key = os.path.join(tile_segmented_dir, '*_segments_polyg.shp')
    segments_paths = glob.glob(file_key)

    if PARAMS['build_dataset']:

        polyg_ok_bool = {t: [] for t in range(len(segments_paths))}   ## empty dictionary to store boolean indices for each tile telling which polygons have positive areas
        classif_df = pd.DataFrame()

        ## Loop over segment tiles (not over point cloud tiles bc some could be without vegetation)
        for i_tile, _ in enumerate(segments_paths):

            ## Get tile name
            filename = segments_paths[i_tile].split('\\')[-1]
            tile_name = filename.split('_')[0] + '_' + filename.split('_')[1]

            print('Extracting metrics and reading GT data on tile %s (%d/%d)' % (tile_name, i_tile + 1, len(segments_paths)))

            result = arcpy.GetCount_management(segments_paths[i_tile])
            count = int(result.getOutput(0))
            if count == 0:
                continue

            segments_classified_path = os.path.join(tile_classified_dir, tile_name+'_segments_classif.shp')
            arcpy.CopyFeatures_management(segments_paths[i_tile], segments_classified_path)

            ## Add segmented crown ID field to attribute table
            fields = ['Crown_ID']
            for field in fields:
                arcpy.AddField_management(segments_classified_path, field, "TEXT", field_length=60)

            with arcpy.da.UpdateCursor(segments_classified_path, fields) as cursor:  ## Create update cursor for feature class
                for irow, row in enumerate(cursor):
                    for icol, field in enumerate(fields):
                        row[icol] = 'tile_%s_seg_%d' % (tile_name, irow)
                    cursor.updateRow(row)

            ## Save lascanopy() results in CSV file
            csv_path = os.path.join(tile_lidar_metrics_dir, tile_name + '_lidar_metrics.csv')
            pt_cloud_path = os.path.join(tile_ht_norm_classif_sorted_dir, tile_name + '_denoised_ht_norm_classif_sorted.laz')

            polyg_ok_bool[i_tile] = lascanopy(in_laz=pt_cloud_path, in_shp=segments_classified_path, out_csv=csv_path,
                                              ht_cutoff=PARAMS['ht_cutoff'], bicentiles_upper=PARAMS['bicentiles_upper'],
                                              metric_cmd=PARAMS['lidar_metrics'], verbose=False)

            ## Read back files and keep only polygons with positive area
            lidar_metric_df = pd.read_csv(csv_path)
            lidar_metric_df = lidar_metric_df[polyg_ok_bool[i_tile]]

            ## Join lidar metrics and segments df to then allow merging based on ID
            segments_df = Dbf5(segments_classified_path.replace('shp', 'dbf')).to_dataframe()

            lidar_metric_segments_df = pd.concat([lidar_metric_df.reset_index(drop=True), segments_df], axis=1)

            ## Spatial join to check which reference trees are contained in each segment
            arcpy.SpatialJoin_analysis(segments_classified_path, PARAMS['ref_trees_path'],
                                       os.path.join(arcgis_temp_dir, "ref_trees_segments_join_classif.shp"), "JOIN_ONE_TO_MANY",
                                       "KEEP_COMMON", "", "CONTAINS")

            ## If tile does not overlap any reference tree skip to next tile
            if arcpy.management.GetCount(os.path.join(arcgis_temp_dir, "ref_trees_segments_join_classif.shp"))[0] == "0":
                continue

            ## Read attribute table and remove reference trees whose height is below a threshold
            join_df = Dbf5(os.path.join(arcgis_temp_dir, 'ref_trees_segments_join_classif.shp').replace('shp', 'dbf')).to_dataframe()

            drop_bool = join_df[PARAMS['ht_filter_colname']] < PARAMS['tree_ht_thresh']  ## to be dropped a tree has be under a certain threshold in filtering height retieved based on the raw CHM
            join_df = join_df.drop(join_df[drop_bool].index)

            ## Initialize classif_df_tile df to contain data for this tile
            unique_seg_IDs = np.unique(join_df["ID"])
            classif_df_tile = pd.DataFrame({'seg_ID': unique_seg_IDs})
            classif_df_tile['TREE_ID'] = classif_df_tile['CONIFEROUS'] = ''

            ## Loop over all unique segment IDs
            for id in unique_seg_IDs:
                candidates = join_df.loc[join_df['ID'] == id]
                distance_XY = np.linalg.norm(candidates[['X_coord', 'Y_coord']].values - candidates[['POINT_X', 'POINT_Y']].values, axis=1)
                classif_df_tile.loc[classif_df_tile['seg_ID'] == id, ['TREE_ID']] = candidates['TREE_ID'].iloc[np.argmin(distance_XY)]
                classif_df_tile.loc[classif_df_tile['seg_ID'] == id, ['CONIFEROUS']] = candidates['CONIFEROUS'].iloc[np.argmin(distance_XY)] == 'Y'

            ## Merge lidar metrics df from CSV with matched trees df (containing the class of each reference tree)
            classif_df = classif_df.append(pd.merge(lidar_metric_segments_df, classif_df_tile, left_on='ID', right_on ='seg_ID', how='inner'))

        ## Computing relative centiles of height by dividing every column by the 99th centile
        classif_df = classif_df.reset_index(drop=True)
        classif_df = classif_df.convert_objects(convert_numeric=True)   ## convert columns with dtype 'object' to 'float64'
        df_rel = classif_df[PARAMS['to_normalize']].div(classif_df['p99'], axis=0)
        df_rel.columns = ['norm_'+str for str in PARAMS['to_normalize']]
        classif_df = pd.concat([classif_df.reset_index(drop=True), df_rel], axis=1)

        ## Drop duplicate Tree IDs
        classif_df = classif_df.drop_duplicates(['TREE_ID'])

        ## Save dataset and indices of valid polygins (positive area) to a pickle file
        classif_df.to_pickle(os.path.join(PARAMS['exp_dir'], 'classif_df.pkl'))
        with open(os.path.join(PARAMS['exp_dir'], 'polyg_ok_bool.pkl'), 'wb') as f:
            cPickle.dump(polyg_ok_bool, f)

    else:

        ## Load dataset and indices
        classif_df = pd.read_pickle(os.path.join(PARAMS['exp_dir'], 'classif_df.pkl'))
        with open(os.path.join(PARAMS['exp_dir'], 'polyg_ok_bool.pkl'), 'rb') as f:
            polyg_ok_bool = cPickle.load(f)


    print('Training & testing RF')

    ## Build training and test sets
    np.random.seed(2013)
    classif_df['is_train'] = np.random.uniform(0, 1, len(classif_df)) <= PARAMS['train_pct']
    train, test = classif_df[classif_df['is_train'] == True].dropna(), classif_df[classif_df['is_train'] == False].dropna()
    labels_name = 'CONIFEROUS'
    rf = RandomForestClassifier(n_estimators=PARAMS['ntrees'], random_state=2013, n_jobs=PARAMS['nr_cores'])
    x_trn = train[PARAMS['feature_names']]
    y_trn = train[labels_name]
    x_tst = test[PARAMS['feature_names']]
    y_tst = test[labels_name]

    if PARAMS['train_rf_model']:
        ## Train RF and save model
        rf.fit(X=x_trn, y=y_trn)
        with open(PARAMS['rf_model_path'], 'wb') as f:
            cPickle.dump(rf, f)
    else:
        with open(PARAMS['rf_model_path'], 'rb') as f:
            rf = cPickle.load(f)

    ## Predict on test set
    y_tst_pred = rf.predict(x_tst)

    ## Assess RF and save results
    RES = {}
    RES['conf_mat'] = pd.crosstab(y_tst, y_tst_pred, rownames=['actual'], colnames=['predicted'])
    RES['OA'] = accuracy_score(y_tst, y_tst_pred)
    RES['Kappa'] = cohen_kappa_score(y_tst, y_tst_pred)
    RES['class_measures'] = classification_report(y_tst, y_tst_pred)

    print('Classification results:\n\n '
          'Confusion matrix:\n %s \n\n '
          'OA=%.3f, Kappa=%.3f \n\n '
          'Class-specific measures:\n %s'
          % (RES['conf_mat'], RES['OA'], RES['Kappa'], RES['class_measures']))

    ## Save RES dictionary to a json file
    RES['conf_mat'] = RES['conf_mat'].to_json()
    res_filename = 'RES_CLASSIFY_%s_%s.json' % (PARAMS['dataset_name'], PARAMS['experiment_name'])
    with open(os.path.join(PARAMS['exp_dir'], res_filename), 'w') as fp:
        json.dump(RES, fp)

    test = test.reset_index(drop=True)
    test['Pred_Conif'] = pd.Series(y_tst_pred.astype(bool))

    if PARAMS['predict_on_tiles']:

        ## Loop over the tiles
        for i_tile, _ in enumerate(segments_paths):

            ## Get tile name
            filename = segments_paths[i_tile].split('\\')[-1]
            tile_name = filename.split('_')[0] + '_' + filename.split('_')[1]

            print('Predicting on tile %s (%d/%d)' % (tile_name, i_tile + 1, len(segments_paths)))

            result = arcpy.GetCount_management(segments_paths[i_tile])
            count = int(result.getOutput(0))
            if count == 0:
                continue

            csv_path = os.path.join(tile_lidar_metrics_dir, tile_name + '_lidar_metrics.csv')

            ## Read back files and keep only polygons with positive area
            lidar_metric_df = pd.read_csv(csv_path)
            lidar_metric_df = lidar_metric_df[polyg_ok_bool[i_tile]]

            lidar_metric_df['Crown_ID'] = ['tile_%s_seg_%d' % (tile_name, i) for i in range(lidar_metric_df.shape[0])]

            segments_classified_path = os.path.join(tile_classified_dir, tile_name + '_segments_classif.shp')
            segments_df = Dbf5(segments_classified_path.replace('shp', 'dbf')).to_dataframe()
            if 'Pred_Conif' in segments_df.columns:
                segments_df = segments_df.drop('Pred_Conif', axis=1)

            ## Normalize variables by the 99th centile
            try:
                df_norm_tile = lidar_metric_df[PARAMS['to_normalize']].div(lidar_metric_df['p99'], axis=0)
            except:
                to_norm_ok = lidar_metric_df[PARAMS['to_normalize']].convert_objects(convert_numeric=True)
                denom_ok = lidar_metric_df['p99'].convert_objects(convert_numeric=True)
                df_norm_tile = to_norm_ok.div(denom_ok, axis=0)

            df_norm_tile.columns = ['norm_' + str for str in PARAMS['to_normalize']]
            lidar_metric_df = pd.concat([lidar_metric_df, df_norm_tile], axis=1)

            x_tile = lidar_metric_df[PARAMS['feature_names']].convert_objects(convert_numeric=True)
            ok_idx = pd.notnull(x_tile).any(1) & ~np.isnan(x_tile).any(1)

            lidar_metric_df['Pred_Conif'] = ''
            lidar_metric_df.loc[ok_idx, 'Pred_Conif'] = rf.predict(x_tile.loc[ok_idx])
            lidar_metric_df['Pred_Conif'] = np.where(lidar_metric_df['Pred_Conif'], 'Y', 'N')
            segments_predicted_df = pd.merge(segments_df, lidar_metric_df, on='Crown_ID', how='outer')

            ## Add empty fields to attribute table
            fields = ['Pred_Conif']
            for field in fields:
                if field in [f.name for f in arcpy.ListFields(segments_classified_path)]:
                    arcpy.DeleteField_management(segments_classified_path, field)
                arcpy.AddField_management(segments_classified_path, field, "TEXT", field_length=1)

            ## Update the Pred_Conif new fields of attribute table
            if segments_predicted_df.shape[0] > 0:
                with arcpy.da.UpdateCursor(segments_classified_path, fields) as cursor:  ## Create update cursor for feature class
                    for irow, row in enumerate(cursor):
                        for icol, field in enumerate(fields):
                            row[icol] = segments_predicted_df[field][irow]

                        try:
                            cursor.updateRow(row)
                        except:
                            cursor.updateRow('A')

    if PARAMS['load_in_MXD']:

        print("Loading layers into MXD")

        ## Get MXD and dataframe
        mxd = arcpy.mapping.MapDocument(PARAMS['output_mxd'])
        df = arcpy.mapping.ListDataFrames(mxd)[0]  ## assuming there is only 1 df (called Layer, the default)
        mxd.activeView = df.name
        mxd.title = df.name

        layer_paths = glob.glob(os.path.join(tile_classified_dir, '*.shp'))   ## we do not plot VegMaskRaw anymore

        source_lyr_path = os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'conif_decid.lyr')
        source_lyr = arcpy.mapping.Layer(source_lyr_path)

        for i_layer, layer_path in enumerate(layer_paths):

            layer_name = layer_path.split('\\')[-1]

            print('adding layer %s (%d/%d) ...' % (layer_name, i_layer+1, len(layer_paths)))

            layer_to_add = arcpy.mapping.Layer(layer_path)
            arcpy.mapping.AddLayer(df, layer_to_add, "TOP")

            ## Applying symbology
            layer_to_update = arcpy.mapping.ListLayers(mxd, layer_to_add.name, df)[0]  ## redefine layer_to_update as an arcpy object
            arcpy.ApplySymbologyFromLayer_management(layer_to_update, source_lyr)  ## first apply the symbology with ApplySymbologyFromLayer_management (needs a path to the layer providing the symbology)
            arcpy.mapping.UpdateLayer(df, layer_to_update, source_lyr, True)  ## update symbology by using the one of source_lyr (has to be an arpy object and not just the path)

        group_arcmap_layers(mxd, df, os.path.join(PARAMS['base_dir'], 'mxds', 'lyrs', r'NewGroupLayer.lyr'),
                            {'TreeCrownClassif': ['segments_classif', 'TOP']})

        ## Saving final MXD in same file
        mxd.save()


    print('Total ' + toc(start_time))



