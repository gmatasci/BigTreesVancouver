"""
Project Name: BigTreesVan
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: PREPROCESS_BigTrees_Vancouver.py
Objective: Merge shps (City of Vancouver and Ira Sutherland's datasets), assign a Tree_ID to the trees missing it,
remove duplicate Tree_IDs creating unique values, attach tentative height values from CHM and fill reference_trees
attribute table with field collected data.
"""


## IMPORT MODULES ---------------------------------------------------------------

from __future__ import division  # to allow floating point divisions
import os
import glob
import sys
import arcpy
import numpy as np
import pandas as pd
from simpledbf import Dbf5
from collections import Counter

sys.path.append(r'D:\Research\MyPythonModules')
sys.path.append(r'D:\Research\ANALYSES\NationalMappingForestAttributes\WKG_DIR\code')
from mybasemodule import*

if __name__ == '__main__':


## PARAMETERS ----------------------------------------------------------------

    PARAMS = {}

    PARAMS['merge_clean_data'] = True   ## main switch to be set to False not to rerun the merge of initial shapefiles and duplicate ID removal (risk of messing up tree IDs used in the field)
    PARAMS['attach_field_data'] = False    ## switch to join the xls table with field data measurements

    PARAMS['measures_fieldnames'] = ['avg_ht', 'cr_diam', 'ht_liv_cr', 'NOTES']

    PARAMS['ref_trees_path'] = r'E:\BigTreesVan_data\GroundTruth\reference_trees.shp'

    PARAMS['field_data_path'] = r'E:\BigTreesVan_data\GroundTruth\VanBigTrees_FieldTrip\VanBigTrees_FieldTrip.xlsx'

    PARAMS['base_dir'] = r'D:\Research\ANALYSES\BigTreesVan'

    PARAMS['dataset_name'] = 'Vancouver_500m_tiles'
    PARAMS['experiment_name'] = 'step_0p3m_mindist_8'

    PARAMS['wkg_dir'] = os.path.join(PARAMS['base_dir'], 'wkg')
    PARAMS['exp_dir'] = os.path.join(PARAMS['wkg_dir'], PARAMS['dataset_name'], PARAMS['experiment_name'])

    PARAMS['Ira_tree_ID_start'] = 26681   ## set as the number after the largest Tree ID in City of Vancouver park database

    PARAMS['ht_thresh_to_sample'] = 30    ## height threshold above which to sample trees
    PARAMS['parks_to_sample'] = ['Queen Elizabeth Park', 'Musqueam Park', 'Locarno Park', 'Memorial West Park']  ## set as the number after the largest Tree ID in City of Vancouver park database

## START ---------------------------------------------------------------------

    print('preprocess_GT_merge_shps.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))
    start_time = tic()

    arcpy.CheckOutExtension("3D")
    arcpy.CheckOutExtension("spatial")
    arcpy.env.overwriteOutput = True

    if PARAMS['merge_clean_data']:

        ## Merging GT shps
        print('Merging shps')

        park_trees_path = r'E:\BigTreesVan_data\GroundTruth\Tree_Park.shp'
        arcpy.AddXY_management(park_trees_path)
        stanley_park_big_tree_path = r'E:\BigTreesVan_data\GroundTruth\Stanley_Park_big_tree_data_Sutherland_I.shp'
        stanley_park_big_tree_OKheight_path = stanley_park_big_tree_path.replace('Stanley_Park_big_tree_data_Sutherland_I', 'Stanley_Park_big_tree_data_Sutherland_I_OKheight')
        arcpy.Copy_management(stanley_park_big_tree_path, stanley_park_big_tree_OKheight_path)
        with arcpy.da.UpdateCursor(stanley_park_big_tree_OKheight_path, "avg_ht") as cursor:
            for row in cursor:
                if row[0] < 30:
                    cursor.deleteRow()
        arcpy.AddXY_management(stanley_park_big_tree_OKheight_path)
        kerrisdale_big_trees_path = r'E:\BigTreesVan_data\GroundTruth\Kerrisdale_big_trees.shp'
        arcpy.AddXY_management(kerrisdale_big_trees_path)
        arcpy.Merge_management([park_trees_path, stanley_park_big_tree_OKheight_path, kerrisdale_big_trees_path], PARAMS['ref_trees_path'])

        print('Making Tree IDs unique')

        ## Read Tree_ID column to replace values of duplicates
        tree_ID_df = pd.DataFrame.from_records(arcpy.da.FeatureClassToNumPyArray(PARAMS['ref_trees_path'], ('TREE_ID')))
        tree_ID_df['is_duplicated'] = tree_ID_df.duplicated(['TREE_ID'])    ## assign True to each duplicated entry, including those with values of 0
        nr_dupl = sum(tree_ID_df['is_duplicated'])
        tree_ID_df.loc[tree_ID_df['is_duplicated'], ['TREE_ID']] = np.asarray(np.arange(nr_dupl).reshape((nr_dupl, 1)) + PARAMS['Ira_tree_ID_start'])

        ## Update Tree_ID column in attribute table
        if tree_ID_df.shape[0] == len(np.unique(tree_ID_df['TREE_ID'])):
            fields = ['TREE_ID']
            with arcpy.da.UpdateCursor(PARAMS['ref_trees_path'], fields) as cursor:  ## Create update cursor for feature class
                for irow, row in enumerate(cursor):
                    for icol, field in enumerate(fields):
                        row[icol] = tree_ID_df[field][irow]
                    cursor.updateRow(row)
        else:
            sys.exit("No unique Tree_IDs in tree_ID_df")


        print('Extracting approximate height from rasters')

        ## List file paths of all tiles
        tile_chm_treetops_dir = os.path.join(PARAMS['exp_dir'], 'tiles_chm_treetops')
        file_key = os.path.join(tile_chm_treetops_dir, '*_chm_raw.asc')
        chm_paths = glob.glob(file_key)

        temp_shp = PARAMS['ref_trees_path'].replace('reference_trees', 'temp_reference_trees')
        arcpy.Copy_management(PARAMS['ref_trees_path'], temp_shp)
        arcpy.sa.ExtractMultiValuesToPoints(temp_shp, chm_paths)     ## in the attribute table each tile whose path is in chm_paths will be a column

        print('Selecting samples to be measured')

        ref_trees_df = Dbf5(temp_shp.replace('shp', 'dbf')).to_dataframe()

        colnames = ref_trees_df.columns
        colnames_for_max = [colname for colname in colnames if ('48' in colname.split('_')[0]) | ('49' in colname.split('_')[0])]

        ## Get approximate height as maximum across columns (one per tile)
        ref_trees_df['appr_ht'] = ref_trees_df[colnames_for_max].max(axis=1)
        ref_trees_df = ref_trees_df.drop(colnames_for_max, 1)

        bool_idx = (ref_trees_df['PARK_NAME'].isin(PARAMS['parks_to_sample'])) & (ref_trees_df['appr_ht'] >= PARAMS['ht_thresh_to_sample'])

        ## Subsample candidate trees in parks with many tall trees such as Queen Elizabeth Park or Musqueam Park
        bool_subs = np.random.choice([True, False], ref_trees_df.shape[0], p=[0.5, 0.5])
        ref_trees_df['to_samp'] = np.where( (bool_idx & ref_trees_df['PARK_NAME'].isin(['Locarno Park', 'Memorial West Park']) |
                                            (bool_idx & bool_subs& ref_trees_df['PARK_NAME'].isin(['Queen Elizabeth Park', 'Musqueam Park']))
                                          ), 'T', 'F')

        park_names_high_trees = ref_trees_df.loc[ref_trees_df['to_samp']=='T']['PARK_NAME']
        freq_parks = Counter(park_names_high_trees)

        print('Trees to sample by park:')
        for key, value in freq_parks.items():
            print(key, value)

        print('Updating attribute table')

        ## Add empty fields to attribute table
        existing_fields = arcpy.ListFields(PARAMS['ref_trees_path'])
        field_names = [field.name for field in existing_fields]
        if 'appr_ht' not in field_names:
            arcpy.AddField_management(PARAMS['ref_trees_path'], 'appr_ht', "FLOAT", field_precision=5, field_scale=2)
        if 'to_samp' not in field_names:
            arcpy.AddField_management(PARAMS['ref_trees_path'], 'to_samp', "TEXT", field_length=1)  ## arcgis does not have field type boolean

        ## Update attribute table based on common TREE_ID
        fields = ['appr_ht', 'to_samp', 'TREE_ID']
        with arcpy.da.UpdateCursor(PARAMS['ref_trees_path'], fields) as cursor:   ## create update cursor for feature class
            for irow, row in enumerate(cursor):
                for icol, field in enumerate(fields[:-1]):
                    row[icol] = ref_trees_df.loc[ref_trees_df['TREE_ID'] == row[-1]][field].values[0]    ## .values[0] means getting the first (and only) element of the series of values
                cursor.updateRow(row)

    if PARAMS['attach_field_data']:

        field_data = pd.read_excel(PARAMS['field_data_path'])
        field_data = field_data.loc[~field_data[PARAMS['measures_fieldnames']].isnull().any(axis=1)]
        field_data = field_data.replace('na', 0)

        for field in PARAMS['measures_fieldnames']:
            if field not in [f.name for f in arcpy.ListFields(PARAMS['ref_trees_path'])]:
                if field == 'NOTES':
                    arcpy.AddField_management(PARAMS['ref_trees_path'], 'NOTES', "TEXT", field_length=150)
                else:
                    arcpy.AddField_management(PARAMS['ref_trees_path'], field, "FLOAT", field_precision=5, field_scale=2)

        fieldnames = PARAMS['measures_fieldnames']
        fieldnames.append('TREE_ID')
        with arcpy.da.UpdateCursor(PARAMS['ref_trees_path'], fieldnames) as cursor:  ## Create update cursor for feature class
            for irow, row in enumerate(cursor):
                if (row[-1] == field_data['TREE_ID']).any():
                    for icol, field in enumerate(fieldnames[:-1]):
                        if field == 'NOTES':
                            row[icol] = field_data.loc[field_data['TREE_ID'] == row[-1]][field].item()
                        else:
                            row[icol] = np.asscalar(field_data.loc[field_data['TREE_ID'] == row[-1]][[field]].convert_objects(convert_numeric=True).values)
                    cursor.updateRow(row)

    print('Total ' + toc(start_time))


