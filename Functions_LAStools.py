## Add LAStools bin directory to the system PATH environment variable by:
# "Edit the system environment variables" --> "Environment variables"
# --> select variable "PATH" in "User variables for gmatasci" --> "Edit..."
# --> paste "C:\LAStools\bin" after a ";" without any space

import arcpy
import numpy as np
import os

## Print infos about las/laz file
def lasinfo(in_dir, file_name='', in_ext='laz', verbose=False):
    if verbose:
        print('lasinfo: getting infos on file(s)')
    cmd = 'lasinfo %s\*%s.%s' \
          % (in_dir, file_name, in_ext)
    os.system(cmd)

## Zip las files to laz
def laszip(in_dir, out_dir, in_ext='laz', verbose=False):
    if verbose:
        print('laszip: compressing las files to laz (lossless)')
    cmd = 'laszip %s\*.%s ' \
          '-odir %s' \
          % (in_dir, in_ext, out_dir)
    os.system(cmd)

## To access to the points from neighbouring tiles.
def lasindex(in_dir, nr_cores, in_ext='laz', verbose=False):
    if verbose:
        print('lasindex: indexing existing tiles')
    cmd = 'lasindex -i %s\*.%s -cores %d' \
           % (in_dir, in_ext, nr_cores)
    os.system(cmd)

## Create a buffered tiling from the original flight strips
def lastile(in_dir, out_dir, tile_size, tile_buffer, nr_cores, in_ext='laz', out_ext='laz', verbose=False):

    if out_ext == 'laz':
        olaz_cmd = '-olaz'
    else:
        olaz_cmd = ''

    if tile_buffer == 'remove':
        if verbose:
            print('lastile: removing buffer from tiles')
        tile_cmd = '-remove_buffer'
        odix_cmd = '_unbuffered'
    else:
        if verbose:
            print('lastile: tiling')
        tile_cmd = '-tile_size %d -buffer %d ' % (tile_size, tile_buffer)
        odix_cmd = '_tile'

    cmd = 'lastile -i %s\*.%s ' \
          '%s %s -odir %s -odix %s -cores %d' \
          % (in_dir, in_ext, tile_cmd, olaz_cmd, out_dir, odix_cmd, nr_cores)
    os.system(cmd)

## Remove isolated points
def lasnoise(in_dir, out_dir, nr_cores, cell_size=4, isolated=5, in_ext='laz', verbose=False):
    if verbose:
        print('lasnoise: denoising')
    cmd = 'lasnoise -i %s\*.%s ' \
          ' -remove_noise -step %.2f -isolated %d ' \
          '-olaz -odir %s -odix _denoised -cores %d' \
          % (in_dir, in_ext, cell_size, isolated, out_dir, nr_cores)
    os.system(cmd)

## Find the bare-earth points in all tiles
def lasground_new(in_dir, out_dir, landscape, nr_cores, in_ext='laz', buffer_mode='NONE', tile_buffer=0, spike_down=0, verbose=False):
    if verbose:
        print('lasground_new: finding ground points')
    if buffer_mode == 'OTF':
        buffer_cmd = '-buffered %d -remain_buffered' % (tile_buffer)
    else:
        buffer_cmd = ''
    cmd = 'lasground_new -i %s\*.%s ' \
          '%s %s -extra_fine -spike_down %d ' \
          '-olaz -odir %s -odix _ground -cores %d' \
          % (in_dir, in_ext, buffer_cmd, landscape, spike_down, out_dir, nr_cores)
    os.system(cmd)

## Remove low and high outliers that are often just noise (e.g. clouds or birds)
## By default lasheight uses the points classified as ground to construct a TIN and then calculates
## the height of all other points in respect to this ground surface TIN
def lasheight(in_dir, out_dir, nr_cores, in_ext='laz', above=9999, below=-9999, buffer_mode='NONE', replace_z=False, verbose=False):
    if verbose:
        print('lasheight: filtering and computing height')

    if buffer_mode == 'OTF':
        buffer_cmd = '-remain_buffered'
    else:
        buffer_cmd = ''

    odix = ''
    if replace_z:
        replace_z_cmd = '-replace_z'
        odix = odix+'_ht_norm'
    else:
        replace_z_cmd = ''

    if above == 9999 and below == -9999:
        above_below_cmd = ''
    else:
        odix = odix + '_ht_filtered'
        above_below_cmd = '-drop_above % d -drop_below % d' % (above, below)

    cmd = 'lasheight -i %s\*.%s ' \
          '%s %s %s ' \
          '-olaz -odir %s -odix %s -cores %d' \
          % (in_dir, in_ext, buffer_cmd, replace_z_cmd, above_below_cmd, out_dir, odix, nr_cores)
    os.system(cmd)

## Identify buildings and trees in all denoised tiles
def lasclassify(in_dir, out_dir, nr_cores, in_ext='laz', buffer_mode='NONE', verbose=False):
    if verbose:
        print('lasclassify: classifying buildings and trees')
    if buffer_mode == 'OTF':
        buffer_cmd = '-remain_buffered'
    else:
        buffer_cmd = ''
    cmd = 'lasclassify -i %s\*.%s ' \
          '%s -olaz -odir %s -odix _classif -cores %d' \
          % (in_dir, in_ext, buffer_cmd, out_dir, nr_cores)
    os.system(cmd)

##
def las2dem(in_dir, DEM_type, step, out_dir, nr_cores, in_ext='laz', freeze_dist=0, hillshaded=False, buffer_mode='NONE', verbose=False):

    if hillshaded:
        file_name = '_hshad_'+DEM_type
        hillshade_cmd='-hillshade'
        message = 'Hillshaded '+DEM_type
    else:
        file_name = '_'+DEM_type
        hillshade_cmd = ''
        message = DEM_type

    if buffer_mode == 'OTF':   ## On-the-fly buffer
        buffer_cmd = '-use_orig_bb'
    elif buffer_mode == 'LASTILE':   ## LAStile buffer
        buffer_cmd = '-use_tile_bb'
    elif buffer_mode == 'NONE':
        buffer_cmd = ''

    if DEM_type == 'DTM':  ## uses points classified as ground only (LAS class 2)
        return_type = '-keep_class 2'
    elif DEM_type == 'DSM':   ## uses first returns only
        return_type = '-first_only'
    elif DEM_type == 'PF_DSM':     ## the spike free uses all returns implicitely
        return_type = '-spike_free %.2f' % freeze_dist
        file_name = (file_name+'_%.2f' % freeze_dist).replace('.', 'p')

    if verbose:
        print('las2dem: '+message)

    cmd = 'las2dem -i %s\*.%s ' \
          '%s -extra_pass -step %.2f ' \
          '%s %s -oasc -odir %s -odix %s -cores %d' \
          % (in_dir, in_ext, return_type, step, hillshade_cmd, buffer_cmd, out_dir, file_name, nr_cores)
    os.system(cmd)


##
def lasgrid(in_dir, out_dir, out_name, step, action, classes_to_keep='', subcircle=0, fill=0,
            key='*', nr_MB_mem=1000, in_ext='laz', out_ext='asc', buffer_mode='NONE', verbose=False):

    if action == 'mosaic':
        if verbose:
            print('lasgrid: mosaicking tiles')
        action_cmd = '-merged -highest -o %s\%s.%s' % (out_dir, out_name, out_ext)
    elif action == 'mask':
        if verbose:
            print('lasgrid: producing mask')
        action_cmd = '-occupancy -subcircle %.2f -fill %d -keep_class %s ' \
                     '-oasc -odir %s -odix %s' \
                     % (subcircle, fill, classes_to_keep, out_dir, out_name)
    else:
        action_cmd = ''

    if buffer_mode == 'OTF':
        buffer_cmd = '-use_orig_bb'
    elif buffer_mode == 'LASTILE':
        buffer_cmd = '-use_tile_bb'
    elif buffer_mode == 'NONE':
        buffer_cmd = ''

    cmd = 'lasgrid -i %s\%s.%s ' \
          '%s %s -step %.2f -mem %d'  \
          % (in_dir, key, in_ext, action_cmd, buffer_cmd, step, nr_MB_mem)
    os.system(cmd)




#
#
#
# def lasthin(in_dir, out_dir, sub_step, subcircle, nr_cores, key='*', in_ext='laz', verbose=False):
#     if verbose:
#         print('lasthin: thinning tiles')
#
#     cmd = 'lasthin -i %s\%s.%s ' \
#           '-highest -step %.2f -subcircle %.2f ' \
#           '-odir %s -odix _beam -olaz -cores %d' \
#           % (in_dir, key, in_ext, sub_step, subcircle, out_dir, nr_cores)
#     os.system(cmd)
#
#
#
#
#
# def pit_free_chm(dir_ht_norm, dir_beams, sub_step, beam_radius, nr_cores, key, in_ext, verbose=False):
#
#     lasthin(in_dir=dir_ht_norm, out_dir=dir_beams, sub_step=sub_step, beam_radius=beam_radius, nr_cores=nr_cores, key=key, in_ext=in_ext, verbose=False)
#
#     ## First, we thin the point cloud (height normalized points WITH buffers) by keeping only the highest return
#     ## (splatted into a circle to account for the laser beam diameter -subcircle) per grid cell of size -step
#     cmd = '%s/lasthin -i %s/*.laz ' \
#           '-highest -step %.3f -subcircle %.3f ' \
#           '-odir %s -odix _beam -olaz -cores %d' \
#           % (PARAMS['lastools_path'], tile_chm_path, sub_step, PARAMS['beam_radius'], temp_path, PARAMS['nr_cores'])
#     os.system(cmd)
#     #
#     # ## Then, we create a series of CHMs only using points above a set of thresholds PARAMS['pit_free_levels']
#     # ## (assigned to -drop_z_below with level) by considering only points within the tile WITHOUT buffer (-use_tile_bb)
#     # for level in PARAMS['pit_free_levels']:
#         cmd = '%s/las2dem -i %s/*.laz -use_tile_bb ' \
#               '-drop_z_below %d -step %.3f -kill %.3f ' \
#               '-odir %s -odix _%d -oasc -cores %d' \
#               % (PARAMS['lastools_path'], temp_path, level, PARAMS['step'], kill, temp_path, level, PARAMS['nr_cores'])
#         os.system(cmd)
#     #
#     # ## Then, we merge the partial CHMs together into a pit-free CHM by keeping the maximum value across all rasters
#     # os.chdir(temp_path)
#     # tiles = glob.glob('*.laz')
#     # tile_names = [s.strip('.laz') for s in tiles]
#     # for tile_name in tile_names:
#     #     cmd = '%s/lasgrid -i %s/%s_*.asc -merged ' \
#     #               '-step %.3f -highest ' \
#     #               '-o %s/%s_chm_pit_free.asc' \
#     #               % (PARAMS['lastools_path'], temp_path, tile_name, PARAMS['step'], tile_chm_path, tile_name)
#     #     os.system(cmd)
#
#

















