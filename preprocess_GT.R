## Project Name: BigTreesVan
## Authors: Giona Matasci (giona.matasci@gmail.com)
## File Name: preprocess_GT.R
## Objective: build shapefiles of tree measurements data starting from CSVs with UTM zone 10 coordinates

#### TO DO -------------------------------------------------------------------

## STILL TO DO:
# -

## SOLVED:


#### INIT --------------------------------------------------------------------

print('Preprocess GT data')

rm(list=ls()) # clear all variables


params <- list()

params$GT.dir <- "E:/BigTreesVan_data/GroundTruth"

params$GT.csv.filenames <- c("Stanley Park big tree data_Sutherland_I.csv", "Kerrisdale big trees.csv") 
params$GT.shp.filenames <- c("Stanley_Park_big_tree_data_Sutherland_I", "Kerrisdale_big_trees") 

params$height.col.name <- "avg_ht"  ## column with average height values (already there for Kerrisdale, added manually in Excel for Stanley Park)

#### LOAD PACKAGES ----------------------------------------------------------

list.of.packages <- c("rgdal",
                      "raster",
                      "sp",
                      "spdep",
                      "spatstat",
                      "rgeos",
                      "maptools", 
                      "plyr",
                      "dplyr",
                      "ggplot2",
                      "data.table",
                      "lubridate", 
                      "doParallel", 
                      "foreach"
)
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]   # named vector members whose name is "Package"
if(length(new.packages)) install.packages(new.packages)
for (pack in list.of.packages){
  library(pack, character.only=TRUE)
}


#### START ------------------------------------------------------------------

shp.reference <-  readOGR(dsn=params$GT.dir, layer = "Tree_Park") 
CRS.reference <- crs(shp.reference)

for (f in 1:length(params$GT.csv.filenames)) {

  
  GT.csv.filename <- params$GT.csv.filenames[f]
  
  GT.dt <- fread(file.path(params$GT.dir, GT.csv.filename, fsep=.Platform$file.sep))

  setnames(GT.dt, c("UTM eastings", "UTM northings"), c("UTM_eastings", "UTM_northings"))
  GT.dt.OK <- GT.dt[!is.na(GT.dt[["UTM_eastings"]])]

  GT.points <- SpatialPointsDataFrame(GT.dt.OK[, c("UTM_eastings", "UTM_northings")],  ## coordinates
                                      GT.dt.OK,    ## R object to convert
                                      proj4string=CRS.reference)

  writeOGR(GT.points , params$GT.dir, params$GT.shp.filenames[f], driver="ESRI Shapefile", overwrite_layer=TRUE)  ## save it for future checks

}

