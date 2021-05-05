# BigTreesVancouver
Python code for a project carried out at UBC's Integrated Remote Sensing Studio for the City of Vancouver to map and characterize the entirety of the city's urban canopy using LiDAR data. Focus was on the detection of large trees that are of significance for local neighborhoods.

## Goal
Develop point cloud data processing routines with LAStools, object-based image analysis algorithms (region growing) and machine learning models (Random Forests) in Python for individual tree detection, crown segmentation and classification (coniferous/deciduous).

## Approach and main results
<table style="padding:2px">
  <tr>
    <td> Canopy Height Model (CHM) </td>
    <td> Treetop detection </td>
  </tr>
  <tr>
    <td> <img src="https://github.com/gmatasci/BigTreesVancouver/blob/master/Figures/Sequence2_CHM.png" align="right" alt="2" width = 400px></td>
    <td> <img src="https://github.com/gmatasci/BigTreesVancouver/blob/master/Figures/Sequence3_treetops.png" align="right" alt="3" width = 400px></td>
  </tr>
    <tr>
    <td> Crown segmentation </td>
    <td> Coniferous vs. deciduous classification </td>
  </tr>
   <tr>
    <td> <img src="https://github.com/gmatasci/BigTreesVancouver/blob/master/Figures/Sequence4_crowns.png" align="right" alt="4" width = 400px></td>
    <td> <img src="https://github.com/gmatasci/BigTreesVancouver/blob/master/Figures/Sequence5_classification.png" align="right" alt="5" width = 400px></td>
  </tr>
</table>
