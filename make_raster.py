import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal,ogr,osr
import rasterio
import shapely

class Raster:
    def __init__(self):
        self.pixel_size = 0.00000325872 #0.000325872
        self.src_layer = None
        self.xmin = None
        self.xmax = None
        self.x_res = None
        self.ymin = None
        self.ymax = None
        self.y_res = None

    def rasterize_shapefile(self,input_shapefile_path,output_raster_path):

        shp = ogr.Open(input_shapefile_path) #Get_shapefile_attributes():r'RiodeJaneiro_shp\RJ_shape\buildings.shp'
        self.src_layer = shp.GetLayer()

        
        self.xmin, self.xmax, self.ymin, self.ymax = self.src_layer.GetExtent() # Boundaries

        self.x_res = int(round((self.xmax-self.xmin)/self.pixel_size)) #Resolução X
        self.y_res = int(round((self.ymax-self.ymin)/self.pixel_size)) #Resolução Y

        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster_path,self.x_res,self.y_res,1,gdal.GDT_Float32,['COMPRESS=LZW']) # 1 = numbandas 
        target_ds.SetGeoTransform((self.xmin,self.pixel_size,0.0,self.ymax,0.0,-self.pixel_size))
        srse = osr.SpatialReference()
        projection = 'EPSG:4326'
        srse.SetWellKnownGeogCS(projection)
        target_ds.SetProjection(srse.ExportToWkt()) 
        band = target_ds.GetRasterBand(1)
        target_ds.GetRasterBand(1).SetNoDataValue(0) # era -9999
        band.Fill(0) #-9999
        gdal.RasterizeLayer(target_ds,[1],self.src_layer,None,None,[1],options = ['ALL_TOUCHED=TRUE','ATRIBUTE=osm_id'])
        target_ds = None
    
    def open_rasterfile(self,output_raster_path):     
        with rasterio.open(output_raster_path) as raster_file: # Abrindo o raster -> Open_Raster():
            raster = raster_file.read()
            raster = raster.squeeze()
        return raster