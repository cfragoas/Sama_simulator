import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal,ogr,osr
import rasterio
from make_grid import Grid
import pickle

class Raster(Grid):
    def __init__(self,input_shapefile,output_raster,projection,burner_value,pixel_size,no_data_value,raster_cell_value):
        super().__init__()
        self.pixel_size = pixel_size # Fator de escala (graus por pixel)
        self.temp_raster_array_path = 'rasters/temp/temp_raster_array.npy' # NÃO MEXER!
        self.src_layer = None
        self.xmin = None
        self.xmax = None
        self.x_res = None # Número de pixels na direção x
        self.x_scale = None # Fator de escala no eixo x em metros por pixel
        self.ymin = None
        self.ymax = None
        self.y_res = None # Número de pixels na direção y
        self.y_scale = None # Fator de escala no eixo y em metros por pixel
        self.raster = None
        self.no_data_value = no_data_value
        self.input_shapefile_path = input_shapefile
        self.output_raster_path = output_raster
        self.projection = projection # Proejeção do shapefile
        self.burner_value = burner_value # Atributo usado para rasterizar o shapefile. Ele irá representar a escala da layer
        self.raster_cell_value = raster_cell_value # Define o tipo se a célula será preenchida com int ou float
    
    def rasterize_shapefile(self):

        if os.path.isfile(self.input_shapefile_path): #Conferindo se o shapefile selecionado existe
            shp = ogr.Open(self.input_shapefile_path) #Get_shapefile_attributes():r'RiodeJaneiro_shp\RJ_shape\buildings.shp'
        else:
            raise Exception('Shapefile selecionado nao encontrado. Verifique novamente!')
        self.src_layer = shp.GetLayer()

        
        self.xmin, self.xmax, self.ymin, self.ymax = self.src_layer.GetExtent() # Boundaries

        self.x_res = int(round((self.xmax-self.xmin)/self.pixel_size)) #Resolução X
        self.y_res = int(round((self.ymax-self.ymin)/self.pixel_size)) #Resolução Y

        if os.path.isfile(self.output_raster_path): # Conferindo se o raster que será criado já existe 
            print('Raster desejado ja existe com esse nome! Pulando a rasterizacao ...')
            pass
        
        else:
            if self.raster_cell_value == 'int32':
                bit_type = gdal.GDT_Int32
            elif self.raster_cell_value == 'float32':
                bit_type = gdal.GDT_Float32
            elif self.raster_cell_value == 'int16':
                bit_type = gdal.GDT_Int16
            elif self.raster_cell_value == 'float64':
                bit_type = gdal.GDT_Float64
            elif self.raster_cell_value == 'int64':
                bit_type = gdal.GDT_Int64
            else:
                raise Exception('Valor de celula nao definido! Por favor defina se sera usado int ou float no arquivo yaml!')
            #print('Rasterizando o shapefile ...')
            target_ds = gdal.GetDriverByName('GTiff').Create(self.output_raster_path,self.x_res,self.y_res,1,bit_type,['COMPRESS=LZW']) # 1 = numbandas 
            target_ds.SetGeoTransform((self.xmin,self.pixel_size,0.0,self.ymax,0.0,-self.pixel_size))
            srse = osr.SpatialReference()
            projection = self.projection
            srse.SetWellKnownGeogCS(projection)
            target_ds.SetProjection(srse.ExportToWkt()) 
            band = target_ds.GetRasterBand(1)
            target_ds.GetRasterBand(1).SetNoDataValue(self.no_data_value) # era -9999
            band.Fill(0) #-9999
            gdal.RasterizeLayer(target_ds,[1],self.src_layer,None,None,[1],options = ['ALL_TOUCHED=TRUE','ATTRIBUTE='+self.burner_value])
            target_ds = None
            self.src_layer = None # Need to be set to nome, else cannot dump swigpy object with pickle!
            #print('... feito!')


    def make_grid(self):     
        with rasterio.open(self.output_raster_path) as raster_file: # Abrindo o raster -> Open_Raster():
            self.raster = raster_file.read()
            self.raster = self.raster.squeeze()
            super().make_grid(self.raster.shape[0],self.raster.shape[1])
        
    def make_points(self,dist_type, samples, n_centers, random_centers=True, plot=False):
        super().make_points(dist_type, samples, n_centers, random_centers, plot)
       # self.point_condition = self.raster[self.grid != 0]

    #def save_raster(self,raster):
    #    np.save(self.temp_raster_array_path,raster)
    #
    #def load_raster(self):
    #    loaded_raster = np.load(self.temp_raster_array_path)
    #    return loaded_raster
    #
    #def delete_raster_npy_file(self):
    #    os.remove(self.temp_raster_array_path)
#
    #def set_point_condition(self,point_condition):
    #    self.point_condition = point_condition
#
    #def set_raster_transform_paths(self,input_shapefile,output_raster):
    #    self.input_shapefile_path = input_shapefile
    #    self.output_raster_path = output_raster

    def delete_tif_file(self): # Used to remove the tif file.
        os.remove(self.output_raster_path)

    def define_scaling_factor(self):
    
        centroid_lat = (self.ymax+self.ymin)/2 # Valor central
    
        earth_radius = 6378137  # Raio da terra em metros
    
    
        meters_per_degree_lat = (2 * np.pi * earth_radius) / 360 # Fator de escala metros por grau latitude
    
    
        meters_per_degree_long = meters_per_degree_lat * np.cos(np.radians(centroid_lat)) # Fator de escala metros por grau longitude

        # Calcula o fator de escala, resultado em metros por pixel 
        self.y_scale = meters_per_degree_lat * self.pixel_size
        self.x_scale = meters_per_degree_long * self.pixel_size