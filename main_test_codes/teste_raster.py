import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal,ogr,osr
import rasterio
import math
from random import randint, gauss, gammavariate
import sys
import matplotlib.pyplot as plt


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

def highestPowerOf2(n):
    return (np.log2(n & (~(n - 1))))


class Grid:
    def __init__(self):
        self.lines = None
        self.columns = None
        self.grid = None
        self.dist_mtx = None


        # variables calculated inside the class
        self.centers_set = None

    def make_grid(self, lines=None, columns=None):
        self.lines = lines
        self.columns = columns
        self.grid = np.zeros(shape=(lines, columns))

    def clear_grid(self):
        # self.grid = np.zeros(shape=(self.lines, self.columns))
        self.grid[:] = 0

    def make_points(self, dist_type, samples, n_centers, random_centers=True, plot=False):  # generating centers for the distributions
        centers_set = set()  # workaround to generate unique values
        if random_centers:
            while len(centers_set) < n_centers:
                x, y = 7, 0
                while (x, y) == (7, 0):
                    x, y = randint(0, self.lines - 1), randint(0, self.columns - 1)
                # that will make sure we don't add (7, 0) to cords_set
                centers_set.add((x, y))
            self.centers_set = list(centers_set)

        else:
            # Create a grid of points in x-y space
            if n_centers == 1:
                xvals = [np.round(self.columns / 2)]
                yvals = [np.round(self.lines / 2)]
            elif n_centers % 2 != 0:
                sys.exit('n_centers must be a even number!!!')
            else:
                n = highestPowerOf2(n_centers)  # number of lines
                if n % 2 != 0:
                    n -= 1
                if n == 0:
                    n = 2
                xvals = np.round(np.linspace((self.columns / (n_centers / n)) / 2,
                                             self.columns - (self.columns / (n_centers / n)) / 2, int(n_centers / n)))
                yvals = np.round(np.linspace(self.lines / (2 * n), self.lines * (1 - 1 / (2 * n)), int(n)))

            centers_set_ = np.row_stack([(x, y) for x in xvals for y in yvals]).astype(int)
            for i in range(len(centers_set_)):
                centers_set.add((centers_set_[i][0], centers_set_[i][1]))
            self.centers_set = list(centers_set)

            # # Apply linear transform
            # a = np.column_stack([[2, 1], [-1, 1]])
            # print(a)
            # uvgrid = np.dot(a, xygrid)

        if dist_type == 'gaussian':
            mu = 0
            sigma = min(self.lines, self.columns) / 10
            for i in range(n_centers):
                for _ in range(samples):
                    [x, y] = [self.centers_set[i][0] + round(gauss(mu, sigma)), self.centers_set[i][1] + round(gauss(mu, sigma))]
                    if (x < self.lines and y < self.columns) and (x >= 0 and y >= 0):
                        self.grid[x, y] += 1

        elif dist_type == 'gamma':
            alpha = min(self.columns, self.lines) / 10
            beta = 1
            for i in range(n_centers):
                for _ in range(samples):
                    [x, y] = [self.centers_set[i][0] + round(gammavariate(alpha, beta)),
                              self.centers_set[i][1] + round(gammavariate(alpha, beta))]
                    self.grid[x, y] += 1

        elif dist_type == 'uniform':  # for this distribution, the n_centers is not used
            xy_min = [0, 0]
            xy_max = [self.lines, self.columns]
            coord = np.random.default_rng().integers(low=xy_min, high=xy_max, size=(samples, 2))
            for c in coord:
                self.grid[c[0], c[1]] += 1

        if plot:
            plt.matshow(self.grid, origin='lower')
            plt.title('Grid with random points')
            plt.colorbar()
            plt.show()

    def distance_matrix(self, coordinates):
        n_coord = len(coordinates)
        self.dist_mtx = np.ndarray(shape=(n_coord, self.lines, self.columns))
        coord_map = np.indices((self.lines, self.columns)).transpose(
            (1, 2, 0))  # coordinates map used to calculate the distance

        for i in range(1, n_coord):
            map = np.empty(shape=(self.lines, self.columns))
            for line in coord_map:
                for coord in line:
                    map[coord[0], coord[1]] = math.dist((coordinates[i][0], coordinates[i][1]), (coord[0], coord[1]))
                    self.dist_mtx[i] = map




###################################

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal,ogr,osr
import rasterio 

#Daqui para baixo transformar em funções para por no sama
#def Rasterize(input_shapefile_path,output_raster_path):
#    shp = ogr.Open(f'{input_shapefile_path}') #Get_shapefile_attributes():r'RiodeJaneiro_shp\RJ_shape\buildings.shp'
#    src_layer = shp.GetLayer()
#
#    raster_location = f'{output_raster_path}'
#    pixel_size = 0.00000325872
#    xmin, xmax, ymin, ymax = src_layer.GetExtent() # Boundaries
#
#    x_res = int(round((xmax-xmin)/pixel_size)) #Resolução X
#    y_res = int(round((ymax-ymin)/pixel_size)) #Resolução Y
#
#    target_ds = gdal.GetDriverByName('GTiff').Create(raster_location,x_res,y_res,1,gdal.GDT_Float32,['COMPRESS=LZW']) # 1 = numbandas 
#    target_ds.SetGeoTransform((xmin,pixel_size,0.0,ymax,0.0,-pixel_size))
#    srse = osr.SpatialReference()
#    projection = 'EPSG:4326'
#    srse.SetWellKnownGeogCS(projection)
#    target_ds.SetProjection(srse.ExportToWkt()) 
#    band = target_ds.GetRasterBand(1)
#    target_ds.GetRasterBand(1).SetNoDataValue(0) # era -9999
#    band.Fill(0) #-9999
#    gdal.RasterizeLayer(target_ds,[1],src_layer,None,None,[1],options = ['ALL_TOUCHED=TRUE','ATRIBUTE=osm_id'])
#    target_ds = None
#    
#
#    with rasterio.open(raster_location) as ra: # Abrindo o raster -> Open_Raster():
#        raster = ra.read()
#        bounds = ra.bounds
#        transform = ra.transform
#        raster = raster.squeeze()
#        num_bands = ra.count
#        print(f'Dimensões (x,y): {raster.shape}') #Width,Height
#        print(f'num de bandas: {num_bands}')
#        plt.figure(figsize=(8, 6))
#        plt.imshow(raster, cmap='binary')  # Adjust the colormap as needed
#        plt.colorbar(label='Indoor/Outdoor')  # Add colorbar with label
#        plt.title('Raster Data')
#        plt.xlabel('Column/Width#')
#        plt.ylabel('Row/Height#')
#        plt.show()
#    return raster


input = 'rasters/shapefiles/londres/teste_londres.shp'
output = 'rasters/testepy_londres_debug3.tif'

raster = Raster()
raster.rasterize_shapefile(input_shapefile_path=input,output_raster_path=output)
final_raster = raster.open_rasterfile(output_raster_path=output)
print(final_raster.shape)

grid = Grid()

grid.make_grid(lines=final_raster.shape[0], columns=final_raster.shape[1])

# verificando se é  indoor ou outdoor (2 maneiras)
#is_indoor = final_raster[grid.grid!=0] # 0 = outdoor e 1 = indoor  

is_indoor = final_raster != 0 

unique,count = np.unique(final_raster,return_counts=True)

print(np.asarray((unique,count)).T)

print(is_indoor)

#print(outdoor)

grid.make_points(dist_type='uniform', samples=100000, n_centers=0,plot=True) # isso aqui precisa vir do arquivo de parâmetros


