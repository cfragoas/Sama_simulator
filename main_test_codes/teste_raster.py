import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal,ogr,osr
import rasterio
import math
from random import randint, gauss, gammavariate
import sys
import matplotlib.pyplot as plt
#from make_grid import Grid

def highestPowerOf2(n):
    return (np.log2(n & (~(n - 1))))

#################################################
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

################################################
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
        if os.path.exists(self.output_raster_path):
            os.remove(self.output_raster_path)

    def define_scaling_factor(self):
    
        centroid_lat = self.ymax+self.ymin/2 # Valor central
    
        earth_radius = 6378137  # Raio da terra em metros
    
    
        meters_per_degree_lat = (2 * np.pi * earth_radius) / 360 # Fator de escala metros por grau latitude
    
    
        meters_per_degree_long = meters_per_degree_lat * np.cos(np.radians(centroid_lat)) # Fator de escala metros por grau longitude

        # Calcula o fator de escala, resultado em metros por pixel 
        self.y_scale = meters_per_degree_lat * self.pixel_size
        self.x_scale = meters_per_degree_long * self.pixel_size
   
# sobrescrever o make_point para setar as conditions também

###################################


input_file = "rasters/shapefiles/bairros_tcc/gragoata.shp"
#"C:/Users/Usuário/Documents/TCC/resultados_tcc_nicholas/brasil_shp/Estados_Brasil.shp"
output_file = 'rasters/testepy_DUMMY.tif'
burner_value = 'osm_id'
projection = 'EPSG:4326'
rast = Raster(input_shapefile=input_file,output_raster=output_file,projection=projection,burner_value=burner_value,
pixel_size=0.000325872,no_data_value=0,raster_cell_value='int32')
rast.delete_tif_file()
#raster.set_raster_transform_paths(input_shapefile=input,output_raster=output)

rast.rasterize_shapefile() # Cria o raster
print('rasterizou!')
#raster.make_grid()
rast.define_scaling_factor()


with rasterio.open(f'{output_file}') as ra: # Abrindo o raster -> Open_Raster():
    raster = ra.read()
    bounds = ra.bounds
    transform = ra.transform
    raster = raster.squeeze()
    num_bands = ra.count
    dimension = raster.shape


# Obtendo os valores únicos e suas contagens
unique, counts = np.unique(raster, return_counts=True)

# Criando um dicionário para mapear valores únicos às suas contagens
value_counts = dict(zip(unique, counts))

# Calculando o total de valores
total_count = raster.size

# Contando valores diferentes de 0
non_zero_count = total_count - value_counts.get(0, 0)

# Calculando a proporção
non_zero_proportion = non_zero_count / total_count

np.asarray([unique,counts],dtype=np.int32).T
print(f'Dimensao em pixeis no eixo vertical: {dimension[0]} pixeis')
print(f'Dimensao em pxeis no eixo horizontal: {dimension[1]} pixeis')
print(f'Dimensao em metros no eixo vertical: {dimension[0]*rast.x_scale} metros')
print(f'Dimensao em metros no eixo horizontal: {dimension[1]*rast.y_scale} metros')
print(f"Total de valores: {total_count}")
print(f"Valores diferentes de 0: {non_zero_count}")
print(f"Proporcao de valores diferentes de 0: {non_zero_proportion:.2%}") # Para ver o quanto da área do raster corresponde a ambientes indoo;#final_raster = raster.open_rasterfile(output_raster_path=output) # Sai a matriz ndarray
plt.figure(figsize=(10, 10))
plt.imshow(raster, cmap='binary')  # Adjust the colormap as needed
#plt.colorbar(label='Número mapeado para o estado + Distrito Federal')  # Add colorbar with label
#plt.title('Resultado da rasterização do bairro Botafogo')
plt.xlabel('Posição do pixel no eixo horizontal',fontsize=14)
plt.ylabel('Posição do pixel no eixo vertical',fontsize=14)
plt.tight_layout()
#BAIRplt.savefig('tcc_teoricos/raster_estados_brasil.PNG')
plt.show()
#plt.savefig('output/raster_RJ.jpeg')
#print(final_raster.shape,)

#raster.make_grid(lines=final_raster.shape[0], columns=final_raster.shape[1])

#raster.make_points(dist_type='uniform', samples=10000, n_centers=0,plot=False) # isso aqui precisa vir do arquivo de parâmetros

#grid = Grid()

#raster.set_point_condition(raster.grid!=0)

#raster.point_condition
#raster.save_raster(final_raster[raster.grid!=0])

#in_out_user = raster.load_raster()

#print(type(raster.point_condition))

#print('\n Quantidade de usuarios indoor (1) e outdoor (0): ')
#unique,count = np.unique(raster.point_condition,return_counts=True)
#print(np.asarray((unique,count)).T)

#raster.define_scaling_factor()

#grid = Grid()
#raster.delete_raster_npy_file()
#print(raster.grid.shape)
print('\nFoi!')