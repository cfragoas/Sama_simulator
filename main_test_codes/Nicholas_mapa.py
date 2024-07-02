import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal,ogr,osr
import rasterio
import shapely
from make_grid import Grid


 #JUNTAR com o creating raster para mandar um raster output e depois ler ele ( Ver com Christian se dá pra por no PKL e jogar junto com os outputs de cada run do SAMA
raster_location = 'output/testepy.tif'
pixel_size = 0.000325872
xmin, xmax, ymin, ymax = src_layer.GetExtent() # Boundar

x_res = int(round((xmax-xmin)/pixel_size)) #Resolução X
y_res = int(round((ymax-ymin)/pixel_size)) #Resolução y

#Colocar todos as etapas em uma função junto com o raster location Generate_Raster(): Vai ter como parâmetros de entrada os resultados da função get_shapefile_attributes()
target_ds = gdal.GetDriverByName('GTiff').Create(raster_location,x_res,y_res,1,gdal.GDT_Float32,['COMPRESS=LZW']) # 1 = numbandas
target_ds.SetGeoTransform((xmin,pixel_size,0.0,ymax,0.0,-pixel_size))
srse = osr.SpatialReference()
projection = 'EPSG:4326'
srse.SetWellKnownGeogCS(projection)
target_ds.SetProjection(srse.ExportToWkt())
band = target_ds.GetRasterBand(1)
target_ds.GetRasterBand(1).SetNoDataValue(0) # era -9999
band.Fill(0) #-9999
gdal.RasterizeLayer(target_ds,[1],src_layer,None,None,[1],options = ['ALL_TOUCHED=TRUE','ATRIBUTE=osm_id'])
target_ds = None

with rasterio.open('output/testepy.tif') as ra: # Abrindo o raster -> Open_Raster():
    raster = ra.read()
    bounds = ra.bounds
    transform = ra.transform
    raster = raster.squeeze()
    num_bands = ra.count
    print(f'Dimensões (x,y): {raster.shape}') #Width,Height
    print(f'num de bandas: {num_bands}')
    plt.figure(figsize=(8, 6))
    plt.imshow(raster, cmap='binary')  # Adjust the colormap as needed
    plt.colorbar(label='Indoor/Outdoor')  # Add colorbar with label
    plt.title('Raster Data')
    plt.xlabel('Column/Width#')
    plt.ylabel('Row/Height#')
    plt.show()

print(f'Os limites do raster são:\nLimite esquerda:{bounds.left}, Limite direita:{bounds.right}, Limite topo:{bounds.top}, Limite bottom:{bounds.bottom}')

num_pontos = 10000


# classe grid usada no Sama
grid = Grid()
grid.make_grid(lines=raster.shape[0], columns=raster.shape[1])
# agora vou usar a própria função do grid para gerar os pontos
# sugiro você ler as funções da classe para entender o que precisa fazer
# aqui, você precisa verificar o número de pontos gerados fora do mapa (no mar, por ex) para rodar o grid.make points
# até que a condição esteja satisfeita
grid.make_points(dist_type='uniform', samples=num_pontos, n_centers=0) # isso aqui precisa vir do arquivo de parâmetros

# aqui, eu acho que você não precisa,mas são duas maneiras de você obter as coordenadas dos pontos
point_coordinates = np.nonzero(grid.grid) # coordenada em duas dimensões # 1
point_coordinates2 = np.flatnonzero(grid.grid)  # coordenada em uma dimensão 2 

# verificando se é  indoor ou outdoor (2 maneiras)
point_condition = raster[grid.grid!=0] # 0 = outdoor e 1 = indoor  
point_condition2 = raster[point_coordinates] #2

# aí vc terá um grid de pontos no formato da classe do sama, uma matriz de coordenadas
# e uma matriz de condição indoor/outdoor
# aí é vc usar o point_condition para alimentar o seu modelo
# lembre-se de fazer de forma vetorial sempre que possível


