import matplotlib.pyplot as plt

from map import Map

folder = '..\\map_data'
mapa = Map()
mapa.load(path=folder + '\\30m.pkl')
mapa.load_general_map_info(path=folder + '\\Brasil_Sce_2010.csv', id_column='Cod_Setor', delimiter=';')
idx_map, mask = mapa.clip_shape(shape=mapa.idx_mtx, criteria='Tijuca', var='Nm_Bairro', plot=True)
wgt_map = mapa.apply_mask(shape=mapa.wgt_mtx, mask=mask, plot=True)
dst_map = mapa.apply_mask(shape=mapa.dst_mtx, mask=mask, plot=True)

points_map, point_list = mapa.uniform_dist(n_samples=1000, id_mtx=idx_map, dnst_map=dst_map)

plt.imshow(points_map)
plt.colorbar()
plt.show()
