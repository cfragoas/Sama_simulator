import georasters as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simpledbf import Dbf5
from db_connect import get_table_sql
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans
import pickle

class Map:
    def __init__(self):
        self.resolution = None  # map resolution to be extracted
        self.data = None  # reference data for the weight calculations
        self.df_ref = None  # reference index and count table
        self.spacial_ref = None  # coordinate reference table
        self.idx_table = None  # matrix with cell reference
        self.idx_mtx = None  # matrix (from raster) with the reference index
        self.wgt_mtx = None  # matrix (from raster) with the point's weight
        self.dst_mtx = None  # matrix (from raster) with the point's density probability
        self.centers = None  # centers for each region of the original shapefile
        self.sample_mtx = None  # position reference for sampled points for density probability function (not mandatory)
        self.sample_list = None  # list of positions for each sample (not mandatory)
        self.general_info = None  # a general information table related to idx_table (not mandatory)
        self.id_column = None  # id column from general_info to be used as a identifier

    # this function only works with the original data files (from .itf and from a mysql)
    # please ignore it and create a wgt_mtx, idx_mtx and idx_table to use the rest of functions
    def extract_data(self, folder, resolution=100, plot=False):
        self.resolution = resolution # resolution = 100 # it can bem 100, 50 or 30 m

        # file folder
        # folder = 'shape_rj'

        if self.resolution == 100:
            # 100 m resolution
            img = gr.from_file(folder + '/rj_100m2.tif')  # reading .tif raster file

            # this step is needed because the .tif file alone does not contaim the fields extracted from the shapefile
            tab_ref = Dbf5(folder + '/rj_100m2.tif.vat.dbf')  # read .dbf file related to the .tif

        elif self.resolution == 50:
            # 50 m resolution
            img = gr.from_file(folder + '/rj_50m.tif')  # reading .tif raster file

            # this step is needed because the .tif file alone does not contaim the fields extracted from the shapefile
            tab_ref = Dbf5(folder + '/rj_50m.tif.vat.dbf')  # read .dbf file related to the .tif

        elif self.resolution == 30:
            # 30 m resolution
            img = gr.from_file(folder + '/rj_30m.tif')  # reading .tif raster file

            # this step is needed because the .tif file alone does not contaim the fields extracted from the shapefile
            tab_ref = Dbf5(folder + '/rj_30m.tif.vat.dbf')  # read .dbf file related to the .tif


        self.df_ref = tab_ref.to_dataframe()  # converting the .dbf to dataframe
        self.df_ref['COD_SETOR'] = pd.to_numeric(self.df_ref['COD_SETOR'])

        self.spacial_ref = img.to_pandas()  # extracting spacial data from .tif

        # creating and filling a matrix with the reference files from the raster
        img_matrix = np.zeros(img.raster.shape)
        for i, line in enumerate(img.raster):
            img_matrix[i] = line


        # getting the values to be filled in the raster from the sql table
        self.data = get_table_sql('bandalarga_setor_censitario_2017')
        self.data = self.data.rename(columns={'Cod_setor_2017':'COD_SETOR'})  # renaming the field to ease merge with other tables

        # merging the data
        merged_table = self.df_ref.merge(self.data, how='left')

        # raster with reference features
        raster = np.array(img.raster)

        # merging the raster with the merger_table values
        self.idx_mtx = np.zeros(shape=img.raster.shape)
        self.wgt_mtx = np.zeros(shape=img.raster.shape)

        for i, value in enumerate(merged_table.VALUE):
            filter = np.where(raster == value)
            self.idx_mtx[filter] = merged_table.COD_SETOR[i]
            if np.isnan(merged_table.bandalarga_cor_cor[i]):
                if i > len(merged_table.VALUE):
                    j = i
                    k = i
                    non_zero = False
                    while non_zero is False:
                        if (np.nan_to_num(merged_table.bandalarga_cor_cor[i+1]) + np.nan_to_num(merged_table.bandalarga_cor_cor[i-1]))/2 == 0:
                            j = j + 1
                            k = k - 1
                        else:
                            non_zero = True
                    self.wgt_mtx[filter] = (np.nan_to_num(merged_table.bandalarga_cor_cor[j]) + np.nan_to_num(merged_table.bandalarga_cor_cor[k]))/2
                else:
                    self.wgt_mtx[filter] = np.nan_to_num(merged_table.bandalarga_cor_cor[i-1])

            else:
                self.wgt_mtx[filter] = merged_table.bandalarga_cor_cor[i]

        # clipping to reduce raster size
        if resolution == 100:
            self.idx_mtx = self.clip_shape(self.idx_mtx, min_x=14, max_x=700, min_y=143, max_y=475)  # valores para 100 m
            self.wgt_mtx = self.clip_shape(self.wgt_mtx, min_x=14, max_x=700, min_y=143, max_y=475)  # valores para 100 m
        elif resolution == 50:
            self.idx_mtx = self.clip_shape(self.idx_mtx, min_x=72, max_x=1428, min_y=305, max_y=951)  # valores para 50 m
            self.wgt_mtx = self.clip_shape(self.wgt_mtx, min_x=72, max_x=1428, min_y=305, max_y=951)  # valores para 50 m
        elif resolution == 30:
            self.idx_mtx = self.clip_shape(self.idx_mtx, min_x=116, max_x=2310, min_y=489, max_y=1530)  # valores para 30 m
            self.wgt_mtx = self.clip_shape(self.wgt_mtx, min_x=116, max_x=2310, min_y=489, max_y=1530)  # valores para 30 m

        # find cluster centroids
        unq_vals = np.unique(self.idx_mtx)
        self.centers = []
        for j, val in enumerate(unq_vals):
            if val != 0:
                cens_sct = np.where(self.idx_mtx == val)
                cens_sct2 = np.zeros(shape=(cens_sct[0].shape[0],2))
                for i, value in enumerate(cens_sct[0]):
                    cens_sct2[i] = [value, cens_sct[1][i]]
                clf = NearestCentroid()
                kmeans = KMeans(n_clusters=1, random_state=0).fit(cens_sct2)
                center = np.round(kmeans.cluster_centers_[0])
                # clf.fit(cens_sct2, np.ones(shape=cens_sct[0].shape[0]))
                self.centers.append([val, center[0], center[1]])

        self.centers = np.array(self.centers)

        # calculating the density of the weight of each point
        self.idx_table = pd.DataFrame(self.data[['COD_SETOR', 'bandalarga_cor_cor']])
        self.idx_table = self.idx_table.merge(self.df_ref[['COD_SETOR', 'COUNT']])
        self.idx_table.COUNT = self.idx_table.bandalarga_cor_cor/self.idx_table.COUNT
        self.idx_table = self.idx_table.rename(columns={'COUNT' : 'density'}, inplace=False)

        if plot:
            self.plot_map(self.idx_mtx, title='index matrix')
            self.plot_map(map=self.wgt_mtx, title='weight matrix')

        return self.idx_table, self.idx_mtx, self.wgt_mtx, self.centers

    # also ignore this function
    def get_table_sql(name_table):
        import mysql.connector
        name_table = 'bandalarga_setor_censitario_2017'

        localuser_pswd = 'TvGlobo_123'
        cnx = mysql.connector.connect(user='root', password=localuser_pswd,
                                      host='localhost',
                                      database='distribuir')

        df = pd.read_sql('SELECT * FROM ' + name_table, con=cnx)

        return df

    # this function in load a table information related to the indexes form idx_table
    # this is necessary to clip the matrix based on a outside condition
    # this will not save or load with all other variables
    def load_general_map_info(self, path, id_column, delimiter=','):
        if '.dbf' in path:
            ref_tab = Dbf5(path)
            self.general_info = ref_tab.to_dataframe()
        elif '.csv' in path:
            self.general_info = pd.read_csv(path, on_bad_lines='skip', delimiter=delimiter)
        else:
            print('file type needs to be .csv or .dbf !!!')
            return

        self.general_info = self.general_info.rename(columns={id_column: 'id'})
        self.id_column = id_column

    # this exists to free some memory if necessary
    def clear_general_map_info(self):
        self.general_info = None

    def clip_shape(self, shape, min_x=0, min_y=0, max_x=0, max_y=0, criteria=None, var=None, map_info=None, plot=False):
        if min_x+min_y+max_x+max_y != 0:
            shape = shape[:, range(min_x, shape.shape[1])]
            shape = shape[range(min_y, shape.shape[0])]
            shape = shape[:, range(0, max_x - min_x)]
            shape = shape[range(0, max_y - min_y)]

            return shape
        elif criteria is not None and var is not None:
            if map_info is None:
                map_info = self.general_info
            if map_info is None:
                print('To use a criteria, a map_info table is needed !!!')
                return

            # execution the clip per criteria
            shape_features = np.unique(shape[shape != 0])
            criteria_features = pd.merge(pd.DataFrame(shape_features, columns=['id']), map_info[['id', var]])
            criteria_features = criteria_features[criteria_features[var] == criteria]
            mask = np.isin(shape, np.array(criteria_features['id']))
            new_shp = self.apply_mask(shape=shape, mask=mask, plot=plot)

            # new_shp[mask] = shape[mask]
            #
            # # aqui por enquanto NÃO FUNCIONA !!!
            # # find the first non-zero for each dimension
            # max_x_arr = np.max(new_shp, axis=0)
            # max_y_arr = np.max(new_shp, axis=1)
            # x = np.where(max_x_arr != 0)[0]
            # y = np.where(max_y_arr != 0)[0]
            # new_shp = new_shp[y]
            # new_shp = new_shp[:, x]

            return new_shp, mask

        else:
            print('Need to set the x-y max/min values or a conditions to use clip function !!!')
            return

    def apply_mask(self, shape, mask, plot=False):
        new_shp = np.zeros(shape=shape.shape)
        new_shp[mask] = shape[mask]

        max_x_arr = np.sum(mask, axis=0)
        max_y_arr = np.sum(mask, axis=1)
        x = np.where(max_x_arr != 0)[0]
        y = np.where(max_y_arr != 0)[0]
        new_shp = new_shp[y]
        new_shp = new_shp[:, x]

        if plot:
            self.plot_map(new_shp)

        return new_shp


    def density_map(self, id_mtx=None, weight_mtx=None, idx_table=None):
        flag = False  # verifying if the function will use class or external variables
        if id_mtx is None or weight_mtx is None or idx_table is None:
            flag = True  # when flag is true, it will save the results in class variables
            id_mtx = self.idx_mtx
            weight_mtx = self.wgt_mtx
            idx_table = self.idx_table

            if self.idx_mtx is None or self.wgt_mtx is None or self.idx_table is None:
                print('Need to set all input arguments for the function or class')
                return

        dnst_map = np.zeros(shape=weight_mtx.shape)
        for index, row in idx_table.iterrows():
            dnst_map[id_mtx == row.COD_SETOR] = row.density

        weight_sum = np.sum(dnst_map)
        dnst_map = dnst_map/weight_sum

        if flag:
            self.dst_mtx = dnst_map

        return dnst_map

    def uniform_dist(self, n_samples, id_mtx=None, dnst_map=None):
        flag = False  # verifying if the function will use class or external variables
        if id_mtx is None or dnst_map is None:
            flag = True  # when flag is true, it will save the results in class variables
            id_mtx = self.idx_mtx
            # weight_mtx = self.wgt_mtx
            dnst_map = self.dst_mtx

            if self.idx_mtx is None or self.wgt_mtx is None or self.dst_mtx is None:
                print('Need to set all input arguments for the function or class')
                return

        if id_mtx.shape != dnst_map.shape:
            print('CENSITARY AND WWIGHT MATRIXES WITH DIFFERENT SHAPES !!!')
            return

        # checking the size of the matrix to generate the points
        x_size = id_mtx.shape[0]
        y_size = id_mtx.shape[1]
        xy_min = [0, 0]
        xy_max = [x_size-1, y_size-1]

        point_list = []

        points_map = np.zeros(shape=id_mtx.shape)
        dnst_map_scld = dnst_map/np.max(dnst_map)  # scaling to make values between 0 and 1

        complete = False  # this variable is to inform if all the samples are sampled on the matrix map
        to_complete = n_samples  # variable that stores the number of samples not sampled wet
        n_points = np.round(n_samples / 2).astype(int)  # firstly, half of total samples are sampled

        # this variable is important, because we will not draw a minimum value of zero but it needs
        # to be lower than the minimum probability on the matrix
        min_sp = np.min(dnst_map_scld[dnst_map_scld != 0])/1000

        while not complete:
            if to_complete <= n_points:  # this ensures that the proess will always samples the exact necessary number
                n_points = to_complete
                # print(n_points)
            # else:
            #     n_points = np.round(n_samples / 2).astype(int)

            # drawing multiple point coordinates from a uniform distribution
            points = np.round(np.random.uniform(low=xy_min, high=xy_max, size=(n_points, 2))).astype(int)

            # if n_points == 1:
            #     points = np.expand_dims(points, axis=0)

            # checking if at leats one point is in a valid coordinate (not of a zero probability point)
            while np.sum(dnst_map[points[:, 0], points[:, 1]]) == 0:
                points = np.round(np.random.uniform(low=xy_min, high=xy_max, size=(n_points, 2))).astype(int)
            # drawing probability values for each point from a uniform distribution
            probability = np.random.uniform(low=min_sp, high=1, size=n_points)

            to_sample = dnst_map_scld[points[:, 0], points[:, 1]] >= probability  # will only use points that attend this criteria
            points_map[points[to_sample, 0], points[to_sample, 1]] += 1
            point_list.append(points[to_sample])
            to_complete -= np.sum(to_sample)

            if to_complete == 0:
                complete = True

        if flag:
            self.sample_mtx = points_map
            self.sample_list = point_list

        return points_map, point_list

    def clear_points(self):  # to clear the sampled points
        self.sample_mtx = None
        self.sample_list = None

    def save(self, folder, name):
        obj = {
            'resolution':self.resolution, 'data': self.data, 'df_ref': self.df_ref, 'spacial_ref': self.spacial_ref,
            'idx_table': self.idx_table, 'idx_mtx': self.idx_mtx, 'wgt_mtx': self.wgt_mtx, 'dst_mtx': self.dst_mtx,
            'centers': self.centers,
            # 'sample_mtx': self.sample_mtx, 'sample_list': self.sample_list
        }

        with open(folder + name, 'wb') as f:
            pickle.dump(obj, f)
            f.close()

    def load(self, path):
        with open(path, 'rb') as f:
            loaded_obj = pickle.load(f)

        self.resolution = loaded_obj['resolution']
        self.data = loaded_obj['data']  # reference data for the weight calculations
        self.df_ref = loaded_obj['df_ref']  # reference index and count table
        self.spacial_ref = loaded_obj['spacial_ref']  # coordinate reference table
        self.idx_table = loaded_obj['idx_table']  # matrix with cell reference
        self.idx_mtx = loaded_obj['idx_mtx']  # matrix (from raster) with the reference index
        self.wgt_mtx = loaded_obj['wgt_mtx']  # matrix (from raster) with the point's weight
        self.dst_mtx = loaded_obj['dst_mtx']  # matrix (from raster) with the point's density probability
        self.centers = loaded_obj['centers']  # centers for each region of the original shapefile

    def plot_map(self, map, title=None):
        plt.matshow(map)
        plt.colorbar()
        if title is not None:
            plt.title(title)
        plt.show()