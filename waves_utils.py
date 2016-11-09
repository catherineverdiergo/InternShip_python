# ----------------------------------------
# -*- coding: utf-8 -*-
# created by Catherine Verdier on 29/09/2016
# ----------------------------------------

import pandas as pd
import aphp_waves_dic as wd
from matplotlib import pyplot as plt
from datetime import timedelta
import numpy as np
import drill_queries as dq
from sklearn.preprocessing import scale, MinMaxScaler
import pyriemann
from pyriemann import estimation, utils
from pyriemann.utils import mean
import seaborn as sns
import random
import matplotlib.ticker as tkr
from sklearn.linear_model import LinearRegression
import scipy as sc

pd.options.mode.chained_assignment = None


def to_time_series(df_wave):
    """
    Transform a pandas wave dataframe to a time series
    :param df_wave:
    :return: None
    """
    df_wave.drop('id_measure_type', axis=1, inplace=True)
    df_wave.index = df_wave['dt_insert']
    df_wave.drop('dt_insert', axis=1, inplace=True)


def add_time_ticks(t_serie, dt_min, dt_max):
    """
    Add time ticks to a time series
    :param t_serie: time series to update
    :param dt_min: time tick to insert at the begining (with a None value)
    :param dt_max: time tick to insert at the end (with a None value)
    :return: the updated time series
    """
    stop_serie = pd.Series(index=[dt_max])
    start_serie = pd.Series(index=[dt_min])
    t_serie = pd.concat([start_serie, t_serie, stop_serie])
    t_serie.drop(0, axis=1, inplace=True)
    return t_serie


def resample_and_interpolate(t_serie, delay, nb_pts, method='spline', order=3):
    """
    Resample and interpolate a time serie in order to get exactly nb_pts points
    :param t_serie: time series to resample and interpolate
    :param delay: delay between 2 points in minutes
    :param nb_pts: number of points to get at output
    :param method: interpolation method as described at
            http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.interpolate.html
    :param order: interpolation order for methods for which this parameter is needed (as 'polynomial' or 'spline')
    :return: the updated time series
    """
    sp_str = "{}T".format(int(delay))
    t_serie = t_serie.resample(sp_str).mean()
    if len(t_serie) < nb_pts:
        while len(t_serie) < nb_pts:
            dt_max = t_serie.index[len(t_serie) - 1]
            dt_max = dt_max + timedelta(seconds=1)
            t_serie = pd.concat([t_serie, pd.Series(index=[dt_max])])
            t_serie.drop(0, axis=1, inplace=True)
    elif len(t_serie) > nb_pts:
        while len(t_serie) > nb_pts:
            t_serie = t_serie.drop([t_serie.index[len(t_serie) - 1]])
    t_serie = t_serie.interpolate(method=method, order=order, limit=30)
    t_serie = t_serie.ffill()
    t_serie = t_serie.bfill()
    return t_serie


def plot_cov_matrix(cov, title, columns_list, axis, small=False):
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    df = pd.DataFrame(cov)
    df.columns = columns_list
    df.index = columns_list
    plt.title(title)
    if small:
        formatter = tkr.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        sns.heatmap(df, square=True, cmap=cmap, ax=axis, cbar_kws={"orientation": "horizontal", "format": formatter})
        plt.setp(axis.yaxis.get_majorticklabels(), rotation=-35)
        # axis.tick_params(rotation=7)
        plt.setp(axis.xaxis.get_majorticklabels(), rotation=-15)
    else:
        sns.heatmap(df, square=True, cmap=cmap, ax=axis, cbar_kws={"orientation": "horizontal"})
        plt.setp(axis.yaxis.get_majorticklabels(), rotation=-35)


def plot_spectrum(waves_matrix, Fs, labels, colors):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(waves_matrix[0])  # length of the signal
    k = sc.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(n / 2)]  # one side frequency range
    df = pd.DataFrame(columns=labels)
    for i in range(waves_matrix.shape[0]):
        Y = sc.fft(waves_matrix[i].ravel()) / n  # fft computing and normalization
        Y = Y[range(n / 2)]
        df[labels[i]] = abs(Y)
    df.index = frq
    df.plot.area(ax=plt.gcf().axes[1], alpha=0.7, color=colors)
    plt.xlabel("Frq (Hz)")
    plt.ylabel("FFT")


# def plot_ar_psd(waves_matrix, Fs, labels, colors):
#     """
#     Plots autoregressive power spectrum density y(t)
#     """
#     n = len(waves_matrix[0])  # length of the signal
#     k = sc.arange(n)
#     T = n / Fs
#     frq = k / T  # two sides frequency range
#     frq = frq[range(n / 2)]  # one side frequency range
#     df = pd.DataFrame(columns=labels)
#     from spectrum import *
#     for i in range(waves_matrix.shape[0]):
#         ar_values, error = arcovar(waves_matrix[i].ravel(), 15)
#         psd = arma2psd(ar_values, sides='centerdc')
#         df[labels[i]] = abs(psd)
#     df.index = frq
#     df.plot.area(ax=plt.gcf().axes[1], alpha=0.7, color=colors)
#     plt.xlabel("Frq (Hz)")
#     plt.ylabel("FFT")


class WavesUtilities:
    """
    Class holding method utilities to process waves
    """

    def __init__(self, voi, spark_context, hdfs_db_path):
        """
        Constructor
        :param voi: list of interest for waves
        :param spark_context: spark context to be able to write Parquet tables
        :param hdfs_db_path: target hdfs directory for Parquet tables
        """
        self._voi = voi
        self._dic = wd.WavesDic()
        self._spark_ctx = spark_context
        self._db_dir = hdfs_db_path
        self._dq = dq.DrillQueries("drill_eds", voi)
        self._df_stats = self._dq.get_sensor_util_stats()
        self._colors = ['#40bf80', '#668cff', '#ffa64d', '#ff33bb', '#330033', '#4dffc3', '#805500', '#999900']

    def get_waves_from_pdf(self, all_waves):
        """
        Separate measures by wave type regarding to the waves dictionary
        :param all_waves: all measures related to VOI as defined in the waves dictionary selected for a given
                list of case ids
        :return: a dictionary of time series (one entry per VOI)
        """
        result = {}
        for v in self._voi:
            limits = self._dq._sigmas_thres[v]
            aphp_codes = self._dic.get_all_voi_codes([v])
            df_filter = [False] * len(all_waves)
            for code in aphp_codes:
                df_filter |= (all_waves['id_measure_type'] == code)
            df_filtered = all_waves[df_filter]
            df_filtered = df_filtered[(df_filtered['value_numeric'] >= limits[0]) & \
                                      (df_filtered['value_numeric'] <= limits[1])]
            if len(df_filtered) == 0:
                result = None
                break
            result[v] = df_filtered.sort_values(by='dt_insert', ascending=1)
        return result

    def plot_waves(self, waves, create_fig=True):
        """
        Plot waves in a matplotlib figure
        :param waves: all waves for a given case as a dictionary of time series
        :param create_fig: optional ==> True if the function should create a pyplot figure
        :return: None
        """
        if create_fig:
            plt.figure(figsize=(15, 5))
        for i, metric in enumerate(waves.keys()):
            label = self._dic.get_label(metric)
            plt.plot(waves[metric].dt_insert.values, waves[metric].value_numeric.values, label=label,
                     color=self._colors[i % len(self._colors)], linewidth=3.0)
            plt.plot(waves[metric].dt_insert.values, waves[metric].value_numeric.values, '*', color='#000000')
            plt.legend()

    def norm_waves(self, df_ranges, df_sensor_case, nb_pts, method='time'):
        """
        Generate a set of matrices for a set of cases (batch processing)
        :param df_ranges: considered cases ids and features (util ranges for measures, death date)
        :param df_sensor_case: dataframe of measures related with case
        :param nb_pts: number of desired points
        :param method: method for interpolation (default is spline with d° 3)
        :return: a set of matrices (waves for cases) + a set of target (booleans : dead / not dead)
        """
        resultT = np.zeros((len(self._voi), nb_pts), dtype=np.float)
        # separate waves
        waves_dic = self.get_waves_from_pdf(df_sensor_case)
        if waves_dic is None:
            return None, None, None
        df_dt_range = df_ranges[(df_ranges['id_nda'] == str(df_sensor_case['id_nda'].unique()[0])) & \
                                (df_ranges['j1'] == str(df_sensor_case['j1'].unique()[0]))]
        # Estimate the interval in minutes to resample
        mn_interval = np.round(((pd.to_datetime(df_dt_range['dt_max']) -
                                 pd.to_datetime(df_dt_range['dt_min'])).astype('timedelta64[ms]').astype(int)
                                / 1e+03 / nb_pts / 60).unique()[0])
        # for i, key in enumerate(waves_dic.keys()):
        for i, key in enumerate(self._voi):
            # add dt_min and dt_max to each time serie
            serie = waves_dic[key][['dt_insert', 'value_numeric']]
            serie.index = serie['dt_insert']
            serie.drop('dt_insert', axis=1, inplace=True)
            serie = add_time_ticks(serie, df_dt_range['dt_min'], df_dt_range['dt_max'])
            # resample and interpolate
            serie = resample_and_interpolate(serie, mn_interval, nb_pts, method=method)
            serie['dt_insert'] = serie.index
            waves_dic[key] = serie
            resultT[i] = serie['value_numeric'].values
        # print(df_dt_range['dt_deces'], df_dt_range['dt_deces'].unique()[0] != "NaT", df_dt_range['dt_deces'].isnull())
        # return resultT, df_dt_range['dt_deces'].unique()[0] != "NaT", waves_dic  # False when survivor / True when dead
        return resultT, df_dt_range['dt_deces'].unique()[0] != 'None', waves_dic  # False when survivor / True when dead

    def build_pyriemann_input_matrix(self, nb_splits, nb_pts):
        """
        Build the 3d pyriemann input matrix by blocks with selected waves
        :param nb_splits: number of blocks of ids to create
        :param nb_pts: number of points desired to resample waves
        :return: 3d matrix + target vector (mortality)
        """
        from pyspark.sql.functions import concat
        dfs_ids = self._spark_ctx._sql.read.parquet(self._db_dir + "/nda_j1_deces") \
            .select(concat('id_nda', 'j1').alias('id_case')).distinct()
        # split dfs_ids in nb_splits RDDs
        rdd_ids_blocks = dfs_ids.rdd.randomSplit([nb_splits] * nb_splits, 42)
        matrix3d = []
        target_vector = []
        # get dates intervals by case
        df_ranges = self._dq.get_dt_ranges()
        list_cases = []
        # Process by blocks
        for i, rdd in enumerate(rdd_ids_blocks):
            print ("block {} / {}".format(i + 1, len(rdd_ids_blocks)))
            # convert to Dataframe
            df_block = rdd.toDF()
            # save as temporary parquet file
            df_block.write.parquet(self._db_dir + "/tmp_nda_j1_ids", mode='overwrite')
            # prepare Drill query
            dq = self._dq.queries["QUERY_SELECT_SENSOR_BLOCK"]
            dfp = self._dq.drill_conn.df_from_query(dq)
            keys = dfp['id_case'].unique()
            for key in keys:
                # restriction on id_case
                dfp_key = dfp[dfp['id_case'] == key]
                # separe and normalize several waves by type
                # (Heart Rate / Respiration rate / ABP systolic / ABP diastolic)
                xy4case = self.norm_waves(df_ranges, dfp_key, nb_pts)
                if xy4case[0] is None:
                    print("case {} removed (not enough meaningful values)".format(key))
                else:
                    list_cases.append(key)
                    matrix3d.append(xy4case[0])
                    target_vector.append(xy4case[1])
            print("{} keys processed".format(np.size(keys)))
        return np.stack(matrix3d), target_vector, list_cases

    def build_pyriemann_input_matrix_one_shot(self, nb_pts):
        """
        Build the 3d pyriemann input matrix by blocks with selected waves
        :param nb_pts: number of points desired to resample waves
        :return: 3d matrix + target vector (mortality)
        """
        matrix3d = []
        target_vector = []
        # get dates intervals by case
        df_ranges = self._dq.get_dt_ranges()
        # prepare Drill query
        dq = self._dq.queries["QUERY_SELECT_SENSOR"]
        print("loading rows from db...")
        dfp = self._dq.drill_conn.df_from_query(dq)
        print("done")
        list_cases = []
        keys = dfp['id_case'].unique()
        for i, key in enumerate(keys):
            # restriction on id_case
            dfp_key = dfp[dfp['id_case'] == key]
            # separe and normalize several waves by type
            # (Heart Rate / Respiration rate / ABP systolic / ABP diastolic)
            xy4case = self.norm_waves(df_ranges, dfp_key, nb_pts)
            if xy4case[0] is None:
                print("case {} removed (not enough meaningful values)".format(key))
            else:
                list_cases.append(key)
                matrix3d.append(xy4case[0])
                target_vector.append(xy4case[1])
            if i % 300 == 0:
                print("{} keys processed".format(i + 1))
        return np.stack(matrix3d), target_vector, list_cases

    @property
    def dq(self):
        return self._dq


class WaveDataset:
    def __init__(self, l_vois, matrix_number, estimator='scm', method='min_max'):
        """
        Constructor
        :param l_vois: list of labels for variables of interest included in the matrix
        :param matrix_number: number id to identify the matrix data :
                matrix data are in "icu_matrix"+str(matrix_number)+".pyriemann.npy" file
                target data are in "icu_target" + str(matrix_number) + ".pyriemann.npy" file
                case ids data are in "icu_ids" + str(matrix_number) + ".pyriemann.npy" file
        :param estimator: method to compute covariance matrix (default is scm ==> empirical covariance
                can be 'lwf' ==> ledoit & wolf or 'oas' ==> Oracle Approximation Shrinkage)
        """
        self._lvois = l_vois
        self._matrix = np.load("icu_matrix" + str(matrix_number) + ".pyriemann.npy")
        self._original_matrix = np.zeros(self._matrix.shape)
        np.copyto(self._original_matrix, self._matrix)
        self._target = np.load("icu_target" + str(matrix_number) + ".pyriemann.npy")
        self._ids = np.load("icu_ids" + str(matrix_number) + ".pyriemann.npy")
        self.wu = WavesUtilities(self._lvois, None, None)
        # A template query to select all measures for a given patient case
        self._qry_case_tpl = "select lpad(cast(id_nda as VARCHAR),10,'0') as id_nda, CAST(dt_deb as VARCHAR) as j1, \
                            CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'), CAST(dt_deb as VARCHAR)) as id_case, \
                            id_measure_type, dt_insert, value_numeric \
                            from icu_sensor_util \
                            where CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'), CAST(dt_deb as VARCHAR))='{}'"
        # Retrieve the dataframe with stay limits for all cases
        self._df_dt_ranges = self.wu.dq.get_dt_ranges()
        self.scale_matrix(method=method)
        self._estimator = estimator
        self._cov = pyriemann.estimation.Covariances(estimator=self._estimator).fit_transform(self._matrix)
        self._colors = ['#40bf80', '#668cff', '#ffa64d', '#ff33bb', '#330033', '#4dffc3', '#805500', '#999900']

    def scale_matrix(self, method='min_max'):
        """
        scale the input data (self._matrix)
        :param method: min max scaling by default (otherwise standardization is applied)
        :return: None
        """
        np.copyto(self._matrix, self._original_matrix)
        for case in range(self._matrix.shape[0]):
            if method == 'min_max':
                self._matrix[case] = MinMaxScaler().fit_transform(self._matrix[case])
            else:
                self._matrix[case] = scale(self._matrix[case])

    def plot_waves(self, waves_matrix, create_fig=True):
        """
        Plot waves stored in input matrix case (1 row by voi)
        :param waves_matrix: all waves for a given case as a matrix (1 row per wave)
        :param create_fig: optional ==> True if the function should create a pyplot figure
        :return: None
        """
        if create_fig:
            plt.figure(figsize=(15, 5))
        ranges = xrange(waves_matrix.shape[1])
        for i, metric in enumerate(self._lvois):
            plt.plot(ranges, waves_matrix[i], label=metric,
                     color=self._colors[i % len(self._colors)], linewidth=3.0)
            plt.plot(ranges, waves_matrix[i], '*', color='#000000')
            plt.legend()

    def plot_spectrum(self, waves_matrix, create_fig=True):
        """
        Plot spectrums for waves stored in input matrix case (1 row by voi)
        :param waves_matrix: all waves for a given case as a matrix (1 row per wave)
        :param create_fig: optional ==> True if the function should create a pyplot figure
        :return: None
        """
        # compute sample frequency
        nb_pts = np.size(waves_matrix[0])  # number of measures by day
        sf = float(nb_pts) / 24. / 3600.  # sample frequency
        if create_fig:
            plt.figure(figsize=(15, 5))
        plot_spectrum(waves_matrix, sf, self._lvois, self._colors)

    @staticmethod
    def voi_linear_regressors(waves_matrix):
        """
        Create and fit a linear regressor for each voi
        :param waves_matrix: 2D ndarray, case input matrix (one row by voi)
        :return:
        """
        result = []
        steps = range(waves_matrix.shape[0])
        x = np.array(range(waves_matrix.shape[1])).astype(float)
        for i in steps:
            y = waves_matrix[i]
            result.append(LinearRegression().fit(x.reshape(-1, 1), y))
        return result

    def remove_trend_4_case(self, waves_matrix):
        """
        Attempt to eliminate the increasing trend when of each variable using a linear regression
        :param waves_matrix:
        :return:
        """
        lrs = self.voi_linear_regressors(waves_matrix)
        steps = range(waves_matrix.shape[0])
        x = np.array(range(waves_matrix.shape[1])).astype(float)
        for i in steps:
            waves_matrix[i] -= lrs[i].coef_[0] * x

    def remove_trend(self):
        """
        Apply the remove_trend method to each case matrix
        :return:
        """
        self._matrix = np.apply_along_axis(self.remove_trend, 0, self._matrix)

    def case_figure(self, id_case):
        """
        Figure with several data related with a given case (covariance matrix, original waves, resampled waves
        and resampled waves stored in the 3D matrix ==> to be able to check the right labels associated to each
        row in the 3D matrix)
        :param id_case: case identifier as string (id_nda+date_of_entrance) or as index in the 3D pyriemann matrix
        :return: None
        """
        # retrieve case covariance and original data
        if type(id_case) is str:
            idx_case = np.where(self._ids == id_case)[0][0]
        else:
            idx_case = id_case
            id_case = self._ids[idx_case]
        cov_case = self._cov[idx_case].reshape((self._cov.shape[1], self._cov.shape[2]))
        original_X_case = self._original_matrix[idx_case].reshape(
                (self._original_matrix.shape[1], self._original_matrix.shape[2]))
        # retrieve original points from database
        q = self._qry_case_tpl.format(id_case)
        df_case = self.wu.dq.drill_conn.df_from_query(q)
        original_dic = self.wu.get_waves_from_pdf(df_case)
        # resample
        _, y, resampled_dic = self.wu.norm_waves(self._df_dt_ranges, df_case, 96, method='time')
        f = plt.figure(figsize=(15, 10))
        plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=1)
        # plot cov matrix
        f.suptitle(id_case)
        if self._target[idx_case]:
            title = 'Outcome: dead'
        else:
            title = 'Outcome: alive'
        plot_cov_matrix(cov_case, title, self._lvois, f.axes[0])
        plt.subplot2grid((3, 3), (0, 1), rowspan=1, colspan=2)
        self.wu.plot_waves(original_dic, create_fig=False)
        plt.subplot2grid((3, 3), (1, 1), rowspan=1, colspan=2)
        self.wu.plot_waves(resampled_dic, create_fig=False)
        plt.subplot2grid((3, 3), (2, 1), rowspan=1, colspan=2)
        self.plot_waves(original_X_case, create_fig=False)
        plt.show()
        return

    def case_figure_trend(self, id_case):
        """
        Figure with several data related with a given case (covariance matrix, original waves, resampled waves
        and resampled waves stored in the 3D matrix ==> to be able to check the right labels associated to each
        row in the 3D matrix)
        :param id_case: case identifier as string (id_nda+date_of_entrance) or as index in the 3D pyriemann matrix
        :return: None
        """
        # retrieve case covariance and original data
        if type(id_case) is str:
            idx_case = np.where(self._ids == id_case)[0][0]
        else:
            idx_case = id_case
            id_case = self._ids[idx_case]
        original_X_case = self._original_matrix[idx_case].reshape(
                (self._original_matrix.shape[1], self._original_matrix.shape[2]))
        case_copy = np.zeros(original_X_case.shape)
        np.copyto(case_copy, original_X_case)
        self.remove_trend_4_case(case_copy)
        f = plt.figure(figsize=(10, 12))
        # plot cov matrix
        if self._target[idx_case]:
            title = 'Outcome: dead'
        else:
            title = 'Outcome: alive'
        title = id_case + ' -- ' + title
        f.suptitle(title)
        plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
        self.plot_waves(original_X_case, create_fig=False)
        plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
        self.plot_waves(case_copy, create_fig=False)
        plt.show()
        return

    def case_figure_spectrum(self, id_case):
        """
        Figure with several data related with a given case (covariance matrix, original waves, resampled waves
        and resampled waves stored in the 3D matrix ==> to be able to check the right labels associated to each
        row in the 3D matrix)
        :param id_case: case identifier as string (id_nda+date_of_entrance) or as index in the 3D pyriemann matrix
        :return: None
        """
        # retrieve case covariance and original data
        if type(id_case) is str:
            idx_case = np.where(self._ids == id_case)[0][0]
        else:
            idx_case = id_case
            id_case = self._ids[idx_case]
        original_X_case = self._original_matrix[idx_case].reshape(
                (self._original_matrix.shape[1], self._original_matrix.shape[2]))
        case_copy = np.zeros(original_X_case.shape)
        np.copyto(case_copy, original_X_case)
        self.remove_trend_4_case(case_copy)
        f = plt.figure(figsize=(10, 12))
        # plot cov matrix
        if self._target[idx_case]:
            title = 'Outcome: dead'
        else:
            title = 'Outcome: alive'
        title = id_case + ' -- ' + title
        f.suptitle(title)
        plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
        self.plot_waves(original_X_case, create_fig=False)
        plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
        self.plot_spectrum(case_copy, create_fig=False)
        plt.show()
        return

    def mean_cov_plot_by_class(self, metric='riemann', add_examples=False):
        """
        Plot mean covariance matrices by class
        :param: metric: metric used to compute the covariance means of each class
        :param add_examples: boolean to add examples of each class in the figure or not
        :return: None
        """
        dead_indices = np.where(self._target == True)[0]
        alive_indices = np.where(self._target == False)[0]
        mean_cov_dead = pyriemann.utils.mean.mean_covariance(self._cov[dead_indices], metric=metric)
        mean_cov_alive = pyriemann.utils.mean.mean_covariance(self._cov[alive_indices], metric=metric)
        f = None
        if not add_examples:
            f = plt.figure(figsize=(10, 7))
        else:
            f = plt.figure(figsize=(18, 12))
        title = "Mean covariances"
        if self._estimator == 'scm':
            title += " (empirical covariances)"
        elif self._estimator == 'lwf':
            title += " (Ledoit & Wolf regularization)"
        elif self._estimator == 'oas':
            title += " (Oracle Approximating regularization)"
        plt.suptitle(title)
        if not add_examples:
            plt.subplot(1, 2, 1)
        else:
            plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
        # plot mean covariance matrix for dead class
        plot_cov_matrix(mean_cov_dead, "Outcome dead", self._lvois, f.axes[0])
        if not add_examples:
            plt.subplot(1, 2, 2)
        else:
            plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
        # plot mean covariance matrix for alive class
        plot_cov_matrix(mean_cov_alive, "Outcome alive", self._lvois, f.axes[2])
        # if add_examples is True, then add cov matrices for 4 cases of each class
        if add_examples:
            dead_ex_indices = random.sample(dead_indices, 4)
            alive_ex_indices = random.sample(alive_indices, 4)
            for i in range(4):
                plt.subplot2grid((4, 4), (2, i), rowspan=1, colspan=1)
                plot_cov_matrix(self._cov[dead_ex_indices[i]], self._ids[dead_ex_indices[i]] + " - dead", \
                                self._lvois, f.axes[4 + 2 * i], small=True)
            for i in range(4):
                plt.subplot2grid((4, 4), (3, i), rowspan=1, colspan=1)
                plot_cov_matrix(self._cov[alive_ex_indices[i]], self._ids[alive_ex_indices[i]] + " - alive", \
                                self._lvois, f.axes[12 + 2 * i], small=True)
        plt.show()


if __name__ == '__main__':
    db_dir = "D:/tmp/IPYNBv2/eds"
    # waves_u = WavesUtilities(['HR', 'RR', 'ABPS', 'ABPD'], None, db_dir)
    #
    # nb_splits = 15
    # nb_pts = 96
    #
    # from time import gmtime, strftime
    #
    # print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    # # pr = waves_u.build_pyriemann_input_matrix(nb_splits, nb_pts)
    # pr = waves_u.build_pyriemann_input_matrix_one_shot(nb_pts)
    # print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    # pr_3d_matrix = pr[0]
    # tgt_vector = pr[1]
    # ids_vector = np.array(pr[2])
    # #
    # # Save 3D matrix in file
    # np.save("icu_matrix5.pyriemann", pr_3d_matrix)
    # np.save("icu_target5.pyriemann", tgt_vector)
    # np.save("icu_ids5.pyriemann", ids_vector)
    #
    # mat = np.load("icu_matrix5.pyriemann.npy")
    #
    # print(np.array_equal(pr_3d_matrix, mat))
    wds = WaveDataset(['HR', 'RR', 'ABPS', 'ABPD'], 5, estimator='oas')
    # wds.case_figure('00031963862014-07-26')
    # wds.case_figure('00061580842015-06-01')
    # wds.case_figure(1001)
    # wds.case_figure(0)
    # wds.case_figure(1)
    # for i in range(wds._cov.shape[0]):
    #     wds.case_figure(i)
    # wds.case_figure_trend('00031963862014-07-26')
    # wds.case_figure_trend(500)
    # wds.case_figure_spectrum(500)
    wds.mean_cov_plot_by_class(metric='riemann', add_examples=True)
    # wds.case_figure(125)
