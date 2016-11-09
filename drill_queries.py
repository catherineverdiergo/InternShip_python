# ----------------------------------------
# -*- coding: utf-8 -*-
# created by Catherine Verdier on 26/09/2016
# ----------------------------------------

import drill_utilities as du
import aphp_waves_dic as wd
import pandas as pd
import numpy as np
from datetime import timedelta
import pyodbc


class DrillQueries:
    """
    Class to build queries and DataFrames to:
        - get the most frequent types of measures provided by ICUs;
        - filter waves rows per relevant cases;
        - ...
    Aim is to provide descriptive statistics about the bag of measures available
    """

    @property
    def voi(self):
        return self._voi

    @property
    def queries(self):
        return self._queries

    @property
    def drill_conn(self):
        return self._drill_conn

    def __init__(self, dsn, voi):
        """
        Constructor ==> initialize Drill connection and build query templates
        :rtype: object
        :param dsn: ODBC Drill Data Source Name
        :param voi: List of variables of interest
        """
        self._drill_conn = du.DrillODBC(dsn)
        self._dic = wd.WavesDic()
        self._voi = voi
        self._sigmas_thres = {}
        self._queries = {
            # get number of measures by type
            'QUERY_MEASURES_COUNTERS': "select count(1) as counter,s.id_measure_type, r.label \
                                        from icu_sensor_24 s, ref_measure r \
                                        where s.id_measure_type = cast(r.code as INT) \
                                        and s.dt_cancel = '' \
                                        group by s.id_measure_type, r.label \
                                        order by counter desc, s.id_measure_type, r.label",
            # Count available cases for the selected VOI
            'QUERY_COUNT_CASES': "select count(distinct CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR))) \
                                    from icu_sensor_24 s \
                                    where s.id_measure_type in {} and s.dt_cancel = ''",
            # Count available measures by case for selected VOI
            'QUERY_GET_MEASURES': "select CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)) as id_ndaj1, \
                                    s.id_measure_type, count(1) as counter \
                                    from icu_sensor_24 s \
                                    where s.id_measure_type in {} \
                                    and s.dt_cancel = '' \
                                    and s.value_numeric > 0 \
                                    group by CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)), s.id_measure_type \
                                    order by CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)), s.id_measure_type",
            # Get patient data by case: age + death date
            'QUERY_CASE_DATA': "select distinct CONCAT(lpad(cast(s.id_nda as VARCHAR),10,'0'),cast(TO_DATE(s.dt_deb) as VARCHAR)) as id_ndaj1, \
                                    p.age, p.dt_deces as pat_dt_deces, p.cd_sex_tr, p.dpt, \
                                    case when (n.dt_deb_nda <= p.dt_deces and p.dt_deces<=n.dt_fin_nda) \
                                    then p.dt_deces else cast(NULL as TIMESTAMP) end as dt_deces \
                                    from icu_pat_info p, icu_sensor_24 s, icu_nda n \
                                    where p.id_nda = cast(s.id_nda as VARCHAR) \
                                    and n.id_nda = cast(s.id_nda as VARCHAR) \
                                    and s.dt_cancel = ''",
            # Get stays global information for cases selected in icu_sensor_24
            'CREATE_ICU_NDA': "create table icu_nda as \
                                    select id_nda, dt_deb_nda, dt_fin_nda from nda_tr \
                                    where cast(id_nda as VARCHAR) in \
                                    (select distinct cast(id_nda as VARCHAR) from icu_sensor_24)",
            # Get minimum dates per VOI and per case
            'QUERY_DT_MIN_ICU': "select CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)) as id_ndaj1, \
                                id_measure_type, min(dt_insert) as min_dt \
                                from icu_sensor_24 \
                                where id_measure_type in {} \
                                and dt_cancel = '' \
                                group by CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)), id_measure_type \
                                order by CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)), id_measure_type",
            # Get maximum dates per VOI and per case
            'QUERY_DT_MAX_ICU': "select CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)) as id_ndaj1, \
                            id_measure_type, max(dt_insert) as max_dt \
                            from icu_sensor_24 \
                            where id_measure_type in {} \
                            and dt_cancel = '' \
                            group by CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)), id_measure_type \
                            order by CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)), id_measure_type",
            # Test if a given table exists
            'QUERY_TABLE_EXISTS': "select count(1) from {} limit1",
            # Drop table
            'QUERY_DROP_TABLE': "drop table {}",
            # Create table icu_nda_mouv_ufr_tr (movements related with an ICU unit)
            'CREATE_ICU_MOV': "create table icu_nda_mouv_ufr_tr as \
                                select n.* from nda_mouv_ufr_tr n, icu_ufr u \
                                where u.ids_ufr=n.ids_ufr",
            # Create table icu_sensor_24 (measures done during the first 24h in a ICU unit)
            'CREATE_ICU_SENSOR_24': "create table icu_sensor_24 as \
                                    select TO_DATE(n.dt_deb_mouv_ufr) as dt_deb, s.* from sensors s, \
                                    icu_nda_mouv_ufr_tr n \
                                    where cast(s.id_nda as VARCHAR)=n.id_nda and \
                                    s.dt_insert >= n.dt_deb_mouv_ufr and \
                                    s.dt_insert <= n.dt_deb_mouv_ufr + interval '1' DAY(2)",
            # Create table icu_pat_info
            'CREATE_ICU_PAT_INFO': "create table icu_pat_info as \
                                    select n.id_nda, p.age, \
                                    min(n.dt_deb_mouv_ufr) as min_dt_deb_mouv, \
                                    max(n.dt_deb_mouv_ufr) as max_dt_deb_mouv, \
                                    max(n.dt_fin_mouv_ufr) as max_dt_fin_mouv, \
                                    p.cd_sex_tr, p.dt_deces, p.dpt from icu_nda_mouv_ufr_tr n, patient_tr p \
                                    where p.ids_pat = n.ids_pat and \
                                    n.id_nda in (select distinct cast(id_nda as VARCHAR) from icu_sensor_24) \
                                    group by n.id_nda, p.age, p.cd_sex_tr, p.dt_deces, p.dpt",
            # Create table nda_j1_dt_range
            'CREATE_DT_RANGE': "select n.id_nda, n.j1, min(s.dt_insert) as dt_min, max(s.dt_insert) as dt_max \
                                from nda_j1_deces n, icu_sensor_util s \
                                where CAST(n.id_nda as VARCHAR) = CAST(s.id_nda as VARCHAR) \
                                and CAST(n.j1 as VARCHAR) = CAST(s.dt_deb as VARCHAR) \
                                group by n.id_nda, n.j1 \
                                order by n.id_nda, n.j1",
            # create table of useful sensors
            'CREATE_SENSOR_UTIL': "create table icu_sensor_util as \
                                    select * from icu_sensor_24 where dt_cancel = '' \
                                    and value_numeric > 0 \
                                    and CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),cast(TO_DATE(dt_deb) as VARCHAR)) in \
                                    (select distinct(CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'),j1)) from nda_j1_deces)",
            # select sensors related with a temporary table of case ids stored in table tmp_nda_j1_ids
            'QUERY_SELECT_SENSOR_BLOCK': "select lpad(cast(id_nda as VARCHAR),10,'0') as id_nda, CAST(dt_deb as VARCHAR) as j1, \
                        CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'), CAST(dt_deb as VARCHAR)) as id_case, \
                        id_measure_type, dt_insert, value_numeric \
                        from icu_sensor_util \
                        where CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'), CAST(dt_deb as VARCHAR)) in \
                        (select id_case from tmp_nda_j1_ids) \
                        order by id_nda, CAST(dt_deb as VARCHAR), id_measure_type, dt_insert",
            # select sensors related with a all cases (no batch)
            'QUERY_SELECT_SENSOR': "select lpad(cast(id_nda as VARCHAR),10,'0') as id_nda, CAST(dt_deb as VARCHAR) as j1, \
                        CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'), CAST(dt_deb as VARCHAR)) as id_case, \
                        id_measure_type, dt_insert, value_numeric \
                        from icu_sensor_util \
                        where CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'), CAST(dt_deb as VARCHAR)) in \
                        (select CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'), CAST(j1 as VARCHAR)) from nda_j1_deces) \
                        order by id_nda, CAST(dt_deb as VARCHAR), id_measure_type, dt_insert",
            # Store duration intervals and death date
            'QUERY_DT_RANGES': "select t1.id_nda, t1.j1, t1.dt_min, t1.dt_max, t2.dt_deces \
                                from nda_j1_dt_range t1, nda_j1_deces t2 \
                                where t1.id_nda = t2.id_nda and t1.j1 = t2.j1",
            # select all cases
            'QUERY_SELECT_ALL_CASES': "select distinct CONCAT(lpad(cast(id_nda as VARCHAR),10,'0'), CAST(j1 as VARCHAR)) \
                    from nda_j1_deces",
            # statistics for sensors kept
            'QUERY_SENSOR_UTIL_STATS': 'select t0.voi, count(t1.value_numeric) as number, \
                                min(t1.value_numeric) as minimum, max(t1.value_numeric) as maximum, \
                                avg(t1.value_numeric) as mean,variance(t1.value_numeric) as variance, \
                                stddev(t1.value_numeric) as std \
                                from ref_vois t0, icu_sensor_util t1 \
                                where t1.id_measure_type = t0.aphp_code \
                                group by t0.voi',
            # Mapping table between IGS2 and sensors
            'CREATE_MAP_IGS2_SENSORS': "create table map_igs2_sensor as \
                                        select distinct t1.id_nda, t1.j1, t2.id_ndaj1 \
                                        from dfs.prj.nda_j1_deces t1, dfs.prj.igs2_dataset t2 \
                                        where t1.id_nda = substr(t2.id_ndaj1,1,10) \
                                        and (TO_DATE(t1.j1, 'yyyy-MM-dd') + interval '15' DAY(2)) >= \
                                        TO_DATE(substr(t2.id_ndaj1,11,10), 'yyyy-MM-dd')",
            # Table nda_j1_deces with adjustements
            'CREATE_TRUE_DEATH': "create table true_death as \
                                  select t1.id_nda, t1.j1 from nda_j1_deces t1, nda_tr t2 \
                                  where t1.id_nda = lpad(cast(t2.id_nda as VARCHAR),10,'0') \
                                  and t1.dt_deces <> 'None' and t1.dt_deces is not NULL \
                                  and cast(t2.dt_deb_nda as TIMESTAMP) < cast(t1.dt_deces as TIMESTAMP) \
                                  and cast(t1.dt_deces as TIMESTAMP) <= cast(t2.dt_fin_nda as TIMESTAMP)",
            'CREATE_FINAL_NDA_J1': "create table final_nda_j1 as \
                                    select id_nda, j1, HR, RR, ABPS, ABPD, SPO2, dt_deces, dt_min, dt_max, dpt, \
                                    cd_sex_tr, stay_len \
                                    from nda_j1_deces \
                                    where concat(id_nda, cast(j1 as VARCHAR)) in \
                                    (select concat(id_nda, cast(j1 as VARCHAR)) from true_death) \
                                    union \
                                    select id_nda, j1, HR, RR, ABPS, ABPD, SPO2, 'None', dt_min, dt_max, dpt, \
                                    cd_sex_tr, stay_len \
                                    from nda_j1_deces \
                                    where concat(id_nda, cast(j1 as VARCHAR)) not in \
                                    (select concat(id_nda, cast(j1 as VARCHAR)) from true_death)"
        }

    def get_measures_counters(self):
        """
        Returns a pandas DataFrame holding number of measures by type
        :return: a pandas DataFrame
        """
        q = self._queries['QUERY_COUNT_CASES']
        return self._drill_conn.df_from_query(q)

    def get_total_cases(self):
        """
        Returns a pandas DataFrames giving the total number of cases holding at least one of the VOI
        :return: 1 row DataFrame
        """
        q = self._queries['QUERY_COUNT_CASES'].format(wd.np_array_2_string(self._dic.get_all_voi_codes(self._voi)))
        return self._drill_conn.df_from_query(q)

    def get_counter_matrix(self):
        """
        Returns a matrix of counters per case and per VOI
        :return: a DataFrame of row counters per case and VOI
        """
        q = self._queries['QUERY_GET_MEASURES'].format(wd.np_array_2_string(self._dic.get_all_voi_codes(self._voi)))
        grouped_df = self._drill_conn.df_from_query(q)
        return grouped_df.pivot(index='id_ndaj1', columns='id_measure_type', values='counter')

    def case_first_level_filter(self, counter_matrix):
        """"
        First level filter to apply to the counter matrix computed by get_counter_matrix
        We have to check to have at least one measure per case for each VOI except for BT (body temperature)
        If we do not have BT we can:
        * get it from the SAPSII form (it is included in SAPSII computation) if it exists
        * suppose it is normal i.e. ~ 37°C
        :param counter_matrix: matrix of counters per case and per VOI generated by get_counter_matrix
                            (pandas DataFrame)
        :return: filtered counter matrix
        """
        result = counter_matrix
        for v in self._voi:
            if v != 'BT':
                aphp_codes = self._dic.get_all_voi_codes([v])
                bcode = [False] * len(result)
                for code in aphp_codes:
                    bcode |= result[code].notnull()
                result = result[bcode]
        return result

    def case_second_level_filter(self, counter_matrix, min_measures=10):
        """
        Second level filter to apply to the counter matrix computed by get_counter_matrix and filtered
        by case_first_level_filter
        We have to check to have at least min_measures per case for each VOI except for BT (body temperature)
        :param counter_matrix: matrix of counters per case and per VOI generated by get_counter_matrix
                            (pandas DataFrame)
        :param min_measures: minimum number of measures we should have for each VOI to consider the case as
        enough relevant
        :return: filtered counter matrix
        """
        result = counter_matrix
        result = result.fillna(0)
        for v in self._voi:
            if v != 'BT':
                aphp_codes = self._dic.get_all_voi_codes([v])
                counter_v = pd.Series(np.zeros(len(result), dtype=int))
                counter_v.index = result.index
                for code in aphp_codes:
                    counter_v += result[code]
                result = result[counter_v > min_measures]
        return result

    def get_cases_data(self):
        """
        Build a DataFrame with available demographic data per case (age + death date)
        :return: pandas dataframe
        """
        q = self._queries['QUERY_CASE_DATA']
        result = self._drill_conn.df_from_query(q)
        result.index = result['id_ndaj1']
        # print(result.head())
        result.drop('id_ndaj1', axis=1, inplace=True)
        return result

    @staticmethod
    def case_more_age_filter(counter_matrix, demographic_df, age_min=18):
        """
        Filter counter matrix: remove cases having less than 18
        :param counter_matrix: matrix of counters per case and per VOI generated by get_counter_matrix
                            (pandas DataFrame)
        :param demographic_df: dataframe built with get_cases_data method
        :return: filtered counter matrix with related demographic data
        """
        # join the 2 matrix by id
        counter_matrix['id_ndaj1'] = counter_matrix.index
        counter_matrix.drop('id_ndaj1', axis=1, inplace=True)
        return pd.merge(counter_matrix, demographic_df[demographic_df['age'] >= age_min],
                        left_index=True, right_index=True,
                        how='inner')

    def get_dates_min(self):
        """
        Build a DataFrame with extrapolated minimum date per case
        For us the min date is the minimum date on which all selected VOI are available
        :return:
        """
        q = self._queries['QUERY_DT_MIN_ICU'].format(wd.np_array_2_string(self._dic.get_all_voi_codes(self._voi)))
        df_q = self._drill_conn.df_from_query(q)
        df_q = df_q.pivot(index='id_ndaj1', columns='id_measure_type', values='min_dt')
        for v in self._voi:
            if v != 'BT':
                aphp_codes = self._dic.get_all_voi_codes([v])
                sub_df_q = df_q[aphp_codes]
                df_q[v] = pd.to_datetime(sub_df_q.min(axis=1))
        return df_q[self._voi].max(axis=1)

    def get_dates_max(self):
        """
        Build a DataFrame with extrapolated maximum date per case
        For us the max date is the maximum date on which all selected VOI are available
        :return:
        """
        q = self._queries['QUERY_DT_MAX_ICU'].format(wd.np_array_2_string(self._dic.get_all_voi_codes(self._voi)))
        df_q = self._drill_conn.df_from_query(q)
        df_q = df_q.pivot(index='id_ndaj1', columns='id_measure_type', values='max_dt')
        for v in self._voi:
            if v != 'BT':
                aphp_codes = self._dic.get_all_voi_codes([v])
                sub_df_q = df_q[aphp_codes]
                df_q[v] = pd.to_datetime(sub_df_q.max(axis=1))
        return df_q[self._voi].min(axis=1)

    def get_case_stay_interval(self):
        """
        Estimate the stay date interval per case
        :return: counter matrix with added columns ==> dt_min, dt_max and stay interval (dt_max - dt_min)
        """
        df_dt_min = pd.DataFrame(self.get_dates_min())
        df_dt_max = pd.DataFrame(self.get_dates_max())
        df = pd.merge(df_dt_min, df_dt_max, left_index=True, right_index=True, how='inner')
        df.columns = ['dt_min', 'dt_max']
        df['stay_len'] = df['dt_max'] - df['dt_min']
        return df

    def case_filter_by_stay_interval(self, c_matrix, interval_min=6):
        """
        Filter the counter_matrix with the minimum value of stay interval and complete it with additional columns
        dt_min, dt_max, interval_stay
        :param c_matrix: matrix of counters per case and per VOI generated by get_counter_matrix
                            (pandas DataFrame)
        :param interval_min: minimum stay in hours
        :return:
        """
        case_stay_matrix = self.get_case_stay_interval()
        c_matrix = pd.merge(c_matrix, case_stay_matrix, left_index=True, right_index=True, how='inner')
        delta_min = timedelta(hours=interval_min)
        c_matrix = c_matrix[c_matrix['stay_len'] >= delta_min]
        return c_matrix

    def table_exists(self, table_name):
        """
        Check if a table exists or not
        :param table_name: table name
        :return: boolean ==> true if the table exists, else false
        """
        q = self._queries['QUERY_TABLE_EXISTS'].format(table_name)
        result = True
        try:
            self._drill_conn.df_from_query(q)
        except pyodbc.Error:
            result = False
        return result

    def drop_table(self, table_name):
        """
        Drop table
        :param table_name: table name
        :return: None
        """
        q = self._queries['QUERY_DROP_TABLE'].format(table_name)
        try:
            self._drill_conn.conn.execute(q)
        except pyodbc.Error:
            pass

    def create_icu_nda_mouv_ufr_tr(self):
        """
        Create table icu_nda_mouv_ufr_tr if it does not exist
        :return:
        """
        if not self.table_exists("icu_nda_mouv_ufr_tr"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_ICU_MOV'])
            except pyodbc.Error:
                pass

    def create_icu_sensor_24(self):
        """
        Create table icu_sensor_24 if it does not exist
        :return:
        """
        if not self.table_exists("icu_sensor_24"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_ICU_SENSOR_24'])
            except pyodbc.Error:
                pass

    def create_icu_nda(self):
        """
        Create table icu_nda if it does not exist
        :return:
        """
        if not self.table_exists("icu_nda"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_ICU_NDA'])
            except pyodbc.Error:
                pass

    def create_icu_pat_info(self):
        """
        Create table icu_pat_info if it does not exist
        :return:
        """
        if not self.table_exists("icu_pat_info"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_ICU_PAT_INFO'])
            except pyodbc.Error:
                pass

    def create_sensor_util(self):
        """
        Create table icu_sensor_util if it does not exist
        :return:
        """
        if not self.table_exists("icu_sensor_util"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_SENSOR_UTIL'])
            except pyodbc.Error:
                pass

    def create_nda_j1_dt_range(self):
        """
        Create table nda_j1_dt_range if it does not exist
        :return:
        """
        if not self.table_exists("nda_j1_dt_range"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_DT_RANGE'])
            except pyodbc.Error:
                pass

    def create_map_igs2_sensor(self):
        """
        Create table map_igs2_sensor if it does not exist
        :return:
        """
        if not self.table_exists("map_igs2_sensor"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_MAP_IGS2_SENSORS'])
            except pyodbc.Error:
                pass

    def get_dt_ranges(self):
        """
        get the features by case (age, date of death, dates min and max for measures)
        :return: a dataframe
        """
        q = "select * from nda_j1_deces"
        df_ranges = self._drill_conn.df_from_query(q)
        df_ranges['id_nda'] = df_ranges['id_nda'].astype(str)
        df_ranges['j1'] = df_ranges['j1'].astype(str)
        df_ranges['dt_min'] = pd.to_datetime(df_ranges['dt_min'])
        df_ranges['dt_max'] = pd.to_datetime(df_ranges['dt_max'])
        return df_ranges

    def get_filtered_counter_matrix(self, np_pts, min_stay):
        """
        Return the counter matrix regarding the minimum number of points required for all waves and regarding
        the minimum length of the stay in hours
        :param np_pts: minimum number of points required for each wave
        :param min_stay: minimum stay length in hours
        :return: counter matrix
        """
        c_matrix = self.get_counter_matrix()
        c_matrix = self.case_first_level_filter(c_matrix)
        c_matrix = self.case_second_level_filter(c_matrix, np_pts)
        demo_df = self.get_cases_data()
        c_matrix = self.case_more_age_filter(c_matrix, demo_df, age_min=15)
        c_matrix = self.case_filter_by_stay_interval(c_matrix, interval_min=min_stay)
        return c_matrix

    def get_sensor_util_stats(self, sigma_rule=3):
        """
        Return a dataframe with statistical values for each voi
        Compute thresholds by voi applying 1, 2 or 3 sigma rule
        :return: all vois stats dataframe
        """
        q = self._queries['QUERY_SENSOR_UTIL_STATS']
        df_stats = self._drill_conn.df_from_query(q)
        df_stats.index = df_stats.voi
        df_stats.drop('voi', axis=1, inplace=True)
        for v in df_stats.index:
            if v not in self._voi:
                df_stats.drop(v, axis=0, inplace=True)
            else:
                mean = df_stats.loc[v]['mean']
                # sigma rule
                min_limit = mean - sigma_rule*df_stats.loc[v]['std']
                max_limit = mean + sigma_rule*df_stats.loc[v]['std']
                self._sigmas_thres[v] = [min_limit, max_limit]
        return df_stats

    def nda_j1_deces_adjust(self):
        """
        Apply fix to field dt_deces (some patient deaths are related with other hospital stay)
        :return: None
        """
        self._drill_conn.conn.execute(self._queries['CREATE_TRUE_DEATH'])
        self._drill_conn.conn.execute(self._queries['CREATE_FINAL_NDA_J1'])
        self.drop_table("nda_j1_deces")
        self._drill_conn.conn.execute("create table nda_j1_deces as select * from final_nda_j1")
        self.drop_table("true_death")
        self.drop_table("final_nda_j1")


if __name__ == '__main__':
    dq = DrillQueries("drill_eds", ['HR', 'RR', 'ABPS', 'ABPD', 'SPO2'])

    # print(dq.get_total_cases())
    # counter_matrix = dq.get_counter_matrix()
    # print("Number of cases for first 24h: {}".format(len(counter_matrix)))
    # # print(counter_matrix.head(10))
    # counter_matrix = dq.case_first_level_filter(counter_matrix)
    # print("Number of cases for first 24h having values for {}: {}".format(dq._voi, len(counter_matrix)))
    # counter_matrix = dq.case_second_level_filter(counter_matrix, 6)
    # print("Number of cases for first 24h having at least 6 values for {}: {}".format(dq._voi, len(counter_matrix)))
    # demographic_df = dq.get_cases_data()
    # counter_matrix = dq.case_more_age_filter(counter_matrix, demographic_df, age_min=15)
    # # print(counter_matrix.head())
    # print("Number of cases for first 24h having at least 6 values for {} and more than 15 \
    #         years old: {}".format(dq.voi, len(counter_matrix)))
    # # mortality rate
    # mr = float(counter_matrix[counter_matrix['dt_deces'].notnull()].shape[0]) / float(counter_matrix.shape[0]) * 100.
    # print("Mortality rate: {}".format(mr))
    # counter_matrix = dq.case_filter_by_stay_interval(counter_matrix, interval_min=3)
    # print("Number of cases for first 24h having at least 6 values for {} and more than 15 \
    #         years old and stayed at least 3 hours: {}".format(dq.voi, len(counter_matrix)))
    # mr = float(counter_matrix[counter_matrix['dt_deces'].notnull()].shape[0]) / float(counter_matrix.shape[0]) * 100.
    # print("Mortality rate: {}".format(mr))
    # print(counter_matrix.head())

    # print(dq.table_exists('icu_sensor_24'))

    counter_matrix = dq.get_filtered_counter_matrix(96, 20)
    print(len(counter_matrix))
