# ----------------------------------------
# -*- coding: utf-8 -*-
# created by Catherine Verdier on 29/09/2016
# ----------------------------------------


# spark_home = os.environ.get('SPARK_HOME', "C:\spark-1.6.1-bin-hadoop2.6")
# os.environ['SPARK_HOME'] = spark_home
# os.environ['PYSPARK_DRIVER_PYTHON'] ="python"
# # os.environ['PYSPARK_DRIVER_PYTHON_OPTS']="notebook"
# sys.path.insert(0, os.path.join(spark_home, 'python'))
# sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.9-src.zip'))

import py4j
import pyspark
from pyspark.sql import HiveContext
from pyspark.context import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import pandas as pd
import os
import sys


class SparkClient:

    def __init__(self, spark_home, spark_master="local", exec_memory="8g",
                 app_name="SparkClient"):
        """
        Initialize sparkcontext, sqlcontext
        :param spark_master: target spark master
        :param exec_memory: size of memory per executor
        """
        self._spark_master = spark_master
        self._exec_memory = exec_memory
        self._app_name = app_name
        self._spark_home = spark_home
        # Path for spark source folder
        os.environ['SPARK_HOME'] = self._spark_home
        self._spark_url = spark_master
        if spark_master != "local":
            os.environ['SPARK_MASTER_IP'] = spark_master
            self._spark_url = "spark://"+self._spark_master+":7077"
        # Append pyspark  to Python Path
        sys.path.append(self._spark_home)
        # define the spark configuration
        conf = (SparkConf()
                .setMaster(self._spark_url)
                .setAppName(self._app_name)
                .set("spark.executor.memory", self._exec_memory)
                .set("spark.core.connection.ack.wait.timeout", "600")
                .set("spark.akka.frameSize", "512")
                .set("spark.cassandra.output.batch.size.bytes", "131072")
                )
        # create spark context
        self._spark_ctx = None
        if SparkContext._active_spark_context is None:
            self._spark_ctx = SparkContext(conf=conf)
        # create spark-on-hive context
        self._sql = SQLContext(self._spark_ctx)

    def close(self):
        """"
        Close the spark context
        """
        self._spark_ctx.stop()

    @property
    def sc(self):
        return self._spark_ctx

    def save_nda_j1_deces_from_df(self, df, file_dir, vois):
        """
        Save dataframe as Parquet file for nda_j1_deces table
        :param df: source dataframe
        :param file_dir: path to database Parquet files
        :return: None
        """
        # Transform result pandas DataFrame to Spark DataFrame
        pd.options.mode.chained_assignment = None  # to avoid pandas warnings

        #df_src_pd = df['id_ndaj1'].str.extract('(^[0-9]{7})([0-9]{4}-[0-9]{2}-[0-9]{2})', expand=False)
        df_src_pd = df['id_ndaj1'].str.extract('(^[0-9]{10})([0-9]{4}-[0-9]{2}-[0-9]{2})')
        df_src_pd.columns = ['id_nda', 'j1']
        for voi in vois:
            df_src_pd[voi] = df[voi]
        df_src_pd['dt_deces'] = df.dt_deces.apply(str)
        df_src_pd['dt_min'] = df.dt_min.apply(str)
        df_src_pd['dt_max'] = df.dt_max.apply(str)
        df_src_pd['dpt'] = df.dpt
        df_src_pd['cd_sex_tr'] = df.cd_sex_tr
        df_src_pd['stay_len'] = df.stay_len.apply(str)
        spark_df = self._sql.createDataFrame(df_src_pd)
        spark_df.write.parquet(file_dir+"/nda_j1_deces", mode='overwrite')


if __name__ == "__main__":
    mysc = SparkClient("C:\spark-1.6.1-bin-hadoop2.6",
                       spark_master="bbs-ocosrv-q001.bbs.aphp.fr",
                       exec_memory="2g")
    print(mysc.sc.version)
    from pyspark.sql.functions import concat, col, lit
    # dfs_ids = mysc._sql.read.parquet("hdfs://bbs-ocosrv-q001.bbs.aphp.fr:7222/user/mapr/eds/nda_j1_deces") \
    # dfs_ids = mysc._sql.read.parquet("hdfs:///user/mapr/eds/nda_j1_deces") \
    #             .select(concat('id_nda', 'j1').alias('id_case')).distinct()
    # dfs_ids.limit(10).show()
    words = mysc.sc.parallelize(["scala", "java", "hadoop", "spark", "akka"])
    print words.count()
    mysc.close()
