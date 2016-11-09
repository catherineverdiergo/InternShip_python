# ----------------------------------------
# -*- coding: utf-8 -*-
# created by Catherine Verdier on 26/09/2016
# ----------------------------------------
import pyodbc
import pandas as pd


class DrillODBC:
    """
    Class to connect and query throughout a Drill ODBC connection
    """

    def __init__(self, dsn):
        """
        Constructor : initialize the ODBC connection
        :param dsn: target ODBC Data Source Name (string)
        """
        #
        self._conn = pyodbc.connect("DSN=" + dsn, autocommit=True)

    def df_from_query(self, query):
        """
        Query an ODBC datasource and set results in a pandas Dataframe

        Parameters
        ----------
        pyodbc.Connection conn : Datasource ODBC connection
        string           query : query to execute on datasource
        """
        c = self._conn.cursor()
        c.execute(query)
        cols = [column[0] for column in c.description]
        data = []
        for row in c.fetchall():
            data.append(tuple(row))
        df = pd.DataFrame(data, columns=cols)
        return df

    @property
    def conn(self):
        return self._conn
