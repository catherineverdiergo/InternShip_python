# ----------------------------------------
# -*- coding: utf-8 -*-
# created by Catherine Verdier on 26/09/2016
# ----------------------------------------
import pandas as pd
import numpy as np

"""
   Create a global dictionary to rationalize measures types
   The aim of this Dictionary is to relate measures types codes to our variables
   of interest (Heart rate / Respiration rate / Arterial blood pressure ... and so on ...)
   The dictionary is implemented as a pandas DataFrame
"""


# The following list gives us the most frequent measures as found in the huge table of measures
# stored in the Orbis Information System

#  	    counter aphp_code 	label 	                                        UNIT_CD_SHORT
# 0 	246490 	10102 	    Fréquence cardiaque 	                        Puls/min
# 1 	243159 	10168 	    Saturation pulsée Oxygène 	                    %
# 2 	237342 	10120 	    Frequence respiratoire 	                        /min
# 3 	188228 	10121 	    Mode Ventilatoire 	                            Sans dimension
# 4 	146710 	11 	        Pression artérielle non invasive systolique 	mmHg
# 5 	145689 	12 	        Pression artérielle non invasive diastolique 	mmHg
# 6 	142949 	13 	        Pression artérielle non invasive moyenne 	    mmHg
# 7 	140900 	10281 	    MEGS PAS Intervalle 2h 	                        mmHg
# 8 	140433 	10282 	    MEGS PAM intervalle 2h 	                        mmHg
# 9 	140124 	10283 	    MEGS PAD intervalle 2h 	                        mmHg
# 10 	98847 	16 	        Pression artérielle moyenne 	                mmHg
# 11 	98680 	15 	        Pression artérielle diastolique 	            mmHg
# 12 	98658 	14 	        Pression artérielle systolique 	                mmHg
# 13 	84168 	10044 	    Fraction inspirée en oxygène 	                %
# 14 	82303 	10171 	    Volume courant 	                                ml
# 15 	82022 	10173 	    Volume minute 	                                l/min
# 16 	81633 	10174 	    Pic de pression respiratoire 	                mbar
# 17 	80176 	10170 	    Pression expiration positive 	                mbar
# 18 	56072 	10172 	    Pression de plateau 	                        mbar
# 19 	32218 	10280 	    MEGS FC intervalle 2h 	                        Puls/min
# 20 	32085 	10291 	    MEGS SPO2 intervalle 2h 	                    %
# 21 	31401 	10285 	    MEGS FR intervalle 2h 	                        /min
# 22 	25940 	10169 	    Niveau d'aide inspiratoire 	                    mbar
# 23 	24528 	10294 	    MEGS MODE intervalle 2h 	                    Sans dimension
# 24 	23751 	4 	        Température corporelle 	                        °C
# 25 	14698 	10153 	    Diurèse Miction 	                            ml
# 26 	11828 	10286 	    MEGS FIO2 intervalle 2h 	                    %
# 27 	11687 	10293 	    MEGS VMIN intervalle 2h 	                    l/min
# 28 	11661 	10296 	    MEGS PPEAK intervalle 2h 	                    mbar
# 29 	11656 	10292 	    MEGS VT intervalle 2h 	                        ml


class WavesDic:
    """
    Class to manage variable of interest regarding the aphp codes registered in database
    """

    def __init__(self):
        """"
        Constructor : initialize the default dictionary contents
        """
        self._default = pd.DataFrame(columns=('code', 'label', 'aphp_codes'))
        self._default.loc[0] = ['HR', 'Heart rate', [10102, 10280]]
        self._default.loc[1] = ['RR', 'Respiration rate', [10120, 10285]]
        self._default.loc[2] = ['ABPS', 'Systolic arterial blood pressure', [11, 10281, 14]]
        self._default.loc[3] = ['ABPD', 'Diastolic arterial blood pressure', [12, 10283, 15]]
        self._default.loc[4] = ['SPO2', 'Blood oxygen saturation', [10168, 10291]]
        self._default.loc[5] = ['BT', 'Body temperature', [4]]

    def get_all_voi_codes(self, voi):
        """
        Get the complete list of aphp_code for a given list of voi (variables of interest)
        :param voi: list of variables of interest (list of related string codes)
        :return: a list of aphp codes (list of int)
        """
        df = self._default
        bcode = [False]*len(df)
        for v in voi:
            bcode |= df['code'] == v
        df = df[bcode]
        aphp_codes = df['aphp_codes']
        aphp_codes = aphp_codes.apply(lambda x: np.array(x))
        result = None
        if len(aphp_codes) > 0:
            result = aphp_codes.iloc[0]
            for i in range(1, len(aphp_codes)):
                result = np.union1d(result, aphp_codes.iloc[i])
        return result

    def get_label(self, metric):
        """
        Retrieve the label associated with a metric code
        :param metric: metric code
        :return: label (string)
        """
        return self._default[self._default['code'] == metric]['label'].iloc[0]


def np_array_2_string(arr):
    """
    Convert an np array of int to a string list with parenthesis
    :param arr: ndarray of int values
    :return: a string
    """
    result = "("
    for i in range(np.size(arr)):
        if i > 0:
            result += ","
        result += str(arr[i])
    result += ")"
    return result

if __name__ == '__main__':
    import drill_utilities as du

    conn = du.DrillODBC("drill_eds")
    wd = WavesDic()

    aphp_code_restriction = np_array_2_string(wd.get_all_voi_codes(['SPO2', 'ABPD', 'ABPS', 'HR', 'RR', 'BT']))

    q = "select count(distinct s.id_nda) \
        from icu_sensor_24 s \
        where s.id_measure_type in {} and s.dt_cancel = ''".format(aphp_code_restriction)

    print(conn.df_from_query(q))

    print(wd.get_label('HR'))
