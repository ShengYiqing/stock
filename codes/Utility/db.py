# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:51:46 2021

@author: admin
"""

import os
import sys
import datetime
import pandas as pd
from pandas import Series, DataFrame

import Global_Config as gc
import tools
import sqlalchemy as sa
from sqlalchemy import create_engine

if __name__ == '__main__':
    
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    
    sql = """
    delete FROM intermediate.tdailyhffactor 
    where factor_name in ('pvcorr', 'rvcorr')
    """
    with engine.connect() as con:
        con.execute(sql)