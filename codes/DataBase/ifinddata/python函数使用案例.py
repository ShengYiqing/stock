# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:05:13 2018
@author: zhangsuowei
"""

from iFinDPy import *
import pandas as pd
import numpy as np
import Global_Config as gc

username = gc.USERNAME_IFIND
password = gc.PASSWORD_IFIND
thsLogin = THS_iFinDLogin(username, password)

thsData = THS_HQ('CBA00621.CB', 'close', '', '2023-01-10', '2024-01-10')
