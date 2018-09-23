# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:14:15 2018

@author: Suvhradip Ghosh
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets,linear_model
x=pd.read_csv("Temphumi.csv")
y=pd.read_csv("humi.csv")
regr = linear_model.LinearRegression()
plt.xlabel('temperature')
plt.ylabel("humidity")
regr.fit(x,y)
plt.plot(x, regr.predict(x), color='green',linewidth=3)
plt.plot(x,'o', color='red')
plt.plot(y,'o',color="blue")
plt.show()

