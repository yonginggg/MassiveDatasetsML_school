#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/9 10:26
# @Author  : YangYusheng
# @File    : readFile.py
# @Software: PyCharm

# import pandas as pd
# df = pd.read_csv('D:/Work/数据挖掘ML/Logistic回归/telco.csv')
#导入数据分析包
import pandas as pd
import numpy as np
#导入csv数据
#dtype = str,最好读取的时候都以字符串的形式读入，不然可能会使数据失真
#比如一个0010008的编号可能会读取成10008

fileNameStr = 'D:/Work/数据挖掘ML/Logistic回归/telco.csv'

# encoding = "ISO-8859-1" -- 用什么解码，一般会默认系统的编码，如果是中文就用 "utf-8"
DataDF = pd.read_csv(fileNameStr,encoding = "utf-8")
DataDF = DataDF.fillna(method='ffill')
DataDF = DataDF.fillna(method='bfill')
DataDF=DataDF.values
# print(DataDF.shape)
DataX = DataDF[:,:-1]
DataY = DataDF[:,-1]
# print(DataX.shape)