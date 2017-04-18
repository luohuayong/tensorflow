# coding:utf-8

import tushare as ts
import datetime
import numpy as np
import pandas as pd

ts2014 = ts.get_k_data('000002', '2014-01-01', '2014-12-31')
ts2015 = ts.get_k_data('000002', '2015-01-01', '2015-12-31')
ts2016 = ts.get_k_data('000002', '2016-01-01', '2016-12-31')
ts_all = pd.concat([ts2014, ts2015, ts2016])

# start_row = 0
# end_row = 100
# np_temp = np.array(ts_all.iloc[start_row:end_row, 1:6]).ravel()
# data = np.array(np_temp)
# np_temp = np.array(ts_all.iloc[101, 1:6])
# result = np.array(np_temp)

data = None
result = None
for i in range(10):
    start_row = 0 + i
    end_row = 100 + i
    np_temp = np.array(ts_all.iloc[start_row:end_row, 1:6]).ravel()
    if i != 0:
        data = np.row_stack((data, np_temp))
    else:
        data = np.array(np_temp)
    np_temp = np.array(ts_all.iloc[101+i, 1:6])
    if i != 0:
        result = np.row_stack((result, np_temp))
    else:
        result = np.array(np_temp)

print data
print '========='
print result

# np_temp = np.array(ts2014.iloc[:, :6])
# np_data = np.array(np_temp)
# np_temp = np.array(ts2015.iloc[:, :6])
# np_data = np.row_stack((np_data, np_temp))
# np_temp = np.array(ts2016.iloc[:, :6])
# np_data = np.row_stack((np_data, np_temp))

# for i in range(10):


# date = datetime.datetime.strptime('2016-5-1', '%Y-%m-%d')
# end = date - datetime.timedelta(days=1)
# start = end - datetime.timedelta(days=365)
# start = end - datetime.timedelta(days=365)


# str_end = end.strftime('%Y-%m-%d')
# str_start = start.strftime('%Y-%m-%d')

# data = ts.get_k_data('000002', str_start, str_end)

# print data
# data.iloc[-10:, 1:6]
# # print np.array(data.iloc[-10:, 1:6])
# data1 = np.array(data.iloc[-10:, 1:6])
# data1 = data1.ravel()
# # print data1.ravel()
# data2 = np.array(data1)
# for i in range(10):
#     date = date + datetime.timedelta(days=1)
#     end = date - datetime.timedelta(days=1)
#     start = end - datetime.timedelta(days=365)
#     str_end = end.strftime('%Y-%m-%d')
#     str_start = start.strftime('%Y-%m-%d')
#     data = ts.get_k_data('000002', str_start, str_end)
#     data1 = np.array(data.iloc[-10:, 1:6])
#     data1 = data1.ravel()
#     data2 = np.row_stack((data2, data1))
# print data2