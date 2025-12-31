import xarray as xr
import datetime
import pymannkendall as mk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes
from dask.diagnostics import ProgressBar



# load landcover and anette's landcover! (first normal landcover)
land_cover = xr.open_dataset('/mnt/data7/nfs4/avh_ndvi/sdupuis/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7b.nc', engine='netcdf4')

lc = land_cover.sel(lat=slice(74,60), lon=slice(65,74))

# read per year and then concact!
lst_1981 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1981/AVN07/LST_AVN07_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1982 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1982/AVN07/LST_AVN07_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1983 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1983/AVN07/LST_AVN07_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1984 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1984/AVN07/LST_AVN07_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1985 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1985/AVN07/LST_AVN07_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1985_09 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1985/AVN09/LST_AVN09_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1985 =xr.concat([lst_1985, lst_1985_09], dim='time')

lst_1986 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1986/AVN09/LST_AVN09_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1987 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1987/AVN09/LST_AVN09_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1988 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1988/AVN09/LST_AVN09_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1989 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1989/AVN11/LST_AVN11_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1990 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1990/AVN11/LST_AVN11_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1991 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1991/AVN11/LST_AVN11_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1992 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1992/AVN11/LST_AVN11_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1993 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1993/AVN11/LST_AVN11_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1994 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1994/AVN11/LST_AVN11_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1995 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1995/AVN14/LST_AVN14_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1996 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1996/AVN14/LST_AVN14_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1997 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1997/AVN14/LST_AVN14_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1998 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1998/AVN14/LST_AVN14_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_1999 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/1999/AVN14/LST_AVN14_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2000 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2000/AVN14/LST_AVN14_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2001 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2001/AVN14/LST_AVN14_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')



lc_interp = lc.interp(lat=lst_2001.coords['lat'], lon=lst_2001.coords['lon'], method='nearest')

def semimonth_time(time):
    year = time.dt.year
    month = time.dt.month

    day = xr.where(time.dt.day <= 15, 1, 16)
    #print(day)

    ymd = year * 10000 + month * 100 + day

    return xr.apply_ufunc(
        pd.to_datetime,
        ymd,
        kwargs={"format": "%Y%m%d"},
        vectorize=True,
        output_dtypes=["datetime64[ns]"],
    )

results = []

for year in range(1981, 2002):

    ds = globals()[f"lst_{year}"]

    # --------------------------------------------------
    # 2. Mask unwanted landcover
    # --------------------------------------------------
    ds = ds.where(lc_interp['lccs_class'] != 210)

    # --------------------------------------------------
    # 3. Remove clouds (flag = 110)
    # --------------------------------------------------
    clean_LST = ds['LST'].where(ds['LST'] > 110)

    # --------------------------------------------------
    # 4. Create semi-monthly bins
    # --------------------------------------------------
    clean_LST = clean_LST.assign_coords(
        semimonth_time=semimonth_time(clean_LST.time)
    )

    # --------------------------------------------------
    # 5. Aggregate (MAX per semi-month)
    # --------------------------------------------------
    semi_month_max = (
        clean_LST
        .groupby("semimonth_time")
        .max()
        .rename({"semimonth_time": "time"})
    )

    results.append(semi_month_max)

lst_semimonth = xr.concat(results, dim="time").sortby("time")

# by-monthly seems ok


month = lst_semimonth.time.dt.month
day   = lst_semimonth.time.dt.day

# 0 = first half (01), 1 = second half (16)
half = xr.where(day == 1, 0, 1)

bin_index = (month - 1) * 2 + half + 1

lst_semimonth = lst_semimonth.assign_coords(bin=bin_index)

climatology = lst_semimonth.groupby("bin").mean("time")
print(climatology)
# (lat=slice(74,60), lon=slice(65,74))
climatology.sel(lat=slice(60, 74), lon=slice(65,74)).isel(bin=0).plot();
plt.savefig('yamal_climatology_bin0.png')

anomalies = lst_semimonth.groupby("bin") - climatology

anomalies.sel(lat=slice(60, 74), lon=slice(65,74), time=slice('1982-05','1982-08')).plot(x="lon", y="lat", col="time", col_wrap=5, vmin=-10, vmax=10);
plt.savefig('anomalies.png')

anoms = anomalies.sel(lat=slice(60, 74), lon=slice(65,74))
anoms = anoms.chunk(dict(time=-1))   # <<< REQUIRED

time_numeric = (
    anoms.time.dt.year +
    (anoms.time.dt.dayofyear - 1) / 365.0
)

def theil_sen_1d(y, x):
    # remove NaNs
    mask = np.isfinite(y) & np.isfinite(x)
    
    # not enough data points → return NaN
    if mask.sum() < 5:
        return np.nan
    
    slope, intercept, lower, upper = theilslopes(y[mask], x[mask])
    return slope


slope = xr.apply_ufunc(
    theil_sen_1d,
    anoms,
    time_numeric,
    input_core_dims=[["time"], ["time"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float],
)

trend_per_decade = slope * 1

with ProgressBar():
    res = trend_per_decade.compute()


def mk_pvalue(y):
    y = y[np.isfinite(y)]
    if y.size < 5:
        return np.nan
    return mk.original_test(y).p

mk_p = xr.apply_ufunc(
    mk_pvalue,
    anoms,
    input_core_dims=[["time"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float],
)


with ProgressBar():
    p_val = mk_p.compute()


significant_trend = res.where(p_val < 0.05)

significant_trend.to_netcdf('yamal_significant_trends_1981_2001.nc')