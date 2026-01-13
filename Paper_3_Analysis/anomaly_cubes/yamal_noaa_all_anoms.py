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

lst_2001 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2001/AVN16/LST_AVN16_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2002 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2002/AVN16/LST_AVN16_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2003 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2003/AVN16/LST_AVN16_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2004 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2004/AVN16/LST_AVN16_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2005 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2005/AVN16/LST_AVN16_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2006 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2006/AVN18/LST_AVN18_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2007 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2007/AVN18/LST_AVN18_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2008 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2008/AVN18/LST_AVN18_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2009 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2009/AVN18/LST_AVN18_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2010 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2010/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2011 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2011/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2012 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2012/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2013 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2013/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2014 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2014/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2015 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2015/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2016 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2016/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2017 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2017/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')

lst_2018 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2018/AVN19/LST_AVN19_All_Arctic__v.11.0__*DAY.nc', engine='netcdf4')


lc_interp = lc.interp(lat=lst_2018.coords['lat'], lon=lst_2018.coords['lon'], method='nearest')

# load the time series somewhere ?
results = {}

for year in range(1981, 2019):

    ds = globals()[f"lst_{year}"]   # load lst_1981, lst_1982, ...

    # 2. Fixed 10-day bins anchored at Jan 1 every year
    
    ds = ds.where(lc_interp['lccs_class'] != 210)
    print(ds)
    clean_LST = ds['LST'].where(ds['LST'] != 110, np.nan)
    max_10d = (
        clean_LST
        .resample(
            time='1MS'
        )
        .mean()
    )

    # Ensure bins exist even with no data
    # Xarray automatically creates them and fills with NaN

    results[year] = max_10d

combined = xr.concat([results[y] for y in range(1981, 2019)], dim="time")

climatology = combined.groupby('time.month').mean("time")

anomalies = combined.groupby('time.month') - climatology

clim_std = combined.groupby('time.month').std("time")

stand_anomalies = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    combined.groupby("time.month"),
    climatology,
    clim_std,
    dask="parallelized"
)

standard_anoms = stand_anomalies.compute()

standard_anoms.to_netcdf('standard_anomalies_yamal_day_monthly_mean.nc')

standard_anoms.sel(lat=slice(55,72), lon=slice(-168,-150)).plot.hist(bins=50)
plt.savefig('yamal_day_anomalies_bins_hist_monthly_mean.png')
above_1 = (standard_anoms > 1).sum(dim="time")
below_minus1 = (standard_anoms < -1).sum(dim="time")