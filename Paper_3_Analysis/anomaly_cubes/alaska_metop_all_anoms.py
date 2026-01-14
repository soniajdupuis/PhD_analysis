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

lc = land_cover.sel(lat=slice(72,55), lon=slice(-168,-150))

lst_2007 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2007/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2008 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2008/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2009 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2009/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2010 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2010/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2011 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2011/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2012 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2012/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2013 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2013/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2014 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2014/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2015 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2015/AVMEA/LST_AVMEA_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2016 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2016/AVMEB/LST_AVMEB_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2017 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2017/AVMEB/LST_AVMEB_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2018 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2018/AVMEB/LST_AVMEB_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2019 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2019/AVMEB/LST_AVMEB_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2020 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2020/AVMEB/LST_AVMEB_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2021 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2021/AVMEB/LST_AVMEB_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2022 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2022/AVMEB/LST_AVMEB_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')
lst_2023 = xr.open_mfdataset('/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic/2023/AVMEB/LST_AVMEB_All_Arctic__v.11.0__*NIGHT.nc', engine='netcdf4')


lc_interp = lc.interp(lat=lst_2018.coords['lat'], lon=lst_2018.coords['lon'], method='nearest')

# load the time series somewhere ?
results = {}

for year in range(2007, 2024):

    ds = globals()[f"lst_{year}"]   # load lst_1981, lst_1982, ...

    # 2. Fixed 10-NIGHT bins anchored at Jan 1 every year
    
    ds = ds.where(lc_interp['lccs_class'] != 210)
    print(ds)
    clean_LST = ds['LST'].where(ds['LST'] != 110, np.nan)
    max_10d = (
        clean_LST
        .resample(
            time='1MS'
        )
        .max()
    )

    # Ensure bins exist even with no data
    # Xarray automatically creates them and fills with NaN

    results[year] = max_10d

combined = xr.concat([results[y] for y in range(2007, 2024)], dim="time")

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

standard_anoms.to_netcdf('standard_anomalies_alaska_night_metop.nc')

standard_anoms.sel(lat=slice(55,72), lon=slice(-168,-150)).plot.hist(bins=50)
plt.savefig('alaska_night_anomalies_bins_hist_metop.png')
above_1 = (standard_anoms > 1).sum(dim="time")
below_minus1 = (standard_anoms < -1).sum(dim="time")