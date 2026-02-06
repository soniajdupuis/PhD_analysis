import xarray as xr


anoms_swe = xr.open_dataset('output_data/standard_anoms_swe.nc')

std_anoms_day = xr.open_dataset('../anomaly_cubes/output_data/standard_anomalies_Greenland_day.nc')

anoms_swe_interp = anoms_swe.interp(lat=std_anoms_day.coords['lat'], lon=std_anoms_day.coords['lon'], method='nearest')

da1, da2 = xr.align(anoms_swe_interp, std_anoms_day, join="inner")
da1 = da1.chunk({"time": -1, "lat": 100, "lon": 100})
da2 = da2.chunk({"time": -1, "lat": 100, "lon": 100})

corr_map = xr.corr(da1['swe'], da2['LST'], dim="time")
corr_map = corr_map.compute()

corr_map.to_netcdf('corr_map_swe.nc')