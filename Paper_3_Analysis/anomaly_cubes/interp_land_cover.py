import xarray as xr
# load landcover and anette's landcover! (first normal landcover)
land_cover = xr.open_dataset('/mnt/data7/nfs4/avh_ndvi/sdupuis/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc', engine='netcdf4')

yearly = xr.open_dataset('output_data/standard_anomalies_yamal_day_yearly_2.nc')

lc_interp = land_cover.interp(lat=yearly.coords['lat'], lon=yearly.coords['lon'], method='nearest')

lc_interp.to_netcdf('lc_downscaled_2020.nc')