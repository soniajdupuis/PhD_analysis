import xarray as xr
import numpy as np

swe = xr.open_mfdataset(f'/mnt/data7/nfs4/avh_ndvi/sdupuis/swe/MERGED/v3.1/*/*/*-ESACCI-*.nc', engine='netcdf4')

swe_mo = swe['swe'].resample(time='1MS').mean()
swe_mo_arctic = swe_mo.sel(lat=slice(50,85))

snow_data = f'/mnt/data6/nfs4/cci_snow_1/products/production_40_20250217/*/netcdf/scfg/*-*NOAA07-*.nc'
snow_noaa07 = xr.open_mfdataset(snow_data, chunks='auto', engine='netcdf4')

snow_noaa09 = xr.open_mfdataset(f'/mnt/data6/nfs4/cci_snow_1/products/production_40_20250217/*/netcdf/scfg/*-*NOAA09-*.nc', chunks='auto', engine='netcdf4')

snow_noaa09 = snow_noaa09.sel(time=slice('1985', '1988-10-31'))

snow_noaa11 = xr.open_mfdataset(f'/mnt/data6/nfs4/cci_snow_1/products/production_40_20250217/*/netcdf/scfg/*-*NOAA11-*.nc', chunks='auto', engine='netcdf4')

snow_noaa14 = xr.open_mfdataset(f'/mnt/data6/nfs4/cci_snow_1/products/production_40_20250217/*/netcdf/scfg/*-*NOAA14-*.nc', chunks='auto', engine='netcdf4')

snow_noaa14 = snow_noaa14.sel(time=slice('1995', '2000'))

snow_noaa16 = xr.open_mfdataset(f'/mnt/data6/nfs4/cci_snow_1/products/production_40_20250217/*/netcdf/scfg/*-*NOAA16-*.nc', chunks='auto', engine='netcdf4')

snow_noaa16 = snow_noaa16.sel(time=slice('2001', '2005'))

snow_noaa18 = xr.open_mfdataset(f'/mnt/data6/nfs4/cci_snow_1/products/production_40_20250217/*/netcdf/scfg/*-*NOAA18-*.nc', chunks='auto', engine='netcdf4')
snow_noaa18 = snow_noaa18.sel(time=slice('2006', '2009'))

snow_noaa19 = xr.open_mfdataset(f'/mnt/data6/nfs4/cci_snow_1/products/production_40_20250217/*/netcdf/scfg/*-*NOAA19-*.nc', chunks='auto', engine='netcdf4')

snow_noaa19 = snow_noaa19.sel(time=slice('2010', '2018'))

results = {}

for nb in ['07','09', '11', '14', '16', '18', '19']:

    ds = globals()[f"snow_noaa"+nb]   



    #print(ds)
    clean_snow = ds['scfg'].where(ds['scfg'] < 101, np.nan)

    max_10d = (
        clean_snow
        .resample(
            time='1MS'
        )
        .mean()
    )

    # Ensure bins exist even with no data
    # Xarray automatically creates them and fills with NaN

    results[nb] = max_10d

combined = xr.concat([results[y] for y in ['07','09', '11', '14', '16', '18', '19']], dim="time")

combined_arctic = combined.sel(lat=slice(50,85))

# compute snow anomalies
climatology = combined_arctic.groupby('time.month').mean("time")

anomalies = combined_arctic.groupby('time.month') - climatology

clim_std = combined_arctic.groupby('time.month').std("time")


stand_anomalies = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    combined_arctic.groupby("time.month"),
    climatology,
    clim_std,
    dask="parallelized"
)

standard_anoms_scfg = stand_anomalies.compute()

standard_anoms_scfg.to_netcdf('output_data/standard_anoms_scfg.nc')



# swe anoms
climatology_swe = swe_mo_arctic.groupby('time.month').mean("time")

anomalies_swe = swe_mo_arctic.groupby('time.month') - climatology

clim_std_swe = swe_mo_arctic.groupby('time.month').std("time")


stand_anomalies_swe = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    swe_mo_arctic.groupby("time.month"),
    climatology_swe,
    clim_std_swe,
    dask="parallelized"
)

standard_anoms_swe = stand_anomalies_swe.compute()

standard_anoms_swe.to_netcdf('output_data/standard_anoms_swe.nc')
