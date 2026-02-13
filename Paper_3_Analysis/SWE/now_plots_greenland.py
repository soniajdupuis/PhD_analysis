import xarray as xr
import matplotlib.transforms as mtransforms
from cmcrameri import cm
from matplotlib import rc
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# load landcover and anette's landcover! (first normal landcover)
land_cover = xr.open_dataset('/mnt/data7/nfs4/avh_ndvi/sdupuis/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7b.nc', engine='netcdf4')

lc = land_cover.sel(lat=slice(84,58), lon=slice(-73,-7))
std_anoms_day_al = xr.open_dataset('../anomaly_cubes/output_data/standard_anomalies_alaska_day.nc')
lc_interp = lc.interp(lat=std_anoms_day_al.coords['lat'], lon=std_anoms_day_al.coords['lon'], method='nearest')
subset = std_anoms_day_al.sel(lat=slice(58, 84), lon=slice(-73,-7))

anoms_scfg = xr.open_dataset('output_data/standard_anoms_scfg.nc')
anoms_swe = xr.open_dataset('output_data/standard_anoms_swe.nc')
subset_swe = anoms_swe['swe'].sel(lat=slice(58, 84), lon=slice(-73,-7), time=slice('1981-08', '2018'))

subset_snow = anoms_scfg['scfg'].sel(lat=slice(58, 84), lon=slice(-73,-7))

#interp swe
anoms_swe_interp = subset_swe.interp(lat=subset.coords['lat'], lon=subset.coords['lon'], method='nearest')
anoms_swe_interp_aligned, da2 = xr.align(anoms_swe_interp, subset, join="inner")
swe_large_anoms = anoms_swe_interp_aligned.where((subset['LST'] > 1.0) & (subset['LST'] < 3.0))
swe_pos = swe_large_anoms.where(swe_large_anoms > 1)
swe_neg = swe_large_anoms.where(swe_large_anoms < -1)

swe_neg_season_count = (
    swe_neg
    .groupby("time.season")
    .count(dim="time")
).where(lc_interp['lccs_class'] != 210)
swe_pos_season_count = (
    swe_pos
    .groupby("time.season")
    .count(dim="time")
).where(lc_interp['lccs_class'] != 210)

snow_large_anoms = subset_snow.where((subset['LST'] > 1.0) & (subset['LST'] < 3.0))

snow_pos = snow_large_anoms.where(snow_large_anoms > 1)

snow_neg = snow_large_anoms.where(snow_large_anoms < -1)

snow_neg_season_count = (
    snow_neg
    .groupby("time.season")
    .count(dim="time")
).where(lc_interp['lccs_class'] != 210)

snow_pos_season_count = (
    snow_pos
    .groupby("time.season")
    .count(dim="time")
).where(lc_interp['lccs_class'] != 210)
print(snow_pos_season_count)

vmax = max(
    snow_neg_season_count.max(),
    snow_pos_season_count.max()
).item()




seasons = ["DJF", "MAM", "JJA", "SON"]

fig, axes = plt.subplots(
    nrows=2, ncols=4,
    figsize=(20, 8),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)
#axes.set_autoscale_on(False)

# --- Row 1: negative snow anomalies ---
for i, season in enumerate(seasons):
    axes[0,i].set_autoscale_on(False)
    swe_neg_season_count.sel(season=season).plot(
        ax=axes[0, i],
        transform=ccrs.PlateCarree(),
        add_colorbar=False, vmin=1, vmax=vmax, cmap=cm.lipari_r, levels=15
    )
    axes[0, i].set_title(f"Negative snow anomalies – {season}")
    axes[0, i].coastlines()

# --- Row 2: positive snow anomalies ---
for i, season in enumerate(seasons):
    axes[1,i].set_autoscale_on(False)
    swe_pos_season_count.sel(season=season).plot(
        ax=axes[1, i],
        transform=ccrs.PlateCarree(),
        add_colorbar=False, vmin=1, vmax=vmax, cmap=cm.lipari_r, levels=15
    )
    axes[1, i].set_title(f"Positive snow anomalies – {season}")
    axes[1, i].coastlines()

# --- One shared colorbar (optional but recommended) ---
cbar = fig.colorbar(
    axes[1, 0].collections[0],
    ax=axes,
    
    orientation="vertical",
    shrink=0.8,
    label="Count"
)


plt.savefig('swe_greenland.png')




seasons = ["DJF", "MAM", "JJA", "SON"]

fig, axes = plt.subplots(
    nrows=2, ncols=4,
    figsize=(20, 8),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)


# --- Row 1: negative snow anomalies ---
for i, season in enumerate(seasons):
    axes[0,i].set_autoscale_on(False)
    snow_neg_season_count.sel(season=season).plot(
        ax=axes[0, i],
        transform=ccrs.PlateCarree(),
        add_colorbar=False, vmin=1, vmax=vmax, cmap=cm.lipari_r, levels=15
    )
    axes[0, i].set_title(f"Negative snow anomalies – {season}")
    axes[0, i].coastlines()

# --- Row 2: positive snow anomalies ---
for i, season in enumerate(seasons):
    axes[1,i].set_autoscale_on(False)
    snow_pos_season_count.sel(season=season).plot(
        ax=axes[1, i],
        transform=ccrs.PlateCarree(),
        add_colorbar=False, vmin=1, vmax=vmax, cmap=cm.lipari_r, levels=15
    )
    axes[1, i].set_title(f"Positive snow anomalies – {season}")
    axes[1, i].coastlines()

# --- One shared colorbar (optional but recommended) ---
cbar = fig.colorbar(
    axes[1, 0].collections[0],
    ax=axes,
    
    orientation="vertical",
    shrink=0.8,
    label="Count"
)


plt.savefig('snow_greenland.png')

