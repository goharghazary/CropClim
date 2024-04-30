import xarray
from datacube import Datacube
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.datahandling import load_ard


def feature_layers(query: dict) -> xarray.Dataset:
    """
    A function for generating features layers, which must accept a
    datacube query dictionary and return a single xarray.Dataset
    or xarray.DataArray containing 2D coordinates (i.e x, y - no time dimension). e.g.

    ```
    def feature_function(query):
        dc = datacube.Datacube(app='feature_layers')
        ds = dc.load(**query)
        ds = ds.mean('time')
        return ds
    ```

    Parameters
    ----------
    query : dict
        Datacube query used to load odc data.

    Returns
    -------
    xarray.Dataset
        Feature layers
    """

    # connect to the datacube
    dc = Datacube(app="feature_layers")

    # Load the crop mask
    crop_mask_query = query.copy()
    crop_mask_query.update({"time": ("2019")})
    crop_mask_ds = dc.load(product="crop_mask", measurements=["mask"], **crop_mask_query).squeeze()
    crop_mask_da = crop_mask_ds["mask"]

    # Load Sentinel-1 data
    s1_ds = load_ard(
        dc=dc,
        products=["s1_rtc"],
        measurements=["vv", "vh"],
        group_by="solar_day",
        dtype="native",
        verbose=False,
        **query,
    )
    # Mask the sentinel-1 data using the crop mask.
    s1_ds_masked = s1_ds.where(crop_mask_da)
    # Resample the data to monthly means.
    s1_ds_monthly_means = s1_ds_masked.resample(time="M").mean()

    # Load Sentinel-2 data
    s2_ds = load_ard(
        dc=dc,
        products=["s2_l2a"],
        measurements=[
            "red",
            "green",
            "blue",
            "red_edge_1",
            "red_edge_2",
            "red_edge_3",
            "nir",
            "nir_narrow",
            "swir_1",
            "swir_2",
        ],
        group_by="solar_day",
        dtype="native",
        verbose=False,
        **query,
    )

    # Mask the sentinel-2 data using the crop mask.
    s2_ds_masked = s2_ds.where(crop_mask_da)
    # Resample the data to monthly means.
    s2_ds_monthly_means = s2_ds_masked.resample(time="M").mean()

    # Calculate band indices
    s2_ds_monthly_means = calculate_indices(
        s2_ds_monthly_means, index=["EVI", "SAVI", "MSAVI", "LAI", "NDVI"], satellite_mission="s2"
    )
    s2_ds_monthly_means["CI"] = s2_ds_monthly_means["red"] / s2_ds_monthly_means["nir"]

    # Get all the data in the proper format
    ds_to_merge = []
    for ds in [s1_ds_monthly_means, s2_ds_monthly_means]:
        bands = list(ds.data_vars)
        time_values = ds.time.values
        for band in bands:
            band_da = ds[band]
            for time_step in time_values:
                band_da_time_step = band_da.sel(time=time_step)
                ds_to_merge.append(
                    band_da_time_step.to_dataset(name=f"{band}_{str(time_step)[:10]}")
                )

    # merge results into single dataset
    result_ds = xarray.merge(ds_to_merge, compat="override")
    return result_ds
