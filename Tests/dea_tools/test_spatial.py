import pytest
import rioxarray
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

import datacube
from odc.geo.geom import Geometry
from datacube.utils.masking import mask_invalid_data

from dea_tools.spatial import subpixel_contours, xr_vectorize, xr_rasterize


@pytest.fixture(
    params=[
        "EPSG:4326",  # WGS84
        "EPSG:3577",  # Australian Albers
        "EPSG:32756",  # UTM 56S
    ],
    ids=["dem_da_epsg4326", "dem_da_epsg3577", "dem_da_epsg32756"],
)
def dem_da(request):
    # Read elevation data from file
    raster_path = "Supplementary_data/Reprojecting_data/canberra_dem_250m.tif"
    da = rioxarray.open_rasterio(raster_path).squeeze("band")

    # Reproject if required
    crs = request.param
    if crs:
        # Reproject and mask out nodata
        da = da.odc.reproject(crs)
        da = da.where(da != da.nodata)

    return da


# Create standard datacube load data in different CRSs and resolutions
@pytest.fixture(
    params=[
        ("EPSG:4326", (-0.00025, 0.00025)),  # WGS84, 0.0025 degree pixels
        ("EPSG:3577", (-30, 30)),  # Australian Albers 30 m pixels
    ],
    ids=["satellite_da_epsg4326", "satellite_da_epsg3577"],
)
def satellite_da(request):
    # Obtain CRS and resolution params
    crs, res = request.param

    # Connect to datacube
    dc = datacube.Datacube()

    # Load example satellite data
    ds = dc.load(
        product="ga_ls8c_ard_3",
        measurements=["nbart_nir"],
        x=(122.2183 - 0.08, 122.2183 + 0.08),
        y=(-18.0008 - 0.08, -18.0008 + 0.08),
        time="2020-01",
        output_crs=crs,
        resolution=res,
        group_by="solar_day",
    )

    # Mask nodata
    ds = mask_invalid_data(ds)

    # Return single array
    return ds.nbart_nir


# Create single pixel sample data in different CRSs
@pytest.fixture(
    params=[
        "EPSG:4326",  # WGS84
        "EPSG:3577",  # Australian Albers
    ],
    ids=["pixel_da_epsg4326", "pixel_da_epsg3577"],
)
def pixel_da(request):
    # Obtain CRS and resolution params
    crs = request.param

    # Create point geometry to load data with
    geom_point = Geometry(geom=Point(149.130, -35.284), crs="EPSG:4326")

    # Connect to datacube
    dc = datacube.Datacube()

    # Load example satellite data
    ds = dc.load(
        product="ga_ls8c_ard_3",
        measurements=["nbart_nir"],
        geopolygon=geom_point,
        time="2020-01-07",
        output_crs=crs,
    )

    # Return single array
    return ds.nbart_nir


# Load and reproject categorical raster to use for vectorization/rasterization
@pytest.fixture(
    params=[
        "EPSG:4326",  # WGS84
        "EPSG:3577",  # Australian Albers
        "EPSG:32756",  # UTM 56S
    ],
    ids=[
        "categorical_da_epsg4326",
        "categorical_da_epsg3577",
        "categorical_da_epsg32756",
    ],
)
def categorical_da(request):
    # Read categorical raster from file
    raster_path = "Tests/data/categorical_raster.tif"
    da = rioxarray.open_rasterio(raster_path).squeeze("band")

    # Reproject
    crs = request.param
    da = da.odc.reproject(crs, resampling="nearest")

    return da


# Create test GeoDataFrame data with different geometries and CRSs
@pytest.fixture(
    params=[
        ("point", "EPSG:4326"),  # Point geometry, WGS84
        ("line", "EPSG:4326"),  # Line geometry, WGS84
        ("poly", "EPSG:4326"),  # Polygon geometry, WGS84
        ("all", "EPSG:4326"),  # Multiple, WGS84
        ("point", "EPSG:3577"),  # Point geometry, Australian Albers
        ("line", "EPSG:3577"),  # Line geometry, Australian Albers
        ("poly", "EPSG:3577"),  # Polygon geometry, Australian Albers
        ("all", "EPSG:3577"),  # Multiple, Australian Albers
    ],
    ids=[
        "point_epsg4326",
        "line_epsg4326",
        "poly_epsg4326",
        "all_epsg4326",
        "point_epsg3577",
        "line_epsg3577",
        "poly_epsg3577",
        "all_epsg3577",
    ],
)
def sample_gdf(request):
    # Obtain geom type and CRS param
    geom, crs = request.param

    # Create geometries
    point = Point(149.130, -35.284)
    line = LineString(((149.134, -35.291), (149.138, -35.294)))
    poly = Polygon(
        (
            (149.144, -35.300),
            (149.149, -35.300),
            (149.149, -35.305),
            (149.144, -35.305),
        )
    )

    # Dict containing sample geometries
    geom_dict = {
        "point": [point],
        "line": [line],
        "poly": [poly],
        "all": [point, line, poly],
    }

    # Create geopandas.GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data={"attribute": range(1, len(geom_dict[geom]) + 1)},
        geometry=geom_dict[geom],
        crs="EPSG:4326",
    )

    # Reproject to custom CRS
    gdf = gdf.to_crs(crs)

    return gdf


@pytest.mark.parametrize(
    "attribute_col, expected_col",
    [
        (None, "attribute"),  # Default creates a column called "attribute"
        ("testing", "testing"),  # Use custom output column name
    ],
)
def test_xr_vectorize(categorical_da, attribute_col, expected_col):
    # Vectorize data
    categorical_gdf = xr_vectorize(categorical_da, attribute_col=attribute_col)

    # Test correct columns are included
    assert expected_col in categorical_gdf
    assert "geometry" in categorical_gdf

    # Assert geometry
    assert isinstance(categorical_gdf, gpd.GeoDataFrame)
    assert (categorical_gdf.geometry.type.values == "Polygon").all()

    # Assert values
    assert len(categorical_gdf.index) >= 26
    assert len(categorical_gdf[expected_col].unique()) == len(np.unique(categorical_da))
    assert categorical_gdf.crs == categorical_da.odc.crs


def test_xr_vectorize_mask(categorical_da):
    # Vectorize data using a mask to remove non-1 values
    categorical_gdf = xr_vectorize(categorical_da, mask=categorical_da == 1)

    # Verify only values in array are 1
    assert (categorical_gdf.attribute == 1).all()


def test_xr_vectorize_output_path(categorical_da):
    # Vectorize and export to file
    categorical_gdf = xr_vectorize(categorical_da, output_path="testing.geojson")

    # Test that data on file is the same as original data
    assert gpd.read_file("testing.geojson").equals(categorical_gdf)


def test_xr_rasterize(categorical_da, sample_gdf):
    """
    Tests xr_rasterize against many combinations of input rasters CRSs,
    and geodataframes (with point, line, polygon and combined features)
    in different CRSs.
    """
    # Rasterize vector
    rasterized_da = xr_rasterize(
        gdf=sample_gdf,
        da=categorical_da,
    )

    # Assert that output is an xarray.DataArray
    assert isinstance(rasterized_da, xr.DataArray)

    # Assert that output has same Geobox as input
    assert rasterized_da.odc.geobox == categorical_da.odc.geobox

    # Assert coordinates match in input and output
    in_y, in_x = categorical_da.odc.spatial_dims
    out_y, out_x = rasterized_da.odc.spatial_dims
    assert np.allclose(rasterized_da.coords[out_x], categorical_da.coords[in_x])
    assert np.allclose(rasterized_da.coords[out_y], categorical_da.coords[in_y])

    # Assert that outputs have at least one valid value
    assert rasterized_da.sum() > 0


@pytest.mark.parametrize(
    "name",
    [
        None,  # Default does not rename output array
        "testing",  # Use custom output array name
    ],
)
def test_xr_rasterize_roundtrip(categorical_da, name):
    """
    Tests whether a raster can be vectorized then rasterized and finish
    up identical to the original input.
    """
    # Create vector to rasterize
    categorical_gdf = xr_vectorize(categorical_da)

    # Rasterize vector using attributes
    rasterized_da = xr_rasterize(
        gdf=categorical_gdf, da=categorical_da, attribute_col="attribute", name=name
    )

    # Assert that output is an xarray.DataArray
    assert isinstance(rasterized_da, xr.DataArray)

    # Assert that array has correct name
    assert rasterized_da.name == name

    # Assert that rasterized output is the same as original input after round trip
    assert np.allclose(rasterized_da, categorical_da)
    assert rasterized_da.odc.geobox.crs == categorical_da.odc.geobox.crs


def test_xr_rasterize_output_path(categorical_da):
    # Create vector to rasterize
    categorical_gdf = xr_vectorize(categorical_da)

    # Rasterize vector using attributes
    rasterized_da = xr_rasterize(
        gdf=categorical_gdf,
        da=categorical_da,
        attribute_col="attribute",
        output_path="testing.tif",
    )

    # Assert that output GeoTIFF data is same as input
    loaded_da = rioxarray.open_rasterio("testing.tif").squeeze("band")
    assert np.allclose(loaded_da, rasterized_da)


def test_xr_rasterize_pixel(pixel_da, sample_gdf):
    """
    Tests if we can succesfully rasterize data into an xarray dataset
    with a single pixel (relevant to machine learning applications)
    """
    # Rasterize vector into footprint of single pixel dataset
    rasterized_da = xr_rasterize(
        gdf=sample_gdf,
        da=pixel_da,
    )
    
    # Assert that output is an xarray.DataArray
    assert isinstance(rasterized_da, xr.DataArray)


def test_xr_vectorize_pixel(pixel_da):
    """
    Tests if we can succesfully vectorize an xarray dataset with a 
    single pixel (relevant to machine learning applications).
    """
    # Vectorize single pixel dataset
    vectorized_gdf = xr_vectorize(pixel_da)
    
    # Assert that output is a geopandas.GeoDataFrame
    assert isinstance(vectorized_gdf, gpd.GeoDataFrame)


def test_subpixel_contours_dataseterror(dem_da):
    # Verify that function correctly raises error if xr.Dataset is passed
    with pytest.raises(ValueError):
        subpixel_contours(dem_da.to_dataset(name="test"), z_values=600)


@pytest.mark.parametrize(
    "z_values, expected",
    [
        (600, [600]),  # Single z-value, within DEM range
        ([600], [600]),  # Single z-value in list, within DEM range
        ([600, 700, 800], [600, 700, 800]),  # Multiple z, all within DEM range
        (0, []),  # Single z-value, outside DEM range
        ([0], []),  # Single z-value in list, outside DEM range
        ([0, 100, 200], []),  # Multiple z, all outside DEM range
        ([0, 700, 800], [700, 800]),  # Multiple z, some within DEM range
    ],
)
def test_subpixel_contours_dem(dem_da, z_values, expected):
    contours_gdf = subpixel_contours(dem_da, z_values=z_values)

    # Test correct columns are included
    assert "z_value" in contours_gdf
    assert "geometry" in contours_gdf

    # Test output is GeoDataFrame and all geometries are MultiLineStrings
    assert isinstance(contours_gdf, gpd.GeoDataFrame)
    assert (contours_gdf.geometry.type.values == "MultiLineString").all()

    # Verify that outputs are as expected
    assert contours_gdf.z_value.astype(int).to_list() == expected


@pytest.mark.parametrize(
    "z_values",
    [
        (0),  # Single z-value, all outside DEM range
        ([0]),  # Single z-value in list, all outside DEM range
        ([0, 100, 200]),  # Multiple z-values, all outside DEM range
    ],
)
def test_subpixel_contours_raiseerrors(dem_da, z_values):
    # Verify that function correctly raises error
    with pytest.raises(ValueError):
        subpixel_contours(dem_da, z_values=z_values, errors="raise")


@pytest.mark.parametrize(
    "min_vertices, expected_lines",
    [
        (2, [23, 25, 28]),  # Minimum 2 vertices; 23-28 linestrings expected
        (20, 5),  # Minimum 20 vertices; 5 linestrings expected
        (250, 1),  # Minimum 250 vertices; one linestring expected
    ],
)
def test_subpixel_contours_min_vertices(dem_da, min_vertices, expected_lines):
    contours_gdf = subpixel_contours(dem_da, z_values=600, min_vertices=min_vertices)

    # Check that number of individual linestrings match expected
    exploded_gdf = contours_gdf.geometry.explode(index_parts=False)
    assert np.isin(len(exploded_gdf.index), expected_lines)

    # Verify that minimum vertices are above threshold
    assert exploded_gdf.apply(lambda row: len(row.coords)).min() >= min_vertices


@pytest.mark.parametrize(
    "z_values, expected",
    [
        ([600, 700, 800], ["a", "b", "c"]),  # Valid data for a, b, c
        ([0, 700, 800], ["b", "c"]),  # Valid data for b, c
        ([0, 100, 800], ["c"]),  # Valid data for c only
        ([0, 100, 200], []),  # No valid data
    ],
)
def test_subpixel_contours_attribute_df(dem_da, z_values, expected):
    # Set up attribute dataframe (one row per elevation value above)
    attribute_df = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})

    contours_gdf = subpixel_contours(
        dem_da, z_values=z_values, attribute_df=attribute_df
    )

    # Verify correct columns are included in output
    assert "foo" in contours_gdf
    assert "bar" in contours_gdf
    assert "z_value" in contours_gdf
    assert "geometry" in contours_gdf

    # Verify that attributes are correctly included
    assert contours_gdf.bar.tolist() == expected


@pytest.mark.parametrize(
    "z_values, expected",
    [
        (3000, ["2020-01-04", "2020-01-13", "2020-01-20", "2020-01-29"]),
        (5000, ["2020-01-04", "2020-01-13", "2020-01-29"]),
        (7000, ["2020-01-04", "2020-01-13"]),
        (8000, ["2020-01-04"]),
    ],
)
def test_subpixel_contours_satellite_da(satellite_da, z_values, expected):
    contours_gdf = subpixel_contours(satellite_da, z_values=z_values)

    # Test correct columns are included
    assert "time" in contours_gdf
    assert "geometry" in contours_gdf

    # Test output is GeoDataFrame and all geometries are MultiLineStrings
    assert isinstance(contours_gdf, gpd.GeoDataFrame)
    assert (contours_gdf.geometry.type.values == "MultiLineString").all()

    # Verify that outputs are as expected
    assert contours_gdf.time.to_list() == expected


def test_subpixel_contours_multiple_z(satellite_da):
    # Verify that function correctly raises error multiple z values are
    # provided on inputs with multiple timesteps
    with pytest.raises(ValueError):
        subpixel_contours(satellite_da, z_values=[600, 700, 800])


def test_subpixel_contours_dim(satellite_da):
    # Rename dim to custom value
    satellite_da_date = satellite_da.rename({"time": "date"})

    # Verify that function correctly raises error if default dim of "time"
    # doesn't exist in the array
    with pytest.raises(KeyError):
        subpixel_contours(satellite_da_date, z_values=600)

    # Verify that function runs correctly if `dim="date"` is specified
    subpixel_contours(satellite_da_date, z_values=600, dim="date")


# def test_subpixel_contours_dem_crs(dem_da):
#     # Verify that an error is raised if data passed with no spatial ref/geobox
#     with pytest.raises(ValueError):
#         subpixel_contours(dem_da.drop_vars("spatial_ref"), z_values=700)

#     # Verify that no error is raised if we provide the correct CRS
#     subpixel_contours(dem_da.drop_vars("spatial_ref"), z_values=700, crs="EPSG:4326")
