# Functions used for loading and processing image data


# Imports
import os
import ee
import gc
import utm
import torch
import geemap
import shutil
import numpy as np
import pandas as pd
from PIL import Image



# Tile Loading Functions
def generate_ndvi_tile(lat, lon, step = .02):
    """
    Normalized Difference Vegetation Index
    Normalized Difference of Near Infrared (NIR) and Red bands
    NDVI (vegetation index): (NIR - RED)/(NIR + RED)
    lat: latitude, float
    lon: longitude, float
    step: designates tile area, float

    """
    ee.Authenticate()
    ee.Initialize()

    #Defining region
    region = ee.Geometry.Rectangle([lon - step, lat - step, lon + step, lat + step])

    #LANDSAT data
    collection = ee.ImageCollection(
        'LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(region) \
        .filterDate('2022-01-01', '2022-12-31') \
        .sort('CLOUD_COVER') \
        .first()

    #Scaling
    def scale_bands(img):
        return img.select(
                    ['SR_B4', 'SR_B5']) \
                  .multiply(0.0000275).add(-0.2) \
                  .rename(['Red', 'NIR'])

    image = scale_bands(collection)

    #calculate NDVI, clipping and producing image
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndvi = ndvi.clip(region)
    ndvi_array = ndvi.sampleRectangle(region=region, defaultValue=0).get('NDVI').getInfo()
    ndvi_np = np.array(ndvi_array).astype(np.float32).copy()
    ndvi_norm = np.clip((ndvi_np + 1) / 2, 0, 1)
    image = Image.fromarray((ndvi_norm * 255).astype(np.uint8))

    return image


def generate_ndbi_tile(lat, lon, step = .02):
    """
    Normalized Difference Built-Up Index
    Normalized Difference of Near Infrared (NIR) and Showt-Wave Infrared (SWIR)
    NDVI (Built-Up Index): (SWIR - NIR) / (SWIR + NIR)
    lat: latitude, float
    lon: longitude, float
    step: designates tile area, float
    """
    ee.Initialize()

    #Defining region
    region = ee.Geometry.Rectangle([lon - step, lat - step, lon + step, lat + step])

    #LANDSAT Data
    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterBounds(region) \
    .filterDate('2020-01-01', '2020-12-31') \
    .filterMetadata('CLOUD_COVER', 'less_than', 50) \
    .sort('CLOUD_COVER') \
    .first()

    #Scaling
    def scale_bands(img):
        return img.select(
                    ['SR_B6', 'SR_B5']) \
                  .multiply(0.0000275).add(-0.2) \
                  .rename(['SWIR', 'NIR'])



    image = scale_bands(landsat)

    #Calculating Index
    ndbi = image.normalizedDifference(['SWIR', 'NIR']).rename('NDBI')

    #Clipping index values to the region
    ndbi = ndbi.clip(region).unmask(0).reproject(crs='EPSG:4326', scale=30)

    #To array + resized
    ndbi_array = ndbi.sampleRectangle(region=region, defaultValue=0).get('NDBI').getInfo()
    ndbi_np = np.array(ndbi_array).astype(np.float32).copy()
    ndbi_norm = np.clip((ndbi_np + 1) / 2, 0, 1)

    img = Image.fromarray((ndbi_norm * 255).astype(np.uint8))

    return img



def generate_rgb_tile(lat,lon,step = 0.02, authenticate=False):
    """
    Standard Satellite Imagery
    lat: latitude, float
    lon: longitude, float
    step: designates tile area, float
    """

    #Optional authentication to generate a session token
    if authenticate:
        ee.Authenticate(auth_mode='notebook', force=True)
    else:
        ee.Authenticate()
    ee.Initialize()

    #Defining region
    region = ee.Geometry.Rectangle([lon-step, lat-step, lon+step, lat+step])

    #Copernicus data
    sentinel = ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(region) \
        .filterDate('2022-01-01', '2022-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .median() \
        .clip(region)

    #URL to get RGB
    url = sentinel.getThumbURL({
        'bands': ['B4', 'B3', 'B2'],  # RGB bands
        'region': region,
        'dimensions': 512,
        'min': 0,
        'max': 3000,
        'format': 'png'
    })

    #Get image, return image and region

    response = requests.get(url)
    image_pil = Image.open(io.BytesIO(response.content))
    geo_bounds = {'lat_min': lat-.02, 'lat_max': lat+.02, 'lon_min': lon-.02, 'lon_max':lon+.02}
    return image_pil, geo_bounds


def generate_elevation_tile(lat,lon,step = 0.02):
    """
    Elevation in Meters, Topograchic Data from the USGS
    lat: latitude, float
    lon: longitude, float
    step: designates tile area, float
    """
    #Defining region
    aoi = ee.Geometry.BBox(lon-step, lat-step, lon+step, lat+step)

    #Specifying dataset
    srtm = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    out_dem = "dem.tif"
    geemap.ee_export_image(
        srtm,
        filename=out_dem,
        scale=30,
        region=aoi,
        file_per_band=False
    )

    with rasterio.open(out_dem) as src:
        elevation_array = src.read(1)

    #Normalizing
    array_min = elevation_array.min()
    array_ptp = elevation_array.ptp() + 1e-8  # Zeros handling
    normalized = (elevation_array - array_min) / array_ptp

    image = Image.fromarray((normalized * 255).astype(np.uint8))

    return image






#for single bands (NDBI, NDVI, Elevation)
def preprocess_band(pil_img, normalize=True):
    """
    Normalize + resize image bands

    pil_img: PIL Image object
    normalize: bool
    """
    img = pil_img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[np.newaxis, :, :]

    if normalize:
        mean = arr.mean()
        std = arr.std() if arr.std() > 0 else 1.0
        arr = (arr - mean) / std

    tensor = torch.tensor(arr, dtype=torch.float32)  #[Channels, 224, 224]
    return tensor


def tile_to_tensor(tile):
    """
    Produces full tensor ready for feature extraction

    """


    # RGB first
    rgb_tensor = rgb_preprocess(tile["RGB"])  #[3, 224, 224]

    # Scientific bands
    ndvi_tensor = preprocess_band(tile["NDVI"])  #[1, 224, 224]

    ndbi_tensor = preprocess_band(tile["NDBI"])  #[1, 224, 224]

    elev_tensor = preprocess_band(tile["ELEV"])  #[1, 224, 224]

    #concatenate: [3 + 1 + 1 + 1 = 6, 224, 224]
    full_tensor = torch.cat([rgb_tensor, ndvi_tensor, ndbi_tensor, elev_tensor], dim=0)

    return full_tensor  #[6, 224, 224]

def generate_tile_from_row(row,authenticate=False):
    """
    Builds a tile; a collection of arrays of the image bands for each coordinate pair.

    row: each row in the create_batches_from_df function iterations, pandas dataframe object
    """

    lat, lon = row["latitude"], row["longitude"]
    tile = {}
    np_tile = {}

    rgb, meta = generate_rgb_tile(lat, lon, step=0.01, authenticate=authenticate)
    ndvi, _ = generate_ndvi_tile(lat, lon, step=0.01)
    ndbi, _ = generate_ndbi_tile(lat, lon, step=0.01)
    elev, _ = generate_elevation_tile(lat, lon, step=0.01)

    tile['RGB'] = rgb
    tile['NDVI'] = ndvi
    tile['NDBI'] = ndbi
    tile['ELEV'] = elev
    tile["metadata"] = meta


    np_tile['RGB'] = np.array(rgb).astype(np.float32)
    np_tile['NDVI'] = np.array(ndvi).astype(np.float32)
    np_tile['NDBI'] = np.array(ndbi).astype(np.float32)
    np_tile['ELEV'] = np.array(elev).astype(np.float32)
    np_tile['LAT'] = row['latitude']
    np_tile['LON'] = row['longitude']
    return tile, np_tile


def create_batches_from_df(df, batch_size=64,authenticate=False):
    """
    Generates the final tensor output from input coordinates dataframe

    df: pandas dataframe
    """
    all_batches = []
    all_coords = []
    tile_df = []
    current_batch = []
    current_meta = []

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        try:

            tile, _ = generate_tile_from_row(row,authenticate)
            tensor = tile_to_tensor(tile) #[6,224,224]
            if tensor.shape[0] != 6:
                print(f"Skipping ({row['latitude']}, {row['longitude']}) due to missing bands: {tensor.shape[0]}")
            else:
                current_batch.append(tensor)
                current_meta.append(tile["metadata"])
            if len(current_batch) == batch_size:
                batch_tensor = torch.stack(current_batch)
                all_batches.append(batch_tensor)
                all_coords.append(current_meta)
                current_batch = []
                current_meta = []

        except Exception as e:
            print(f"Skipping ({row['latitude']}, {row['longitude']}) due to error: {e}")

    if current_batch:
        batch_tensor = torch.stack(current_batch)
        all_batches.append(batch_tensor)
        all_coords.append(current_meta)


    return all_batches, all_coords


def save_chunk_torch(batches, coords, chunk_id, out_dir="chunks"):
    """
    saves batches and coordinates to directory
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "batches": batches,
        "coords": coords
    }, f"{out_dir}/tile_chunk_{chunk_id}.pt")
