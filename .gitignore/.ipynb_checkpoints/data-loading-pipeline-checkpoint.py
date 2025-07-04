{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Important:**\n",
    "\n",
    "This notebook is not essential nor is it recommended to run. The data required is enormous (about ten gigabytes), and the resnet model weights, features, and coords are saved in the Ruins_Forest_Classifier notebook already. Running this notebook will not give any new information, unless you are retraining resnet with your own coordinates, which is the next phase to improve this model. This notebook is also currently configured for gpu."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Amazon Rainforest Data Loading**\n",
    "\n",
    "The purpose of this notebook is to track all the processes used to load data and build features. This is so that all the functions used to create the torch files and CSVs can be reproduced if the current ones become corrupted. It also keeps the actual data analysis pipeline less cluttered."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-07-04T00:19:57.799884Z",
     "iopub.status.busy": "2025-07-04T00:19:57.799443Z",
     "iopub.status.idle": "2025-07-04T00:20:06.118675Z",
     "shell.execute_reply": "2025-07-04T00:20:06.117618Z",
     "shell.execute_reply.started": "2025-07-04T00:19:57.799860Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m22.2/22.2 MB\u001b[0m \u001b[31m67.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m:00:01\u001b[0m00:01\u001b[0m\n",
      "\u001b[?25h"
     ]
    }
   ],
   "source": [
    "!pip install -q rasterio\n",
    "!pip install -q utm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-07-04T00:20:06.122462Z",
     "iopub.status.busy": "2025-07-04T00:20:06.122071Z",
     "iopub.status.idle": "2025-07-04T00:20:14.023303Z",
     "shell.execute_reply": "2025-07-04T00:20:14.022738Z",
     "shell.execute_reply.started": "2025-07-04T00:20:06.122431Z"
    }
   },
   "outputs": [],
   "source": [
    "## Imports\n",
    "\n",
    "#Standard\n",
    "import os\n",
    "import gc\n",
    "import shutil\n",
    "\n",
    "#Coordinates\n",
    "import utm\n",
    "\n",
    "\n",
    "#Visualizations & Imaging\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "#Data & Math\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "\n",
    "#Sklearn\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
    "\n",
    "#Pytorch\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torchvision import models\n",
    "from torchvision.models import resnet50\n",
    "import torchvision.transforms as transforms\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "\n",
    "#Utilities\n",
    "import tqdm\n",
    "\n",
    "\n",
    "# Paths\n",
    "positive_ruins_coordinates_path = '/kaggle/input/amazon-data/amazon_data.csv'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.status.busy": "2025-07-04T00:19:24.801359Z",
     "iopub.status.idle": "2025-07-04T00:19:24.801594Z",
     "shell.execute_reply": "2025-07-04T00:19:24.801499Z",
     "shell.execute_reply.started": "2025-07-04T00:19:24.801485Z"
    }
   },
   "outputs": [],
   "source": [
    "## Original Data from Kaggle Ruins Locations in the Amazon\n",
    "\n",
    "## Constructs a csv of Coordinates of ruins and earthworks from datasets found on kaggle\n",
    "\n",
    "\n",
    "glyph = pd.read_csv('amazon_geoglyphs_sites.csv')\n",
    "arch_data = pd.read_csv('submit.csv')\n",
    "casarabe = pd.read_csv('casarabe_sites_utm.csv')\n",
    "arch_data = arch_data[['x','y']]\n",
    "arch_data['latitude'] = arch_data['y']\n",
    "arch_data['longitude'] = arch_data['x']\n",
    "arch_data = arch_data[['latitude','longitude']]\n",
    "glyph = glyph[['latitude','longitude']]\n",
    "\n",
    "\n",
    "#Fixing Warped Entries\n",
    "glyph.loc[1936,'latitude'],_ = glyph['latitude'][1936].split(' ')\n",
    "glyph.loc[2063,'latitude'],_ = glyph['latitude'][2063].split(' ')\n",
    "glyph.loc[2866,'latitude'],_ = glyph['latitude'][2866].split(' ')\n",
    "\n",
    "#Converting utm to lat/lon\n",
    "casarabe['easting'] = casarabe['UTM X (Easting)']\n",
    "casarabe['northing'] = casarabe['UTM Y (Northing)']\n",
    "casarabe = casarabe[['easting','northing']]\n",
    "casarabe['latitude'] = casarabe['easting']\n",
    "casarabe['longitude'] = casarabe['northing']\n",
    "\n",
    "def convert_to_utm(easting, northing):\n",
    "  zone_number = 20\n",
    "  zone_letter = 'S'\n",
    "  latitude_south, longitude_south = utm.to_latlon(easting, northing, zone_number, northern=False)\n",
    "\n",
    "  return latitude_south, longitude_south\n",
    "\n",
    "for i in range(len(casarabe)):\n",
    "  casarabe.loc[i,'latitude'], casarabe.loc[i,'longitude'] = convert_to_utm(casarabe['easting'][i],casarabe['northing'][i])\n",
    "\n",
    "casarabe = casarabe[['latitude','longitude']]\n",
    "\n",
    "#Glyph & arch_data have some crossover:\n",
    "\n",
    "glyph = glyph.drop_duplicates(subset=['latitude', 'longitude'], keep='first')\n",
    "arch_data = arch_data.drop_duplicates(subset=['latitude', 'longitude'], keep='first')\n",
    "\n",
    "glyph['latitude'] = round(glyph['latitude'],4)\n",
    "glyph['longitude'] = round(glyph['longitude'],3)\n",
    "\n",
    "\n",
    "\n",
    "for index, row in glyph.iterrows():\n",
    "  glyph.loc[index,'latitude'] = round(float(glyph['latitude'][index]),4)\n",
    "\n",
    "holder = pd.concat([glyph,arch_data,casarabe],ignore_index = True)\n",
    "\n",
    "\n",
    "before = len(holder)\n",
    "master_table = holder.drop_duplicates(subset=['latitude', 'longitude'], keep='first')\n",
    "print(len(master_table))\n",
    "after = len(master_table)\n",
    "\n",
    "print(f\"Duplicated coordinates: {before-after}\")\n",
    "master_table['latitude'] = master_table['latitude'].astype(str)\n",
    "\n",
    "for index, row in master_table.iterrows():\n",
    "  # Access the latitude value directly from the 'row' Series\n",
    "  master_table.loc[index,'latitude'] = float(row['latitude'])\n",
    "\n",
    "master_table.head()\n",
    "\n",
    "master_table.to_csv('amazon_data.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-07-04T00:20:14.025000Z",
     "iopub.status.busy": "2025-07-04T00:20:14.024549Z",
     "iopub.status.idle": "2025-07-04T00:20:14.038173Z",
     "shell.execute_reply": "2025-07-04T00:20:14.037425Z",
     "shell.execute_reply.started": "2025-07-04T00:20:14.024981Z"
    }
   },
   "outputs": [],
   "source": [
    "##Tile Loading Functions\n",
    "\n",
    "#from previous notebook, redefined here for visualizations & plotting\n",
    "\n",
    "def generate_ndvi_tile(lat, lon, step = .02):\n",
    "    \"\"\"\n",
    "    Normalized Difference Vegetation Index\n",
    "    Normalized Difference of Near Infrared (NIR) and Red bands\n",
    "    NDVI (vegetation index): (NIR - RED)/(NIR + RED)\n",
    "    lat: latitude, float\n",
    "    lon: longitude, float\n",
    "    step: designates tile area, float\n",
    " \n",
    "    \"\"\"\n",
    "    ee.Authenticate()\n",
    "    ee.Initialize()\n",
    "\n",
    "    #Defining region\n",
    "    region = ee.Geometry.Rectangle([lon - step, lat - step, lon + step, lat + step])\n",
    "\n",
    "    #LANDSAT data\n",
    "    collection = ee.ImageCollection(\n",
    "        'LANDSAT/LC08/C02/T1_L2') \\\n",
    "        .filterBounds(region) \\\n",
    "        .filterDate('2022-01-01', '2022-12-31') \\\n",
    "        .sort('CLOUD_COVER') \\\n",
    "        .first()\n",
    "\n",
    "    #Scaling\n",
    "    def scale_bands(img):\n",
    "        return img.select(\n",
    "                    ['SR_B4', 'SR_B5']) \\\n",
    "                  .multiply(0.0000275).add(-0.2) \\\n",
    "                  .rename(['Red', 'NIR'])\n",
    "\n",
    "    image = scale_bands(collection)\n",
    "\n",
    "    #calculate NDVI, clipping and producing image\n",
    "    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')\n",
    "    ndvi = ndvi.clip(region)\n",
    "    ndvi_array = ndvi.sampleRectangle(region=region, defaultValue=0).get('NDVI').getInfo()\n",
    "    ndvi_np = np.array(ndvi_array).astype(np.float32).copy()\n",
    "    ndvi_norm = np.clip((ndvi_np + 1) / 2, 0, 1)\n",
    "    image = Image.fromarray((ndvi_norm * 255).astype(np.uint8))\n",
    "\n",
    "    return image\n",
    "\n",
    "    \n",
    "def generate_ndbi_tile(lat, lon, step = .02):\n",
    "    \"\"\"\n",
    "    Normalized Difference Built-Up Index\n",
    "    Normalized Difference of Near Infrared (NIR) and Showt-Wave Infrared (SWIR)\n",
    "    NDVI (Built-Up Index): (SWIR - NIR) / (SWIR + NIR)\n",
    "    lat: latitude, float\n",
    "    lon: longitude, float\n",
    "    step: designates tile area, float\n",
    "    \"\"\"\n",
    "    ee.Initialize()\n",
    "\n",
    "    #Defining region\n",
    "    region = ee.Geometry.Rectangle([lon - step, lat - step, lon + step, lat + step])\n",
    "\n",
    "    #LANDSAT Data\n",
    "    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \\\n",
    "    .filterBounds(region) \\\n",
    "    .filterDate('2020-01-01', '2020-12-31') \\\n",
    "    .filterMetadata('CLOUD_COVER', 'less_than', 50) \\\n",
    "    .sort('CLOUD_COVER') \\\n",
    "    .first()\n",
    "\n",
    "    #Scaling\n",
    "    def scale_bands(img):\n",
    "        return img.select(\n",
    "                    ['SR_B6', 'SR_B5']) \\\n",
    "                  .multiply(0.0000275).add(-0.2) \\\n",
    "                  .rename(['SWIR', 'NIR'])\n",
    "    \n",
    "    \n",
    "    \n",
    "    image = scale_bands(landsat)\n",
    "\n",
    "    #Calculating Index\n",
    "    ndbi = image.normalizedDifference(['SWIR', 'NIR']).rename('NDBI')\n",
    "\n",
    "    #Clipping index values to the region\n",
    "    ndbi = ndbi.clip(region).unmask(0).reproject(crs='EPSG:4326', scale=30)\n",
    "    \n",
    "    #To array + resized\n",
    "    ndbi_array = ndbi.sampleRectangle(region=region, defaultValue=0).get('NDBI').getInfo()\n",
    "    ndbi_np = np.array(ndbi_array).astype(np.float32).copy()\n",
    "    ndbi_norm = np.clip((ndbi_np + 1) / 2, 0, 1)\n",
    "\n",
    "    img = Image.fromarray((ndbi_norm * 255).astype(np.uint8))\n",
    "\n",
    "    return img\n",
    "\n",
    "\n",
    "\n",
    "def generate_rgb_tile(lat,lon,step = 0.02, authenticate=False):\n",
    "    \"\"\"\n",
    "    Standard Satellite Imagery \n",
    "    lat: latitude, float\n",
    "    lon: longitude, float\n",
    "    step: designates tile area, float\n",
    "    \"\"\"\n",
    "\n",
    "    #Optional authentication to generate a session token\n",
    "    if authenticate:\n",
    "        ee.Authenticate(auth_mode='notebook', force=True)\n",
    "    else:\n",
    "        ee.Authenticate()\n",
    "    ee.Initialize()\n",
    "\n",
    "    #Defining region\n",
    "    region = ee.Geometry.Rectangle([lon-step, lat-step, lon+step, lat+step])\n",
    "\n",
    "    #Copernicus data\n",
    "    sentinel = ee.ImageCollection(\n",
    "        \"COPERNICUS/S2_SR_HARMONIZED\") \\\n",
    "        .filterBounds(region) \\\n",
    "        .filterDate('2022-01-01', '2022-12-31') \\\n",
    "        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \\\n",
    "        .median() \\\n",
    "        .clip(region)\n",
    "    \n",
    "    #URL to get RGB\n",
    "    url = sentinel.getThumbURL({\n",
    "        'bands': ['B4', 'B3', 'B2'],  # RGB bands\n",
    "        'region': region,\n",
    "        'dimensions': 512,\n",
    "        'min': 0,\n",
    "        'max': 3000,\n",
    "        'format': 'png'\n",
    "    })\n",
    "\n",
    "    #Get image, return image and region\n",
    "    \n",
    "    response = requests.get(url)\n",
    "    image_pil = Image.open(io.BytesIO(response.content))\n",
    "    geo_bounds = {'lat_min': lat-.02, 'lat_max': lat+.02, 'lon_min': lon-.02, 'lon_max':lon+.02}\n",
    "    return image_pil, geo_bounds\n",
    "    \n",
    "\n",
    "def generate_elevation_tile(lat,lon,step = 0.02):\n",
    "    \"\"\"\n",
    "    Elevation in Meters, Topograchic Data from the USGS\n",
    "    lat: latitude, float\n",
    "    lon: longitude, float\n",
    "    step: designates tile area, float\n",
    "    \"\"\"\n",
    "    #Defining region\n",
    "    aoi = ee.Geometry.BBox(lon-step, lat-step, lon+step, lat+step)\n",
    "    \n",
    "    #Specifying dataset\n",
    "    srtm = ee.Image(\"USGS/SRTMGL1_003\").clip(aoi)\n",
    "    out_dem = \"dem.tif\"\n",
    "    geemap.ee_export_image(\n",
    "        srtm,\n",
    "        filename=out_dem,\n",
    "        scale=30,\n",
    "        region=aoi,\n",
    "        file_per_band=False\n",
    "    )\n",
    "    \n",
    "    with rasterio.open(out_dem) as src:\n",
    "        elevation_array = src.read(1)\n",
    "        \n",
    "    #Normalizing\n",
    "    array_min = elevation_array.min()\n",
    "    array_ptp = elevation_array.ptp() + 1e-8  # Zeros handling\n",
    "    normalized = (elevation_array - array_min) / array_ptp\n",
    "\n",
    "    image = Image.fromarray((normalized * 255).astype(np.uint8))\n",
    "    \n",
    "    return image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-07-04T00:20:14.039277Z",
     "iopub.status.busy": "2025-07-04T00:20:14.039008Z",
     "iopub.status.idle": "2025-07-04T00:20:14.066106Z",
     "shell.execute_reply": "2025-07-04T00:20:14.065467Z",
     "shell.execute_reply.started": "2025-07-04T00:20:14.039252Z"
    }
   },
   "outputs": [],
   "source": [
    "#Processing Images to Tensors\n",
    "\n",
    "\n",
    "\n",
    "rgb_preprocess = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.ToTensor(),  # [3, 224, 224]\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406],\n",
    "                         std=[0.229, 0.224, 0.225])\n",
    "])\n",
    "\n",
    "\n",
    "#for single bands (NDBI, NDVI, Elevation)\n",
    "def preprocess_band(pil_img, normalize=True):\n",
    "    \"\"\"\n",
    "    Normalize + resize image bands\n",
    "\n",
    "    pil_img: PIL Image object\n",
    "    normalize: bool\n",
    "    \"\"\"\n",
    "    img = pil_img.resize((224, 224))\n",
    "    arr = np.array(img).astype(np.float32) / 255.0\n",
    "    arr = arr[np.newaxis, :, :] \n",
    "\n",
    "    if normalize:\n",
    "        mean = arr.mean()\n",
    "        std = arr.std() if arr.std() > 0 else 1.0\n",
    "        arr = (arr - mean) / std\n",
    "\n",
    "    tensor = torch.tensor(arr, dtype=torch.float32)  #[Channels, 224, 224]\n",
    "    return tensor\n",
    "    \n",
    "\n",
    "def tile_to_tensor(tile):\n",
    "    \"\"\"\n",
    "    Produces full tensor ready for feature extraction\n",
    "    \n",
    "    \"\"\"\n",
    "\n",
    "    \n",
    "    # RGB first\n",
    "    rgb_tensor = rgb_preprocess(tile[\"RGB\"])  #[3, 224, 224]\n",
    "    \n",
    "    # Scientific bands\n",
    "    ndvi_tensor = preprocess_band(tile[\"NDVI\"])  #[1, 224, 224]\n",
    "\n",
    "    ndbi_tensor = preprocess_band(tile[\"NDBI\"])  #[1, 224, 224]\n",
    "    \n",
    "    elev_tensor = preprocess_band(tile[\"ELEV\"])  #[1, 224, 224]\n",
    "    \n",
    "    #concatenate: [3 + 1 + 1 + 1 = 6, 224, 224]\n",
    "    full_tensor = torch.cat([rgb_tensor, ndvi_tensor, ndbi_tensor, elev_tensor], dim=0)\n",
    "    \n",
    "    return full_tensor  #[6, 224, 224]\n",
    "\n",
    "def generate_tile_from_row(row,authenticate=False):\n",
    "    \"\"\"\n",
    "    Builds a tile; a collection of arrays of the image bands for each coordinate pair.\n",
    "    \n",
    "    row: each row in the create_batches_from_df function iterations, pandas dataframe object\n",
    "    \"\"\"\n",
    "    \n",
    "    lat, lon = row[\"latitude\"], row[\"longitude\"]\n",
    "    tile = {}\n",
    "    np_tile = {}\n",
    "    \n",
    "    rgb, meta = generate_rgb_tile(lat, lon, step=0.01, authenticate=authenticate)\n",
    "    ndvi, _ = generate_ndvi_tile(lat, lon, step=0.01)\n",
    "    ndbi, _ = generate_ndbi_tile(lat, lon, step=0.01)\n",
    "    elev, _ = generate_elevation_tile(lat, lon, step=0.01)\n",
    "    \n",
    "    tile['RGB'] = rgb\n",
    "    tile['NDVI'] = ndvi\n",
    "    tile['NDBI'] = ndbi\n",
    "    tile['ELEV'] = elev\n",
    "    tile[\"metadata\"] = meta\n",
    "    \n",
    "        \n",
    "    np_tile['RGB'] = np.array(rgb).astype(np.float32)\n",
    "    np_tile['NDVI'] = np.array(ndvi).astype(np.float32)\n",
    "    np_tile['NDBI'] = np.array(ndbi).astype(np.float32)\n",
    "    np_tile['ELEV'] = np.array(elev).astype(np.float32)\n",
    "    np_tile['LAT'] = row['latitude']\n",
    "    np_tile['LON'] = row['longitude']\n",
    "    return tile, np_tile\n",
    "    \n",
    "def create_batches_from_df(df, batch_size=64,authenticate=False):\n",
    "    \"\"\"\n",
    "    Generates the final tensor output from input coordinates dataframe\n",
    "\n",
    "    df: pandas dataframe\n",
    "    \"\"\"\n",
    "    all_batches = []\n",
    "    all_coords = []\n",
    "    tile_df = []\n",
    "    current_batch = []\n",
    "    current_meta = []\n",
    "    \n",
    "    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):\n",
    "        try:\n",
    "            \n",
    "            tile, _ = generate_tile_from_row(row,authenticate)\n",
    "            tensor = tile_to_tensor(tile) #[6,224,224]\n",
    "            if tensor.shape[0] != 6:\n",
    "                print(f\"Skipping ({row['latitude']}, {row['longitude']}) due to missing bands: {tensor.shape[0]}\")\n",
    "            else:\n",
    "                current_batch.append(tensor)\n",
    "                current_meta.append(tile[\"metadata\"])\n",
    "            if len(current_batch) == batch_size:\n",
    "                batch_tensor = torch.stack(current_batch)\n",
    "                all_batches.append(batch_tensor)\n",
    "                all_coords.append(current_meta)\n",
    "                current_batch = []\n",
    "                current_meta = []\n",
    "\n",
    "        except Exception as e:\n",
    "            print(f\"Skipping ({row['latitude']}, {row['longitude']}) due to error: {e}\")\n",
    "\n",
    "    if current_batch:\n",
    "        batch_tensor = torch.stack(current_batch)\n",
    "        all_batches.append(batch_tensor)\n",
    "        all_coords.append(current_meta)\n",
    "        \n",
    "  \n",
    "    return all_batches, all_coords\n",
    "\n",
    "\n",
    "def save_chunk_torch(batches, coords, chunk_id, out_dir=\"chunks\"):\n",
    "    \"\"\"\n",
    "    saves batches and coordinates to directory\n",
    "    \"\"\"\n",
    "    os.makedirs(out_dir, exist_ok=True)\n",
    "    torch.save({\n",
    "        \"batches\": batches,\n",
    "        \"coords\": coords\n",
    "    }, f\"{out_dir}/tile_chunk_{chunk_id}.pt\")\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Tensor Batches from Coordinates**\n",
    "\n",
    "The following ten blocks utilize the tile generating functions to call the coordinates from the ruins CSV with a lon/lat buffer of .02 to obtain any characteristics of the earthworks surroundings (gives tiles showing 4.9284 km**2, I've found this range encapsulates larger earthworks and their surroundings nicely). These tiles are concatenated into a tensor with size (6,224,224). 6 Dimensions for the six bands: Red, Green, Blue, NDVI, NDBI, Elevation. Each of these bands has been resized to 224,224 for consistency and simplicity. The tensors are stacked into batches of varying size (max is 64, except for the last batch as it holds the leftovers), giving shape (Batch_Size,6,224,224). Batches are loaded, and the final product is saved using the save_chunk_torch function above. The batch tensors are all loaded in a list together, giving dimension (104,Batch_Size,6,224,224), for 104 batches each with 6 resized image bands. These tensors were created for the ruins locations in the csv and also true negatives (empty forest, uninhabited rivers, any image with a high confidence of no ruins) hand labeled by myself in order to create training data.\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "If you are following along with this notebook, only run these cells to produce tensor batches from coordinates, if features or tensors are already obtained these are unecessary. Be warned each cell can take up to or over an hour, depending on how many points it loads."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ee.Authenticate(auth_mode='notebook', force=True)\n",
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "df_chunk1 = df.iloc[:660]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=1)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part1.zip\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "df_chunk1 = df.iloc[660:1220]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=2)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part2.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "df_chunk1 = df.iloc[1220:1880]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=3)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part3.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "df_chunk1 = df.iloc[1880:2540]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=4)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part4.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "df_chunk1 = df.iloc[2540:3200]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=5)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part5.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "df_chunk1 = df.iloc[3200:3760]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=6)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part6.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('/kaggle/input/amazon-data/amazon_data.csv')\n",
    "import shutil\n",
    "df_chunk1 = df.iloc[3760:4420]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=7)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part7.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "import shutil\n",
    "df_chunk1 = df.iloc[4420:5080]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=8)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part8.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "df_chunk1 = df.iloc[5080:5740]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=9)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_part9.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(positive_ruins_coordinates_path)\n",
    "df_chunk1 = df.iloc[5740:]\n",
    "batch1, coord1 = create_batches_from_df(df_chunk1)\n",
    "save_chunk_torch(batch1, coord1, chunk_id=10)\n",
    "shutil.make_archive(\"tile_chunks\", 'zip', \"chunks\")\n",
    "os.rename(\"tile_chunks.zip\", \"tile_chunks_par10.zip\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Features From Tensor Batches**\n",
    "\n",
    "I played around with how to best exract features from these images in order to train a model accurately. I settled on pretrained ResNet applied to the 6 bands but with 5 epochs of training and validation, with the highest validation accuracy model state saved to the directory for building the features. I chose Stochastic Gradient Descent + Momentum for the optimizer as this dataset appears fairly noisy. I wanted to make sure that there were no chances of getting stuck in a local minima of the loss function, and the momentum helps to give a smooth and quick convergence on the global minima."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-07-04T00:20:17.237527Z",
     "iopub.status.busy": "2025-07-04T00:20:17.237238Z",
     "iopub.status.idle": "2025-07-04T00:21:37.414671Z",
     "shell.execute_reply": "2025-07-04T00:21:37.414031Z",
     "shell.execute_reply.started": "2025-07-04T00:20:17.237508Z"
    }
   },
   "outputs": [],
   "source": [
    "## Extract\n",
    "\n",
    "# Loading all the data that was previously saved to the directory#\n",
    "\n",
    "# Ensuring no Residual RAM\n",
    "\n",
    "# Excluding important libraries & variables from getting feleted\n",
    "keep_vars = {\n",
    "    'torch', 'gc', 'np', 'pd','best_model_state','nne_state','rgb_state', \n",
    "    'train_test_split','confusion_matrix', 'ConfusionMatrixDisplay', 'optim', \n",
    "    'resnet50', 'transforms', 'DataLoader', 'Dataset', 'nn', 'plt'}\n",
    "for name in dir():\n",
    "    if not name.startswith(\"_\") and name not in keep_vars:\n",
    "        del globals()[name]\n",
    "        keep_vars = {\n",
    "                    'torch', 'gc', 'np', 'pd','best_model_state','nne_state','rgb_state', \n",
    "                    'train_test_split','confusion_matrix', 'ConfusionMatrixDisplay', 'optim', \n",
    "                    'resnet50', 'transforms', 'DataLoader', 'Dataset', 'nn', 'plt'}\n",
    "# Device \n",
    "device = 'cuda'\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()\n",
    "\n",
    "## Paths\n",
    "#Actual Ruins found in Kaggle\n",
    "pt1 = '/kaggle/input/amazon-chunk-data/tile_chunk_1.pt'\n",
    "pt2 = '/kaggle/input/amazon-chunk-data/tile_chunk_2.pt'\n",
    "pt3 = '/kaggle/input/amazon-chunk-data/tile_chunk_3.pt'\n",
    "pt4 = '/kaggle/input/amazon-chunk-data/tile_chunk_4.pt'\n",
    "pt5 = '/kaggle/input/amazon-chunk-data/tile_chunk_5.pt'\n",
    "pt6 = '/kaggle/input/amazon-chunk-data/tile_chunk_6.pt'\n",
    "pt7 = '/kaggle/input/amazon-chunk-data/tile_chunk_7.pt'\n",
    "pt8 = '/kaggle/input/amazon-chunk-data/tile_chunk_8.pt'\n",
    "pt9 = '/kaggle/input/amazon-chunk-data/tile_chunk_9.pt'\n",
    "pt10 = '/kaggle/input/amazon-chunk-data/tile_chunk_10.pt'\n",
    "\n",
    "#Hand labeled not-ruins\n",
    "plf1 = '/kaggle/input/ruins-true-negatives/labeled_false_ruins.pt'\n",
    "plf2 = '/kaggle/input/labeled-false-ruins-2/labeled_false_ruins_2.pt'\n",
    "\n",
    "#Tiles of interest\n",
    "potential_ruins_1 = '/kaggle/input/potential-ruins/potential_ruins.pt'\n",
    "potential_ruins_2 = '/kaggle/input/tp-chunk-2/tile_chunk_tp_2.pt'\n",
    "\n",
    "# Loading \n",
    "t1 = torch.load(pt1,weights_only=False)\n",
    "t2 = torch.load(pt2,weights_only=False)\n",
    "t3 = torch.load(pt3,weights_only=False)\n",
    "t4 = torch.load(pt4,weights_only=False)\n",
    "t5 = torch.load(pt5,weights_only=False)\n",
    "t6 = torch.load(pt6,weights_only=False)\n",
    "t7 = torch.load(pt7,weights_only=False)\n",
    "t8 = torch.load(pt8,weights_only=False)\n",
    "t9 = torch.load(pt9,weights_only=False)\n",
    "t10 = torch.load(pt10,weights_only=False)\n",
    "f1 = torch.load(plf1,weights_only=False)\n",
    "f2 = torch.load(plf2,weights_only=False)\n",
    "\n",
    "# Features & Coords \n",
    "t1, t1_coord = torch.cat(t1['batches']),np.concatenate(t1['coords'])\n",
    "t2, t2_coord = torch.cat(t2['batches']),np.concatenate(t2['coords'])\n",
    "t3, t3_coord = torch.cat(t3['batches']),np.concatenate(t3['coords'])\n",
    "t4, t4_coord = torch.cat(t4['batches']),np.concatenate(t4['coords'])\n",
    "t5, t5_coord = torch.cat(t5['batches']),np.concatenate(t5['coords'])\n",
    "t6, t6_coord = torch.cat(t6['batches']),np.concatenate(t6['coords'])\n",
    "t7, t7_coord = torch.cat(t7['batches']),np.concatenate(t7['coords'])\n",
    "t8, t8_coord = torch.cat(t8['batches']),np.concatenate(t8['coords'])\n",
    "t9, t9_coord = torch.cat(t9['batches']),np.concatenate(t9['coords'])\n",
    "t10, t10_coord = torch.cat(t10['batches']),np.concatenate(t10['coords'])\n",
    "f1, f1_coord = torch.cat(f1['batches']),np.concatenate(f1['coords'])\n",
    "f2, f2_coord = torch.cat(f2['batches']),np.concatenate(f2['coords'])\n",
    "\n",
    "\n",
    "# Dataset \n",
    "true_ruins_batches = torch.cat((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10))\n",
    "true_ruins_coords = np.concatenate((t1_coord, t2_coord,\n",
    "                                      t3_coord, t4_coord,\n",
    "                                      t5_coord, t6_coord,\n",
    "                                      t7_coord, t8_coord,\n",
    "                                      t9_coord, t10_coord))\n",
    "\n",
    "\n",
    "false_ruins_batches = torch.cat((f1, f2))\n",
    "false_ruins_coords = np.concatenate((f1_coord, f2_coord))\n",
    "\n",
    "del t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,f1,f2\n",
    "del t1_coord, t2_coord, t3_coord, t4_coord, t5_coord, t6_coord, t7_coord, t8_coord, t9_coord, t10_coord, f1_coord, f2_coord\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()\n",
    "\n",
    "\n",
    "\n",
    "# Current Band layout [R,G,B,NDVI,NDBI,Elevation]\n",
    "\n",
    "# Making labels and creating full dataset\n",
    "false_labels = torch.zeros(len(false_ruins_batches))\n",
    "true_labels = torch.ones(len(true_ruins_batches))\n",
    "\n",
    "x = torch.cat((true_ruins_batches,false_ruins_batches))\n",
    "y = torch.cat((true_labels,false_labels))\n",
    "\n",
    "del true_ruins_batches,false_ruins_batches\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-06-29T22:41:29.614700Z",
     "iopub.status.busy": "2025-06-29T22:41:29.614419Z",
     "iopub.status.idle": "2025-06-29T22:41:29.621332Z",
     "shell.execute_reply": "2025-06-29T22:41:29.620671Z",
     "shell.execute_reply.started": "2025-06-29T22:41:29.614675Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "            <style>\n",
       "                .geemap-dark {\n",
       "                    --jp-widgets-color: white;\n",
       "                    --jp-widgets-label-color: white;\n",
       "                    --jp-ui-font-color1: white;\n",
       "                    --jp-layout-color2: #454545;\n",
       "                    background-color: #383838;\n",
       "                }\n",
       "\n",
       "                .geemap-dark .jupyter-button {\n",
       "                    --jp-layout-color3: #383838;\n",
       "                }\n",
       "\n",
       "                .geemap-colab {\n",
       "                    background-color: var(--colab-primary-surface-color, white);\n",
       "                }\n",
       "\n",
       "                .geemap-colab .jupyter-button {\n",
       "                    --jp-layout-color3: var(--colab-primary-surface-color, white);\n",
       "                }\n",
       "            </style>\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Dataset Class \n",
    "\n",
    "class Tensor_Dataset(Dataset):\n",
    "    def __init__(self,data,labels=None,coord=None,transform=None):\n",
    "        self.data = data\n",
    "        self.labels = labels\n",
    "        self.transform = transform\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.data)\n",
    "\n",
    "    def __getitem__(self,idx):\n",
    "        dset = self.data[idx]\n",
    "        if self.labels is not None:\n",
    "            lab = self.labels[idx]\n",
    "            return dset, lab\n",
    "        if self.transform:\n",
    "            dset = self.transform(dset)\n",
    "        return dset\n",
    "        \n",
    "        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-06-29T22:41:29.622216Z",
     "iopub.status.busy": "2025-06-29T22:41:29.621986Z",
     "iopub.status.idle": "2025-06-29T22:41:37.921017Z",
     "shell.execute_reply": "2025-06-29T22:41:37.920191Z",
     "shell.execute_reply.started": "2025-06-29T22:41:29.622194Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "            <style>\n",
       "                .geemap-dark {\n",
       "                    --jp-widgets-color: white;\n",
       "                    --jp-widgets-label-color: white;\n",
       "                    --jp-ui-font-color1: white;\n",
       "                    --jp-layout-color2: #454545;\n",
       "                    background-color: #383838;\n",
       "                }\n",
       "\n",
       "                .geemap-dark .jupyter-button {\n",
       "                    --jp-layout-color3: #383838;\n",
       "                }\n",
       "\n",
       "                .geemap-colab {\n",
       "                    background-color: var(--colab-primary-surface-color, white);\n",
       "                }\n",
       "\n",
       "                .geemap-colab .jupyter-button {\n",
       "                    --jp-layout-color3: var(--colab-primary-surface-color, white);\n",
       "                }\n",
       "            </style>\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## Train/Test/Val Split \n",
    "\n",
    "\n",
    "X_train, X_temp, y_train, y_temp = train_test_split(x,y, test_size = .2, random_state = 42, stratify = y)\n",
    "X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size = .5, random_state = 42, stratify= y_temp)\n",
    "train_set = Tensor_Dataset(X_train,y_train)\n",
    "test_set = Tensor_Dataset(X_test,y_test)\n",
    "val_set = Tensor_Dataset(X_val,y_val)\n",
    "\n",
    "del x, y\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()\n",
    "\n",
    "train = DataLoader(train_set, batch_size=64)\n",
    "test = DataLoader(test_set, batch_size=64)\n",
    "val = DataLoader(val_set, batch_size=64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-06-29T22:49:57.727271Z",
     "iopub.status.busy": "2025-06-29T22:49:57.726713Z",
     "iopub.status.idle": "2025-06-29T22:49:57.734004Z",
     "shell.execute_reply": "2025-06-29T22:49:57.733396Z",
     "shell.execute_reply.started": "2025-06-29T22:49:57.727247Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "            <style>\n",
       "                .geemap-dark {\n",
       "                    --jp-widgets-color: white;\n",
       "                    --jp-widgets-label-color: white;\n",
       "                    --jp-ui-font-color1: white;\n",
       "                    --jp-layout-color2: #454545;\n",
       "                    background-color: #383838;\n",
       "                }\n",
       "\n",
       "                .geemap-dark .jupyter-button {\n",
       "                    --jp-layout-color3: #383838;\n",
       "                }\n",
       "\n",
       "                .geemap-colab {\n",
       "                    background-color: var(--colab-primary-surface-color, white);\n",
       "                }\n",
       "\n",
       "                .geemap-colab .jupyter-button {\n",
       "                    --jp-layout-color3: var(--colab-primary-surface-color, white);\n",
       "                }\n",
       "            </style>\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## Plotting\n",
    "\n",
    "\n",
    "def display_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):\n",
    "    \"\"\"\n",
    "    Displays model performance plots\n",
    "\n",
    "    train_acc, valid_acc, train_loss, valid_loss: lists of epoch, acc/loss\n",
    "    \"\"\"\n",
    "\n",
    "    plt.figure(figsize=(10, 7))\n",
    "    plt.plot(train_loss, color='tab:blue', linestyle='-', label='train loss')\n",
    "    plt.plot(valid_loss, color='tab:red', linestyle='-', label='validataion loss')\n",
    "    plt.xlabel('Epochs')\n",
    "    plt.ylabel('Loss')\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "\n",
    "    plt.figure(figsize=(10, 7))\n",
    "    plt.plot(train_acc, color='tab:blue', linestyle='-', label='train accuracy')\n",
    "    plt.plot(valid_acc, color='tab:red', linestyle='-', label='validataion accuracy')\n",
    "    plt.xlabel('Epochs')\n",
    "    plt.ylabel('Accuracy')\n",
    "    plt.legend()\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-06-29T22:49:58.579891Z",
     "iopub.status.busy": "2025-06-29T22:49:58.578696Z",
     "iopub.status.idle": "2025-06-29T22:56:03.021682Z",
     "shell.execute_reply": "2025-06-29T22:56:03.021045Z",
     "shell.execute_reply.started": "2025-06-29T22:49:58.579832Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "            <style>\n",
       "                .geemap-dark {\n",
       "                    --jp-widgets-color: white;\n",
       "                    --jp-widgets-label-color: white;\n",
       "                    --jp-ui-font-color1: white;\n",
       "                    --jp-layout-color2: #454545;\n",
       "                    background-color: #383838;\n",
       "                }\n",
       "\n",
       "                .geemap-dark .jupyter-button {\n",
       "                    --jp-layout-color3: #383838;\n",
       "                }\n",
       "\n",
       "                .geemap-colab {\n",
       "                    background-color: var(--colab-primary-surface-color, white);\n",
       "                }\n",
       "\n",
       "                .geemap-colab .jupyter-button {\n",
       "                    --jp-layout-color3: var(--colab-primary-surface-color, white);\n",
       "                }\n",
       "            </style>\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.\n",
      "  warnings.warn(\n",
      "/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.\n",
      "  warnings.warn(msg)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validating\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 13/13 [00:02<00:00,  4.46it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Initial Validation Loss: 0.6305159559616675, Initial Validation Accuracy: 75.65543071161049\n",
      "Training\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 101/101 [01:02<00:00,  1.61it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validating\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 13/13 [00:03<00:00,  4.29it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1 of 5: Train Accuracy: 95.69019362898189, Train Loss:0.10563450847927591, Validation Accuracy: 83.270911360799, Validation Loss: 0.42357763189535874\n",
      "Training\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 101/101 [01:07<00:00,  1.50it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validating\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 13/13 [00:03<00:00,  4.08it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2 of 5: Train Accuracy: 98.20424734540912, Train Loss:0.04927303354338844, Validation Accuracy: 98.12734082397004, Validation Loss: 0.04839057973227822\n",
      "Training\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 101/101 [01:10<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validating\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 13/13 [00:03<00:00,  4.11it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 3 of 5: Train Accuracy: 99.10993129294191, Train Loss:0.02645857119558167, Validation Accuracy: 92.38451935081149, Validation Loss: 0.1840582342388538\n",
      "Training\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 101/101 [01:10<00:00,  1.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validating\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 13/13 [00:03<00:00,  4.05it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 4 of 5: Train Accuracy: 99.45346658338538, Train Loss:0.019734225283425284, Validation Accuracy: 97.50312109862672, Validation Loss: 0.07223210545131363\n",
      "Training\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 101/101 [01:10<00:00,  1.44it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validating\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 13/13 [00:03<00:00,  4.03it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 5 of 5: Train Accuracy: 99.5471580262336, Train Loss:0.013962452132185681, Validation Accuracy: 97.37827715355806, Validation Loss: 0.1546404793446597\n",
      "Validating\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 13/13 [00:03<00:00,  4.09it/s]\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA04AAAJaCAYAAAAYkBe4AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAACRZUlEQVR4nOzdd3hU1d7F8e/MZNLLBEhCSSCAIKAIShDLtaMgitSgwhV7b4go9voqoqigWLGjXiChCBZAESyIUhRFUURKCoQUSG8zmTnvH0iuXCW0JDuTrM/z5BFOZuasQUpWzj6/bbMsy0JERERERET2yW46gIiIiIiISEOn4iQiIiIiIrIfKk4iIiIiIiL7oeIkIiIiIiKyHypOIiIiIiIi+6HiJCIiIiIish8qTiIiIiIiIvuh4iQiIiIiIrIfAaYD1Defz8f27duJiIjAZrOZjiMiIiIiIoZYlkVxcTGtW7fGbq/5mlKTK07bt28nISHBdAwREREREWkgMjIyiI+Pr/ExTa44RUREALt/cSIjIw2nERERERERU4qKikhISKjuCDVpcsVpz/K8yMhIFScRERERETmgW3g0HEJERERERGQ/VJxERERERET2Q8VJRERERERkP5rcPU4iIiIiYpZlWVRVVeH1ek1HkSbA6XTicDgO+3VUnERERESk3rjdbrKysigrKzMdRZoIm81GfHw84eHhh/U6Kk4iIiIiUi98Ph9btmzB4XDQunVrAgMDD2iamcihsiyL3NxcMjMz6dSp02FdeVJxEhEREZF64Xa78fl8JCQkEBoaajqONBExMTFs3boVj8dzWMVJwyFEREREpF7Z7foSVOpPbV3V1O9aERERERGR/VBxEhERERER2Q8VJxERERGRepSYmMjkyZONv4YcHA2HEBERERGpwemnn07Pnj1rraisWrWKsLCwWnktqT8qTiIiIiIih8myLLxeLwEB+//yOiYmph4SSW3TUj0RERERMcayLMrcVfX+YVnWAeW77LLL+OKLL5gyZQo2mw2bzcbWrVtZtmwZNpuNTz75hF69ehEUFMTXX3/Npk2bGDRoEHFxcYSHh9O7d28+++yzvV7zf5fZ2Ww2XnvtNYYMGUJoaCidOnVi/vz5B/XrmJ6ezqBBgwgPDycyMpIRI0aQnZ1d/fkff/yRM844g4iICCIjI+nVqxerV68GIC0tjYEDBxIdHU1YWBhHHXUUH3/88UGdvynQFScRERERMabc46XbA4vq/bzrH+lHaOD+vxSeMmUKv//+O0cffTSPPPII8N99gQDuuusuJk2aRIcOHYiOjiYjI4MBAwbw2GOPERQUxDvvvMPAgQPZsGEDbdu23ed5Hn74YZ588kmeeuopnn/+eUaNGkVaWhrNmjXbb0afz1ddmr744guqqqq48cYbufDCC1m2bBkAo0aN4thjj+Wll17C4XCwdu1anE4nADfeeCNut5svv/ySsLAw1q9fT3h4+H7P29SoOImIiIiI7ENUVBSBgYGEhobSsmXLv33+kUce4eyzz67+ebNmzejRo0f1zx999FHmzp3L/Pnzuemmm/Z5nssuu4yLL74YgMcff5znnnuOlStX0r9///1mXLJkCevWrWPLli0kJCQA8M4773DUUUexatUqevfuTXp6OnfccQddunQBoFOnTtXPT09PZ9iwYXTv3h2ADh067PecTZGKk4iIiIgYE+J0sP6RfkbOWxuSkpL2+nlJSQkPPfQQH330EVlZWVRVVVFeXk56enqNr3PMMcdU/zgsLIzIyEhycnIOKMOvv/5KQkJCdWkC6NatGy6Xi19//ZXevXszduxYrrrqKqZPn07fvn1JTk6mY8eOANxyyy1cf/31LF68mL59+zJs2LC98shuusdJRERERIyx2WyEBgbU+4fNZquV/P87HW/cuHHMnTuXxx9/nK+++oq1a9fSvXt33G53ja+zZ9ncX39dfD5frWQEeOihh/jll18477zz+Pzzz+nWrRtz584F4KqrrmLz5s1ccsklrFu3jqSkJJ5//vlaO3djoeIkIiIiIlKDwMBAvF7vAT12+fLlXHbZZQwZMoTu3bvTsmXL6vuh6krXrl3JyMggIyOj+tj69espKCigW7du1cc6d+7MbbfdxuLFixk6dChvvvlm9ecSEhK47rrrmDNnDrfffjvTpk2r08z+SMXJMMvrxaqqMh1DRERERPYhMTGR7777jq1bt5KXl1fjlaBOnToxZ84c1q5dy48//sjIkSNr9crRP+nbty/du3dn1KhRfP/996xcuZLRo0dz2mmnkZSURHl5OTfddBPLli0jLS2N5cuXs2rVKrp27QrAmDFjWLRoEVu2bOH7779n6dKl1Z+T/1JxMmjX9Hf546y+FC9ebDqKiIiIiOzDuHHjcDgcdOvWjZiYmBrvV3rmmWeIjo7mpJNOYuDAgfTr14/jjjuuTvPZbDY++OADoqOjOfXUU+nbty8dOnRg5syZADgcDnbu3Mno0aPp3LkzI0aM4Nxzz+Xhhx8GwOv1cuONN9K1a1f69+9P586defHFF+s0sz+yWQc6xL6RKCoqIioqisLCQiIjI41myX3uOfJefImwk06k7RtvGM0iIiIiUtcqKirYsmUL7du3Jzg42HQcaSJq+n13MN1AV5wMiho6DGw2Sr9Zgfsva1JFRERERKRhUXEyKDC+DWEnnQRAwezZhtOIiIiIiMi+qDgZ5kpOBqBwzlwNiRARERERaaBUnAyLOPMMHM2aUZWTQ8mXX5mOIyIiIiIi/0DFyTBbYCBRgwcDUJCSYjaMiIiIiIj8IxWnBsA1fBgAJV98gSc723AaERERERH5XypODUBQhw6EJPUCn4/CuXNNxxERERERkf+h4tRAuIYPB6AgdTZWHe8uLSIiIiIiB0fFqYGI7NcPe0QEnsxMyr791nQcEREREalFiYmJTJ48ufrnNpuNefPm7fPxW7duxWazsXbt2jrP9tBDD9GzZ886P8/+3nNDp+LUQNhDQogaeD4ABamphtOIiIiISF3Kysri3HPPrdXXvOyyyxj859CxgzFu3DiWLFlSq1kaIxWnBmTPnk7Fn35GVX6+4TQiIiIiUldatmxJUFCQ6RgAhIeH07x5c9MxGjwVpwYkuGtXgo86CsvjofCDD0zHEREREWnyXn31VVq3bo3vf+5BHzRoEFdccQUAmzZtYtCgQcTFxREeHk7v3r357LPPanzd/122tnLlSo499liCg4NJSkrihx9+2OvxXq+XK6+8kvbt2xMSEsKRRx7JlClTqj//0EMP8fbbb/PBBx9gs9mw2WwsW7YMgPHjx9O5c2dCQ0Pp0KED999/Px6PZ6/n/nWpns/n45FHHiE+Pp6goCB69uzJwoULqz+/ZxnhnDlzOOOMMwgNDaVHjx6sWLHigH5N91i3bh1nnnkmISEhNG/enGuuuYaSkpLqzy9btozjjz+esLAwXC4XJ598MmlpaQD8+OOPnHHGGURERBAZGUmvXr1YvXr1QZ3/YKk4NTCu5D1DIlKxLMtwGhEREZG6ZVkWvrKyev840K+zkpOT2blzJ0uXLq0+tmvXLhYuXMioUaMAKCkpYcCAASxZsoQffviB/v37M3DgQNLT0w/oHCUlJZx//vl069aNNWvW8NBDDzFu3Li9HuPz+YiPjyclJYX169fzwAMPcM899zBr1ixg93K7ESNG0L9/f7KyssjKyuKkk04CICIigrfeeov169czZcoUpk2bxrPPPrvPPFOmTOHpp59m0qRJ/PTTT/Tr148LLriAjRs37vW4e++9l3HjxrF27Vo6d+7MxRdfTFVV1QG959LSUvr160d0dDSrVq0iJSWFzz77jJtuugmAqqoqBg8ezGmnncZPP/3EihUruOaaa7DZbACMGjWK+Ph4Vq1axZo1a7jrrrtwOp0HdO5DFVCnry4HLfL888me+CTuPzZR/sNaQo871nQkERERkTpjlZez4bhe9X7eI79fgy00dL+Pi46O5txzz+X999/nrLPOAiA1NZUWLVpwxhlnANCjRw969OhR/ZxHH32UuXPnMn/+/OoiUJP3338fn8/H66+/TnBwMEcddRSZmZlcf/311Y9xOp08/PDD1T9v3749K1asYNasWYwYMYLw8HBCQkKorKykZcuWe73+fffdV/3jxMRExo0bx4wZM7jzzjv/Mc+kSZMYP348F110EQATJ05k6dKlTJ48mRdeeKH6cePGjeO8884D4OGHH+aoo47ijz/+oEuXLgf0nisqKnjnnXcICwsDYOrUqQwcOJCJEyfidDopLCzk/PPPp2PHjgB07dq1+vnp6enccccd1efq1KnTfs95uHTFqYFxhIcT2b8/oCERIiIiIg3BqFGjmD17NpWVlQC89957XHTRRdjtu7+ULikpYdy4cXTt2hWXy0V4eDi//vrrAV9x+vXXXznmmGMIDg6uPnbiiSf+7XEvvPACvXr1IiYmhvDwcF599dUDOsfMmTM5+eSTadmyJeHh4dx33337fF5RURHbt2/n5JNP3uv4ySefzK+//rrXsWOOOab6x61atQIgJydnv3lg93vu0aNHdWnacw6fz8eGDRto1qwZl112Gf369WPgwIFMmTKFrKys6seOHTuWq666ir59+/LEE0+wadOmAzrv4dAVpwbIlZxM4dy5FH3yCXH33I0jPNx0JBEREZE6YQsJ4cjv1xg574EaOHAglmXx0Ucf0bt3b7766qu9lrqNGzeOTz/9lEmTJnHEEUcQEhLC8OHDcbvdtZZ3xowZjBs3jqeffpoTTzyRiIgInnrqKb777rsan7dixQpGjRrFww8/TL9+/YiKimLGjBk8/fTTh53pr0vj9iyh+997wQ7Hm2++yS233MLChQuZOXMm9913H59++iknnHACDz30ECNHjuSjjz7ik08+4cEHH2TGjBkMGTKk1s7/v1ScGqCQY3sS2LEj7k2bKPrwI6IvutB0JBEREZE6YbPZDmjJnEnBwcEMHTqU9957jz/++IMjjzyS4447rvrzy5cv57LLLqv+or2kpIStW7ce8Ot37dqV6dOnU1FRUX3V6dv/2ddz+fLlnHTSSdxwww3Vx/73KktgYCBer3evY9988w3t2rXj3nvvrT62Z8DCP4mMjKR169YsX76c0047ba/zH3/88Qf8nvana9euvPXWW5SWllZfdVq+fDl2u50jjzyy+nHHHnssxx57LHfffTcnnngi77//PieccAIAnTt3pnPnztx2221cfPHFvPnmm3VanLRUrwGy2Wz/HRKRkmI4jYiIiIiMGjWKjz76iDfeeKN6KMQenTp1Ys6cOaxdu5Yff/yRkSNHHtSVl5EjR2Kz2bj66qtZv349H3/8MZMmTfrbOVavXs2iRYv4/fffuf/++1m1atVej0lMTOSnn35iw4YN5OXl4fF46NSpE+np6cyYMYNNmzbx3HPPMXfu3Brz3HHHHUycOJGZM2eyYcMG7rrrLtauXcutt956wO9pf0aNGkVwcDCXXnopP//8M0uXLuXmm2/mkksuIS4uji1btnD33XezYsUK0tLSWLx4MRs3bqRr166Ul5dz0003sWzZMtLS0li+fDmrVq3a6x6ouqDi1EBFDRqEzemk4pdfqFi/3nQcERERkSbtzDPPpFmzZmzYsIGRI0fu9blnnnmG6OhoTjrpJAYOHEi/fv32uiK1P+Hh4SxYsIB169Zx7LHHcu+99zJx4sS9HnPttdcydOhQLrzwQvr06cPOnTv3uvoEcPXVV3PkkUeSlJRETEwMy5cv54ILLuC2227jpptuomfPnnzzzTfcf//9Nea55ZZbGDt2LLfffjvdu3dn4cKFzJ8/v1YHMISGhrJo0SJ27dpF7969GT58OGeddRZTp06t/vxvv/3GsGHD6Ny5M9dccw033ngj1157LQ6Hg507dzJ69Gg6d+7MiBEjOPfcc/canlEXbFYTm3ldVFREVFQUhYWFREZGmo5To21jx1L08SdEj7yYlg88YDqOiIiI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\n",
      "text/plain": [
       "<Figure size 1000x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1IAAAJaCAYAAAAlAnbeAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAACQHElEQVR4nOzdd3hUZd7G8e+09GRC6IEUmqDYEEFBxYai0puKvtJUVNbCKqjo6qqICIsN61roKitVbKACohRRQbCgUkMSWmiZSc+U8/4RiEQIJpDkzCT357pyQWbOzNwTUefmec7vWAzDMBAREREREZEys5odQEREREREJNioSImIiIiIiJSTipSIiIiIiEg5qUiJiIiIiIiUk4qUiIiIiIhIOalIiYiIiIiIlJOKlIiIiIiISDmpSImIiIiIiJST3ewAgcDv97Nr1y6io6OxWCxmxxEREREREZMYhkFWVhbx8fFYraWvO6lIAbt27SIhIcHsGCIiIiIiEiDS0tJo3LhxqferSAHR0dFA0Q8rJibG5DQiIiIiImIWt9tNQkJCcUcojYoUFG/ni4mJUZESEREREZG/PeVHwyZERERERETKSUVKRERERESknFSkREREREREysnUc6S+/vpr/vOf/7B27Vp2797N/Pnz6dWrV/H9hmHw73//m7feeovMzEwuuugiXn/9dVq0aFF8zMGDB7nnnnv46KOPsFqt9O3bl5deeomoqKgKzerz+fB4PBX6nCLBwGazYbfbdWkAERERkaOYWqRycnI455xzGDp0KH369Dnm/gkTJjBp0iSmTZtGkyZNeOyxx+jSpQsbN24kLCwMgJtvvpndu3fzxRdf4PF4GDJkCMOGDeO9996rsJzZ2dmkp6djGEaFPadIMImIiKBhw4aEhISYHUVEREQkIFiMAGkHFoulxIqUYRjEx8fzwAMPMHLkSABcLhf169dn6tSp3Hjjjfz222+cccYZfP/995x//vkALFq0iOuuu4709HTi4+PL9Nputxun04nL5Tpmap/P52Pz5s1ERERQt25d/a281CiGYVBYWMi+ffvw+Xy0aNHihBemExEREQl2J+oGRwvY8efbt29nz549dO7cufg2p9PJBRdcwOrVq7nxxhtZvXo1sbGxxSUKoHPnzlitVtasWUPv3r1POYfH48EwDOrWrUt4ePgpP59IsAkPD8fhcLBj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lzZt2jB58mR69OjBr7/+SosWLYqPe/TRR5k4cSItWrTg0UcfZcCAAWzZsgW7/fj/KlitViZNmkSTJk3Ytm0bw4cP58EHH+S1114DiorPlVdeydChQ3nppZew2+0sW7YMn88HwOjRo3nrrbd44YUXuPjii9m9eze///773/58j5abm8v48eN5++23qV27NvXq1WPbtm0MGjSIl19+GcMweO6557juuuvYvHkz0dHR+P1+rr32WrKyspg5cybNmjVj48aN2Gw2IiMjufHGG5kyZUqJInXke5UoqW7yPT62789h276cw4Upm637cti2L5ucQl+Zn8diKdoOd/yLo6oEiYjUNCpS5WTk5fHHeW1Nee2W69ZiiYj42+Nq1arFtddey3vvvVdcpObMmUOdOnW4/PLLATjnnHM455xzih8zZswY5s+fz8KFC7n77rv/9jXee+89/H4/77zzDmFhYbRu3Zr09HTuuuuu4mMcDgdPPvlk8fdNmjRh9erVfPDBB1x//fVERUURHh5OQUHBMdsG//WvfxX/Pjk5mZEjRzJr1qxSi9TEiRN56KGHuPHGGwEYP348y5Yt48UXX+TVV18tPm7kyJF07doVgCeffJLWrVuzZcsWWrVqddznPTJo40iOp59+mjvvvLO4SE2YMIHzzz+/+HuA1q1bA5CVlcVLL73EK6+8wqBBgwBo1qwZF198cSk/1ePzeDy89tprJf55XXHFFSWOefPNN4mNjWX58uV069aNL7/8ku+++47ffvuN0047DYCmTZsWH3/bbbfRsWNHdu/eTcOGDcnIyODTTz/lyy+/LFc2kUBhGAb7sgvYmpHDtv3ZbM04XJr2Z5N+KA+jlAUlm9VCUlwETetG0axuJI3jIoomxR1nS1ykSpCIiBxFRaqauvnmm7n99tt57bXXCA0N5d133+XGG28s3uqXnZ3NE088wSeffMLu3bvxer3k5eWRmppapuf/7bffOPvsswkLCyu+7Xhb5F599VUmT55MamoqeXl5FBYWlmmi3//+9z8mTZrE1q1byc7Oxuv1EhMTc9xj3W43u3bt4qKLLipx+0UXXcSGDRtK3Hb22WcX/75hw4YAZGRklFqkvvzyS8aNG8fvv/+O2+3G6/WSn59Pbm4uERERrF+/nv79+x/3sb/99hsFBQXFZfZkhYSElMgNsHfvXv71r3/x1VdfkZGR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yEnyHjxI1pdLAF07SiRYWUNCaPziC0RfdRWGx0PB5i0aby4iImVS7iKVnJzMU089RWpqamXkEQkargUfgsdDWOvWhJ1xhtlxROQkWUJCaPT8c9S6aQAR7dppvLmIiJRJuYvUiBEjmDdvHk2bNuWqq65i1qxZFBQUVEY2kYBlGAaZc4quHaUhEyLBz+Jw0ODxx0maMV3jzUVEpExOqkitX7+e7777jtNPP5177rmHhg0bcvfdd7Nu3brKyCgScPLWraNw2zYs4eHEdOtqdhwRERERqWInfY7Ueeedx6RJk9i1axf//ve/efvtt2nXrh3nnnsukydPxjCMiswpElAyPzg8ZOLaa7FFRZmcRkRERESqmv1kH+jxeJg/fz5Tpkzhiy++4MILL+TWW28lPT2dRx55hC+//JL33nuvIrOKBASf24178WIAYvv3MzmNiIiIiJih3EVq3bp1TJkyhffffx+r1crAgQN54YUXaNWqVfExvXv3pl27dhUaVCRQuD76CCM/n9AWzQk/91yz44iIiIiICcpdpNq1a8dVV13F66+/Tq9evXA4HMcc06RJE2688cYKCSgSSAzDIHP2n0MmLBaLyYlERERExAzlPkdq27ZtLFq0iP79+x+3RAFERkYyZcqUUw4HkJWVxYgRI0hKSiI8PJyOHTvy/fffA0XbCx966CHOOussIiMjiY+PZ+DAgezatatCXlvkr/J/+ZWC33/HEhKCs0cPs+OIiIiIiEnKXaQyMjJYs2bNMbevWbOGH374oUJCHe22227jiy++YMaMGfz8889cffXVdO7cmZ07d5Kbm8u6det47LHHWLduHfPmzeOPP/6ghz7gSiXJnF00ZCL66quxxcaaG0ZERERETGMxyjler3379jz44IP061fyJPt58+Yxfvz445ask5WXl0d0dDQffvghXbv+OWK6bdu2XHvttTz99NPHPOb777+nffv27Nixg8TExDK9jtvtxul04nK5iImJqbD8Ur34c3LYfEkn/Lm5JE6dSuSFF5gdSUREREQqWFm7QbnPkdq4cSPnnXfeMbe3adOGjRs3lvfpTsjr9eLz+QgLCytxe3h4OCtWrDjuY1wuFxaLhdgTrBYUFBSUuIiw2+2ukLxSvbk/+wx/bi6OpEQiLmhvdhwRERERMVG5t/aFhoayd+/eY27fvXs3dvtJT1M/rujoaDp06MCYMWPYtWsXPp+PmTNnsnr1anbv3n3M8fn5+Tz00EMMGDDghO1x3LhxOJ3O4q+EhIQKzS3V06HD2/pi+/XTkAkRERGRGq7cRerqq69m9OjRuFyu4tsyMzN55JFHuOqqqyo0HMCMGTMwDINGjRoRGhrKpEmTGDBgAFZryegej4frr78ewzB4/fXXT/icR/If+UpLS6vw3FK95P+xifwNP4HdTmzv3mbHERERERGTlXsJaeLEiXTq1ImkpCTatGkDwPr166lfvz4zZsyo8IDNmjVj+fLl5OTk4Ha7adiwITfccANNmzYtPuZIidqxYwdLly792/OcQkNDCQ0NrfCsUn1lzikaeR59+eXY69QxOY2IiIiImK3cRapRo0b89NNPvPvuu2zYsIHw8HCGDBnCgAEDSh2HXhEiIyOJjIzk0KFDLF68mAkTJgB/lqjNmzezbNkyateuXWkZpGbyFxTgWrgQgNj+/f7maBERERGpCU7qpKbIyEiGDRtW0VmOa/HixRiGQcuWLdmyZQujRo2iVatWDBkyBI/HQ79+/Vi3bh0ff/wxPp+PPXv2ABAXF0dISEiVZJTqLevzz/G7XNjjGxJ50UVmxxERERGRAHDS0yE2btxIamoqhYWFJW6v6Gs4uVwuRo8eTXp6OnFxcfTt25exY8ficDhISUlh4eGVgnPPPbfE45YtW8Zll11WoVmkZsr84PCQiT59sdhsJqcRERERkUBQ7utIbdu2jd69e/Pzzz9jsVg48vAjU8x8Pl/Fp6xkuo6UlKZg+3a2XXsdWK00X/IljoYNzY4kIiIiIpWorN2g3FP77rvvPpo0aUJGRgYRERH8+uuvfP3115x//vl89dVXp5JZJOC45s4FIPKSi1WiRERERKRYubf2rV69mqVLl1KnTh2sVitWq5WLL76YcePGce+99/Ljjz9WRk6RKmcUFpI5fwEAtfr3NzeMiIiIiASUcq9I+Xw+oqOjAahTpw67du0CICkpiT/++KNi04mYKGvZV/gOHMBWpw5Rl15qdhwRERERCSDlXpE688wz2bBhA02aNOGCCy5gwoQJhISE8Oabb5a4tpNIsMucfXjIRO/eWCpxtL+IiIiIBJ9yF6l//etf5OTkAPDUU0/RrVs3LrnkEmrXrs3//ve/Cg8oYobC9J3krFwJ6NpRIiIiInKschepLl26FP++efPm/P777xw8eJBatWoVT+4TCXauefPAMIi48EJCEhPNjiMiIiIiAaZc50h5PB7sdju//PJLidvj4uJUoqTaMHw+MufNA7QaJSIiIiLHV64i5XA4SExMDMprRYmUVfY33+Ddsweb00l0585mxxERERGRAFTuqX2PPvoojzzyCAcPHqyMPCKmy5w9BwBnr55YQ0NNTiMiIiIigajc50i98sorbNmyhfj4eJKSkoiMjCxx/7p16yosnEhV82RkkH34wtKxunaUiIiIiJSi3EWqV69elRBDJDC45s0Hn4/wNm0Ibd7c7DgiIiIiEqDKXaT+/e9/V0YOEdMZfj+Zc+cCWo0SERERkRMr9zlSItVV7po1eNLSsEZFEXNNl79/gIiIiIjUWOVekbJarSccda6JfhKsMmfPBiCmW1esEREmpxERERGRQFbuIjV//vwS33s8Hn788UemTZvGk08+WWHBRKqS99Ahsr74EtC2PhERERH5e+UuUj179jzmtn79+tG6dWv+97//ceutt1ZIMJGq5FrwIYbHQ9gZZxDeurXZcUREREQkwFXYOVIXXnghS5YsqainE6kyhmGQOafo2lGx12s1SkRERET+XoUUqby8PCZNmkSjRo0q4ulEqlTejz9SuHUrlvBwYrp1MzuOiIiIiASBcm/tq1WrVolhE4ZhkJWVRUREBDNnzqzQcCJVIfODw0MmrrkGW1SUyWlEREREJBiUu0i98MILJYqU1Wqlbt26XHDBBdSqVatCw4lUNp/bjXvRIkBDJkRERESk7MpdpAYPHlwJMUTM4fr4Y4z8fEKaNyO8zblmxxERERGRIFHuc6SmTJnC7MPX2zna7NmzmTZtWoWEEqkKhmGQObtoyESt/v1PeH00EREREZGjlbtIjRs3jjp16hxze7169XjmmWcqJJRIVcj/dSMFv/2GxeEgpkcPs+OIiIiISBApd5FKTU2lSZMmx9yelJREampqhYQSqQqZh1dWo6++GrvO7xMRERGRcih3kapXrx4//fTTMbdv2LCB2rVrV0gokcrmz8nB/fHHAMT272dyGhEREREJNuUuUgMGDODee+9l2bJl+Hw+fD4fS5cu5b777uPGG2+sjIwiFc69aBH+nBwciYlEtG9vdhwRERERCTLlnto3ZswYUlJSuPLKK7Hbix7u9/sZOHCgzpGSoHHk2lGx/fphsVbIdalFREREpAaxGIZhnMwDN2/ezPr16wkPD+ess84iKSmporNVGbfbjdPpxOVyERMTY3YcqWT5mzaxvUdPsNtpsWwp9rp1zY4kIiIiIgGirN2g3CtSR7Ro0YIWLVqc7MNFTJM5p2jkefTll6lEiYiIiMhJKfeepr59+zJ+/Phjbp8wYQL9+/evkFAilcVfUID7w4VA0bY+EREREZGTUe4i9fXXX3Pdddcdc/u1117L119/XSGhRCpL1udf4HO5sDdsSOTFF5sdR0RERESCVLmLVHZ2NiEhIcfc7nA4cLvdFRJKpLIcuXZUbJ8+WGw2k9OIiIiISLAqd5E666yz+N///nfM7bNmzeKMM86okFAilaEwJYXc774Di4XYvn3MjiMiIiIiQazcwyYee+wx+vTpw9atW7niiisAWLJkCe+99x5zDp/ELxKIMufOBSDykotxxMebnEZEREREglm5i1T37t1ZsGABzzzzDHPmzCE8PJxzzjmHpUuXEhcXVxkZRU6Z4fGQOX8BALEaiiIiIiIip+ikxp937dqVrl27AkVz1t9//31GjhzJ2rVr8fl8FRpQpCJkLVuGb/9+bHXqEH3ZZWbHEREREZEgV+5zpI74+uuvGTRoEPHx8Tz33HNcccUVfPvttxWZTaTCZM4u2nYa27sXFofD5DQiIiIiEuzKtSK1Z88epk6dyjvvvIPb7eb666+noKCABQsWaNCEBCzPzp3krFgB6NpRIiIiIlIxyrwi1b17d1q2bMlPP/3Eiy++yK5du3j55ZcrM5tIhcicNx8Mg4gLLiAkKcnsOCIiIiJSDZR5Reqzzz7j3nvv5a677qJFixaVmUmkwhg+H5nz5gEaMiEiIiIiFafMK1IrVqwgKyuLtm3bcsEFF/DKK6+wf//+yswmcspyVqzAu3s3NqeT6Ks6mx1HRERERKqJMhepCy+8kLfeeovdu3dzxx13MGvWLOLj4/H7/XzxxRdkZWVVZk6Rk3Jo9mwAYnr2wBoaanIaEREREakuyj21LzIykqFDh7JixQp+/vlnHnjgAZ599lnq1atHjx49KiOjyEnxZGSQvewrAGppW5+IiIiIVKCTHn8O0LJlSyZMmEB6ejrvv/9+RWUSqRCu+QvA5yP83HMJ1Xl9IiIiIlKBTqlIHWGz2ejVqxcLFy6siKcTOWWG30/m3LmAhkyIiIiISMWrkCIlEmhyv/sOT2oq1shIYq69xuw4IiIiIlLNqEhJtZT5weEhE926YY2IMDmNiIiIiFQ3KlJS7XgPHSLriy8AbesTERERkcqhIiXVjuvDDzE8HkLPOJ3wM1ubHUdEREREqqGAL1JZWVmMGDGCpKQkwsPD6dixI99//33x/YZh8Pjjj9OwYUPCw8Pp3LkzmzdvNjGxmMkwDDLnzAE08lxEREREKk/AF6nbbruNL774ghkzZvDzzz9z9dVX07lzZ3bu3AnAhAkTmDRpEm+88QZr1qwhMjKSLl26kJ+fb3JyMUPej+sp3LIVS1gYMd26mR1HRERERKqpgC5SeXl5zJ07lwkTJtCpUyeaN2/OE088QfPmzXn99dcxDIMXX3yRf/3rX/Ts2ZOzzz6b6dOns2vXLhYsWGB2fDFB5uzDQyauuQZbdLTJaURERESkugroIuX1evH5fISFhZW4PTw8nBUrVrB9+3b27NlD586di+9zOp1ccMEFrF69uqrjisl8WVm4P/sMgNjrta1PRERERCpPQBep6OhoOnTowJgxY9i1axc+n4+ZM2eyevVqdu/ezZ49ewCoX79+icfVr1+/+L7jKSgowO12l/iS4Of++GOM/HxCmjUjvE0bs+OIiIiISDUW0EUKYMaMGRiGQaNGjQgNDWXSpEkMGDAAq/Xko48bNw6n01n8lZCQUIGJxSyHDm/ri+3fD4vFYnIaEREREanOAr5INWvWjOXLl5OdnU1aWhrfffcdHo+Hpk2b0qBBAwD27t1b4jF79+4tvu94Ro8ejcvlKv5KS0ur1PcglS/v118p2PgbFocDZ8+eZscRERERkWou4IvUEZGRkTRs2JBDhw6xePFievbsSZMmTWjQoAFLliwpPs7tdrNmzRo6dOhQ6nOFhoYSExNT4kuC25EhE9FXdcZeq5bJaURERESkurObHeDvLF68GMMwaNmyJVu2bGHUqFG0atWKIUOGYLFYGDFiBE8//TQtWrSgSZMmPPbYY8THx9OrVy+zo0sV8efm4v7oYwBide0oEREREakCAV+kXC4Xo0ePJj09nbi4OPr27cvYsWNxOBwAPPjgg+Tk5DBs2DAyMzO5+OKLWbRo0TGT/qT6cn+2CH9ODo6EBCIuuMDsOCIiIiJSA1gMwzDMDmE2t9uN0+nE5XJpm18QSrlxAHnr11P3n/+kzh3DzI4jIiIiIkGsrN0gaM6REjmegs2byVu/Hmw2nL17mR1HRERERGoIFSkJaplz5gAQdfllOOrVMzeMiIiIiNQYKlIStPwFBbgWfAhAbL9+JqcRERERkZpERUqCVtYXX+JzubA3aEDUJZeYHUdEREREahAVKQlaR64dFdunDxabzeQ0IiIiIlKTqEhJUCrcsYPcNWvAYiG2bx+z44iIiIhIDaMiJUEpc85cACIvvhhHo0YmpxERERGRmkZFSoKO4fGQOX8+oCETIiIiImIOFSkJOllffYVv/35stWsTffllZscRERERkRpIRUqCTvGQid69sISEmJxGRERERGoiFSkJKp5du8j5ZgWgbX0iIiIiYh4VKQkqmfPmg2EQ0b49IcnJZscRERERkRpKRUqChuHzkTm3aFpfbP/+JqcRERERkZpMRUqCRs7KlXh378bqdBJ99VVmxxERERGRGkxFSoLGkSETzh49sIaGmpxGRERERGoyFSkJCt59+8ha9hUAsf01ZEJEREREzKUiJUEhc/4C8HoJP+ccwk47zew4IiIiIlLDqUhJwDMMg8w5cwCIvV5DJkRERETEfCpSEvBy13yHJzUVa0QEMddcY3YcEREREREVKQl8R4ZMxHTrhjUy0uQ0IiIiIiIqUhLgvIcOkfX554CuHSUiIiIigUNFSgKae+FCDI+H0NNPJ+zM1mbHEREREREBVKQkgJUYMtG/HxaLxeREIiIiIiJFVKQkYOWtX0/B5i1YwsJwdutmdhwRERERkWIqUhKwMmcXrUbFdOmCLSbG5DQiIiIiIn9SkZKA5MvOxv3ZZ4CuHSUiIiIigUdFSgKS++OPMfLyCGnWjPDzzjM7joiIiIhICSpSEpAyPyi6dlRsPw2ZEBEREZHAoyIlASfv11/J37gRHA6cvXqaHUdERERE5BgqUhJwjow8j+58JfZatUxOIyIiIiJyLBUpCSj+3FzcH30MQK3+GjIhIiIiIoFJRUoCinvRYvzZ2TgaNybiwgvNjiMiIiIiclwqUhJQMmcfNWTCqj+eIiIiIhKY9ElVAkbBli3k/fgj2Gw4+/Q2O46IiIiISKlUpCRgZM4uGjIRddllOOrVMzmNiIiIiEjpVKQkIPgLC3F9+CEAsf36mpxGREREROTEVKQkIGR98QW+zEzs9esTdcklZscRERERETkhFSkJCEe29cX27YPFbjc5jYiIiIjIialIiekKU1PJ/fZbsFiI7attfSIiIiIS+FSkxHSZc+YCEHnRRTgaNTI5jYiIiIjI31ORElMZHg+Z8+cBRdeOEhEREREJBipSYqrs5cvx7duPLS6O6CsuNzuOiIiIiEiZqEiJqQ7Nng2As3cvLCEhJqcRERERESkbFSkxjWf3bnK+WQFoW5+IiIiIBBcVKTFN5rx54PcT0a4doU2amB1HRERERKTMVKTEFIbPR+bcoml9sf21GiUiIiIiwUVFSkyRs2oV3l27scbEEH311WbHEREREREpFxUpMUXmB4eHTPTogTUszOQ0IiIiIiLloyIlVc67fz9Zy5YBENu/v8lpRERERETKT0VKqlzm/Png9RJ2ztmEtTzN7DgiIiIiIuUW0EXK5/Px2GOP0aRJE8LDw2nWrBljxozBMIziY7Kzs7n77rtp3Lgx4eHhnHHGGbzxxhsmppYTMQyDzDlzAKil1SgRERERCVJ2swOcyPjx43n99deZNm0arVu35ocffmDIkCE4nU7uvfdeAO6//36WLl3KzJkzSU5O5vPPP2f48OHEx8fTo0cPk9+B/FXud9/j2ZGKNSKCmGuvNTuOiIiIiMhJCegVqVWrVtGzZ0+6du1KcnIy/fr14+qrr+a7774rccygQYO47LLLSE5OZtiwYZxzzjkljpHAkTm7aMhETNeuWCMjTU4jIiIiInJyArpIdezYkSVLlrBp0yYANmzYwIoVK7j2qJWMjh07snDhQnbu3IlhGCxbtoxNmzZx9QlGahcUFOB2u0t8SeXzZWaS9fnnAMRer219IiIiIhK8Anpr38MPP4zb7aZVq1bYbDZ8Ph9jx47l5ptvLj7m5ZdfZtiwYTRu3Bi73Y7VauWtt96iU6dOpT7vuHHjePLJJ6viLchRXAsXYhQWEtqqFWFnnml2HBERERGRkxbQK1IffPAB7777Lu+99x7r1q1j2rRpTJw4kWnTphUf8/LLL/Ptt9+ycOFC1q5dy3PPPcc//vEPvvzyy1Kfd/To0bhcruKvtLS0qng7NZphGGTOLhoyEdu/HxaLxeREIiIiIiInz2IcPQIvwCQkJPDwww/zj3/8o/i2p59+mpkzZ/L777+Tl5eH0+lk/vz5dO3atfiY2267jfT0dBYtWlSm13G73TidTlwuFzExMRX+PgTy1q8n5cYBWEJDafH1cmxOp9mRRERERESOUdZuENArUrm5uVitJSPabDb8fj8AHo8Hj8dzwmMkMBw6MmTimi4qUSIiIiIS9AL6HKnu3bszduxYEhMTad26NT/++CPPP/88Q4cOBSAmJoZLL72UUaNGER4eTlJSEsuXL2f69Ok8//zzJqeXI3zZ2bg//QyAWF07SkRERESqgYAuUi+//DKPPfYYw4cPJyMjg/j4eO644w4ef/zx4mNmzZrF6NGjufnmmzl48CBJSUmMHTuWO++808TkcjT3x59g5OUR0rQp4W3bmh1HREREROSUBfQ5UlVF50hVru19+5H/66/Ue/BBag8dYnYcEREREZFSVYtzpCT45W/cSP6vv4LDgbNXT7PjiIiIiIhUCBUpqVSZc4pGnkdfeSX2uDiT04iIiIiIVAwVKak0/rw8XB99DBRdO0pEREREpLpQkZJK4160GH9WFo7GjYns0MHsOCIiIiIiFUZFSipN5uFrR8X264vFqj9qIiIiIlJ96NOtVIqCrVvJW7cOrFacvfuYHUdEREREpEKpSEmlyJxdNGQi6tJLcdSvZ3IaEREREZGKpSIlFc5fWIhrwQIAYvv3NzeMiIiIiEglUJGSCpf95Zf4MjOx16tHVKdLzI4jIiIiIlLhVKSkwh06PGTC2bcPFrvd5DQiIiIiIhVPRUoqVGFqKrmrvwWLhdi+unaUiIiIiFRPKlJSoTLnzgMgsmNHQho3MjmNiIiIiEjlUJGSCmN4vbjmFRWp2P5ajRIRERGR6ktFSipM9vLlePftwxYXR/QVV5gdR0RERESk0qhISYXJ/ODwkIlevbCEhJicRkRERESk8qhISYXw7NlD9jffABDbT9v6RERERKR6U5GSCpE5bx74/UScfz6hTZuYHUdEREREpFKpSMkpM/x+XHPmAhoyISIiIiI1g4qUnLKclavw7NqFNSaG6C5dzI4jIiIiIlLpVKTklGXOPjxkont3rGFhJqcREREREal8KlJySrz795O1dCkAsdf3NzmNiIiIiEjVUJGSU+JasAC8XsLOPpuwli3NjiMiIiIiUiVUpOSkGYZB5uw5gIZMiIiIiEjNoiIlJy33++8p3LEDS0QEMddeZ3YcEREREZEqoyIlJ+3IapSz63XYoiJNTiMiIiIiUnVUpOSk+DIzyVq8GIDY/hoyISIiIiI1i4qUnBTXwo8wCgsJbdmSsLPOMjuOiIiIiEiVUpGScisaMlF07ajY/v2xWCwmJxIRERERqVoqUlJu+T/9RMHmzVhCQ3F272Z2HBERERGRKqciJeV26PBqVHSXq7E5nSanERERERGpeipSUi6+7Bzcn34GQC0NmRARERGRGkpFSsrF/cknGLm5hDRpQvj555sdR0RERETEFCpSUi7FQyb69dOQCRERERGpsVSkpMzyf/uN/F9+AYcDZ6+eZscRERERETGNipSUWebsOQBEX3EF9tq1TU4jIiIiImIeFSkpE39eHq6PPgKKrh0lIiIiIlKTqUhJmbgXL8aflYWjUSMiO3YwO46IiIiIiKlUpKRMjmzri+3XF4tVf2xEREREpGbTJ2L5WwXbtpG3di1YrTj79DE7joiIiIiI6VSk5G8dWY2K6tQJR/36JqcRERERETGfipSckL+wENeCBQDEXq8hEyIiIiIioCIlfyN7yRJ8hw5hr1ePqE6dzI4jIiIiIhIQVKTkhDJnzwbA2ac3Frvd5DQiIiIiIoFBRUpKVZiWRs6q1QDE9utnchoRERERkcChIiWlypw7F4DIjh0IadzY5DQiIiIiIoFDRUqOy/B6cc2bD0Bsfw2ZEBERERE5moqUHFf211/jzcjAVqsWUVdeaXYcEREREZGAoiIlx5X5weEhE716YQ0JMTmNiIiIiEhgUZGSY3j27CH7668BiO2vIRMiIiIiIn8V0EXK5/Px2GOP0aRJE8LDw2nWrBljxozBMIwSx/3222/06NEDp9NJZGQk7dq1IzU11aTUwc81fz74/YSf35bQpk3NjiMiIiIiEnAC+sJA48eP5/XXX2fatGm0bt2aH374gSFDhuB0Orn33nsB2Lp1KxdffDG33norTz75JDExMfz666+EhYWZnD44GX4/mXOKpvVp5LmIiIiIyPEFdJFatWoVPXv2pGvXrgAkJyfz/vvv89133xUf8+ijj3LdddcxYcKE4tuaNWtW5Vmri5xVq/Hs3Ik1OpqYLl3MjiMiIiIiEpACemtfx44dWbJkCZs2bQJgw4YNrFixgmuvvRYAv9/PJ598wmmnnUaXLl2oV68eF1xwAQsWLDjh8xYUFOB2u0t8SZHM2YeHTHTvjjU83OQ0IiIiIiKBKaCL1MMPP8yNN95Iq1atcDgctGnThhEjRnDzzTcDkJGRQXZ2Ns8++yzXXHMNn3/+Ob1796ZPnz4sX7681OcdN24cTqez+CshIaGq3lJA8x44QNbSpQDEXq9rR4mIiIiIlCagt/Z98MEHvPvuu7z33nu0bt2a9evXM2LECOLj4xk0aBB+vx+Anj178s9//hOAc889l1WrVvHGG29w6aWXHvd5R48ezf3331/8vdvtVpkCXAsWgMdD2FlnEdaqldlxREREREQCVkAXqVGjRhWvSgGcddZZ7Nixg3HjxjFo0CDq1KmD3W7njDPOKPG4008/nRUrVpT6vKGhoYSGhlZq9mBjGAaZs+cAGjIhIiIiIvJ3AnprX25uLlZryYg2m614JSokJIR27drxxx9/lDhm06ZNJCUlVVnO6iDvhx8oTEnBEhFBzOHhHiIiIiIicnwBvSLVvXt3xo4dS2JiIq1bt+bHH3/k+eefZ+jQocXHjBo1ihtuuIFOnTpx+eWXs2jRIj766CO++uor84IHoUOHh0zEXHcttqhIk9OIiIiIiAQ2i/HXq9sGkKysLB577DHmz59PRkYG8fHxDBgwgMcff5yQkJDi4yZPnsy4ceNIT0+nZcuWPPnkk/Ts2bPMr+N2u3E6nbhcLmJiYirjrQQ0n8vF5k6XYhQUkPy/WYSfc47ZkURERERETFHWbhDQRaqq1PQidXDGTPaOHUvoaafR5MMFWCwWsyOJiIiIiJiirN0goM+RkspXNGSiaFtfbP/+KlEiIiIiImWgIlXD5f/8MwWbNmEJCcHZvZvZcUREREREgoKKVA13ZDUquksXbLGx5oYREREREQkSKlI1mC87B9cnnwIQ21/XjhIRERERKSsVqRrM/eknGLm5hCQnE9GundlxRERERESChopUDZY5ew5QtBqlIRMiIiIiImWnIlVD5f/+O/k//wx2O85yXHNLRERERERUpGqsI6tR0Vdcgb1OHZPTiIiIiIgEFxWpGsifn4/ro4+AomtHiYiIiIhI+ahI1UBZixfjd7txxMcTeVFHs+OIiIiIiAQdFaka6NDha0c5+/XFYtUfARERERGR8tKn6BqmYNt28n5YC1YrsX36mB1HRERERCQoqUjVMJlzioZMRF1yCY4GDUxOIyIiIiISnFSkahCjsBDXggUAxF6vIRMiIiIiIidLRaoGyVq6FN/Bg9jr1iXq0kvNjiMiIiIiErRUpGqQzA8OD5no0weL3W5yGhERERGR4KUiVUMUpqeTs2oVALH9+pqcRkREREQkuKlI1RCZc+cCENHhQkISEkxOIyIiIiIS3FSkagDD68U1dx4AtfpryISIiIiIyKlSkaoBsr/+Bm9GBrbYWKI6dzY7joiIiIhI0FORqgEyZx8eMtGrF9aQEJPTiIiIiIgEPxWpas6zdy/Zy5cDENu/n8lpRERERESqBxWpas41fz74/YS3bUtos2ZmxxERERERqRZUpKoxw+8nc07RtL7YflqNEhERERGpKCpS1VjO6tV40tOxRkcTc00Xs+OIiIiIiFQbKlLVWObsOQA4u3fDGh5uchoRERERkepDRaqa8h48SNaSJQDE6tpRIiIiIiIVSkWqmnLNXwAeD2FnnknY6aebHUdEREREpFpRkaqGDMMgc07Rtj4NmRARERERqXgqUtVQ3tq1FG7fjiU8nJhuXc2OIyIiIiJS7ahIVUOZs2cDEHPdtdiiokxOIyIiIiJS/ahIVTM+lwv3osUA1NKQCRERERGRSqEiVc24PvoYo6CA0BYtCDvnHLPjiIiIiIhUSypS1YhhGMXb+mL798NisZicSERERESkelKRqkbyf/mFgj/+wBISgrNHD7PjiIiIiIhUWypS1UjmB0WrUdFXX40tNtbcMCIiIiIi1ZiKVDXhz8nB/cknAMRqyISIiIiISKVSkaomXJ9+ij83l5CkJCLatzM7joiIiIhItaYiVU1kzp4DaMiEiIiIiEhVUJGqBvL/+IP8n34Cux1nr15mxxERERERqfZUpKqBI6tR0Zdfjr1OHZPTiIiIiIhUfypSQc6fn49r4UIAYq/XkAkRERERkaqgIhXksj7/HL/bjT2+IZEdO5odR0RERESkRlCRCnJHrh0V27cvFpvN5DQiIiIiIjWDilQQK9i+ndwffgCrldg+fcyOIyIiIiJSY6hIBbHMOUVDJiIvuRhHw4YmpxERERERqTlUpIKUUViIa/4CAGr115AJEREREZGqpCIVpLKWLsN38CC2unWIuvRSs+OIiIiIiNQoKlJBKnP24SETvftgcThMTiMiIiIiUrMEdJHy+Xw89thjNGnShPDwcJo1a8aYMWMwDOO4x995551YLBZefPHFqg1axQrTd5KzahUAsf36mpxGRERERKTmsZsd4ETGjx/P66+/zrRp02jdujU//PADQ4YMwel0cu+995Y4dv78+Xz77bfEx8eblLbquObNBcMg4sILCUlMNDuOiIiIiEiNE9BF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\n",
      "text/plain": [
       "<Figure size 1000x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final Test Accuracy: 98.25, Final Test Loss: 0.07488370682291973\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## Training\n",
    "\n",
    "def build_resnet(input_channels=6,device='cuda',optimizer='SGD',lr=1e-2):\n",
    "    \"\"\"\n",
    "    Builds a resnet feature extractor based on the number of input channels\n",
    "    \n",
    "    input_channels: int\n",
    "    \"\"\"\n",
    "    assert optimizer in ['SGD','Adam'], f'Optimizer type {optimzer} not supported.'\n",
    "    device = torch.device(device)\n",
    "    model = models.resnet50(pretrained = True)\n",
    "    if input_channels != 3:\n",
    "        \n",
    "        #Pretrained resnet weights from torchvision\n",
    "        pretrained_weights = model.conv1.weight.clone()\n",
    "    \n",
    "        #Replacing initial layer with new conv1, modifying input channels for this dataset\n",
    "        model.conv1 = nn.Conv2d(\n",
    "            in_channels=input_channels,\n",
    "            out_channels=64,\n",
    "            kernel_size=7,\n",
    "            stride=2,\n",
    "            padding=3,\n",
    "            bias=False\n",
    "        )\n",
    "    \n",
    "        #Initialize the new conv1 weights using pretrained weights\n",
    "        with torch.no_grad():\n",
    "            model.conv1.weight[:, :3] = pretrained_weights  #RGB\n",
    "            model.conv1.weight[:, 3:] = pretrained_weights[:, :3] / 3.0  #NDVI, NDBI, Elevation\n",
    "            \n",
    "    num_classes = 2 #yes/no image is a ruin\n",
    "    in_features = model.fc.in_features\n",
    "    model.fc = nn.Linear(in_features,num_classes)\n",
    "    model = model.to(device)\n",
    "\n",
    "    #Adam or SGD+Momentum\n",
    "    if optimizer == 'SGD':\n",
    "        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9)\n",
    "        scheduler = False\n",
    "        \n",
    "    else:\n",
    "        \n",
    "        optimizer = optim.Adam(model.parameters(), lr=lr)\n",
    "        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)\n",
    "    \n",
    "\n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    \n",
    "    return model, optimizer, scheduler, criterion, device\n",
    "\n",
    "\n",
    "def train_resnet(model,optimizer,criterion,device,dataloader):\n",
    "    \"\"\"\n",
    "    resnet training loop, forward and backward pass with weights update\n",
    "\n",
    "    model: torch model\n",
    "    optimizer: torch optimizer\n",
    "    criterion: torch loss function, progress scoring method\n",
    "    \"\"\"\n",
    "    model.train()\n",
    "    total_running_loss = 0\n",
    "    train_running_correct = 0\n",
    "    print('Training')\n",
    "\n",
    "    for x,y in tqdm(dataloader):\n",
    "        \n",
    "        x,y = x.to(device),y.long().to(device)\n",
    "        #reset gradients\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        #forward pass\n",
    "        output = model(x)\n",
    "\n",
    "        #calculating loss\n",
    "        loss = criterion(output,y)\n",
    "    \n",
    "        total_running_loss += loss.item()\n",
    "\n",
    "        _,preds = torch.max(output.data,1)\n",
    "\n",
    "        #correct class predictions\n",
    "        train_running_correct += (preds==y).sum().item()\n",
    "\n",
    "        # Calculate new weights\n",
    "        loss.backward()\n",
    "\n",
    "        # Apply new weights\n",
    "        optimizer.step()\n",
    "        \n",
    "        \n",
    "\n",
    "    train_loss = total_running_loss/len(dataloader)\n",
    "    train_acc =  100.*train_running_correct/len(dataloader.dataset)   \n",
    "    \n",
    "    return train_loss, train_acc\n",
    "\n",
    "\n",
    "def val_resnet(model,optimizer,criterion,device,dataloader,cm=False):\n",
    "    \"\"\"\n",
    "    Validating loop, scoring & performance \n",
    "\n",
    "    model: torch model\n",
    "    optimizer: torch optimizer\n",
    "    criterion: torch loss function, progress scoring method\n",
    "    \"\"\"\n",
    "    \n",
    "    model.eval()\n",
    "    print('Validating')\n",
    "    total_running_loss = 0\n",
    "    val_running_correct = 0\n",
    "    cm_labels = []\n",
    "    cm_preds = []\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        \n",
    "        for x,y in tqdm(dataloader):\n",
    "            \n",
    "            x,y = x.to(device),y.long().to(device)\n",
    "        \n",
    "            #forward pass, no backward for validation\n",
    "            output = model(x).to(device)\n",
    "            \n",
    "            loss = criterion(output,y)\n",
    "            \n",
    "            total_running_loss += loss.item()\n",
    "    \n",
    "            _,preds = torch.max(output.data,1)\n",
    "    \n",
    "            val_running_correct += (preds==y).sum().item()\n",
    "            \n",
    "            #for final test set:\n",
    "            if cm:\n",
    "                cm_preds.extend(preds.cpu().numpy())\n",
    "                cm_labels.extend(y.cpu().numpy())\n",
    "    \n",
    "    \n",
    "    val_loss = total_running_loss/len(dataloader)\n",
    "    val_acc =  100.*val_running_correct/len(dataloader.dataset)                   \n",
    "\n",
    "    #for final test set:\n",
    "    if cm:\n",
    "        cm = confusion_matrix(cm_labels,cm_preds)\n",
    "        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])\n",
    "        return val_loss, val_acc, display\n",
    "    else:\n",
    "        return val_loss, val_acc\n",
    "\n",
    "     \n",
    "    \n",
    "\n",
    "# Full Run Through\n",
    "if __name__ == '__main__':\n",
    "    device='cuda'\n",
    "    train_loss, val_loss = [],[]\n",
    "    train_acc, val_acc = [],[]\n",
    "    \n",
    "    \n",
    "    model, optimizer, scheduler, criterion, device = build_resnet(input_channels=6,device=device, optimizer='SGD',lr=1e-2)\n",
    "\n",
    "    \n",
    "    #Initial val to see what pretrained resnet can do:\n",
    "    initial_val_loss, initial_val_acc = val_resnet(model,optimizer,criterion,device,val)\n",
    "    print(f\"Initial Validation Loss: {initial_val_loss}, Initial Validation Accuracy: {initial_val_acc}\")\n",
    "    \n",
    "    epochs = 5\n",
    "    base_val_acc = 0\n",
    "    for epoch in range(epochs):\n",
    "        train_loss_epoch, train_acc_epoch = train_resnet(model,optimizer,criterion,device,train)\n",
    "        val_loss_epoch, val_acc_epoch = val_resnet(model,optimizer,criterion,device,val)\n",
    "        train_loss.append(train_loss_epoch)\n",
    "        train_acc.append(train_acc_epoch)\n",
    "        val_loss.append(val_loss_epoch)\n",
    "        val_acc.append(val_acc_epoch)\n",
    "\n",
    "        #updating the saved weights to get the highest accuracy model if it begins to overfit by the end of 5 epochs\n",
    "        if val_acc_epoch > base_val_acc:\n",
    "            best_model_state = model.state_dict()\n",
    "            base_val_epoch = val_acc_epoch\n",
    "            \n",
    "        if scheduler:\n",
    "            scheduler.step(val_loss_epoch)\n",
    "     \n",
    "        print(f\"Epoch {epoch+1} of {epochs}: Train Accuracy: {train_acc_epoch}, Train Loss:{train_loss_epoch}, Validation Accuracy: {val_acc_epoch}, Validation Loss: {val_loss_epoch}\")\n",
    "    \n",
    "    #Test run\n",
    "    test_loss, test_acc, cm_display = val_resnet(model,optimizer,criterion,device,test,cm=True)\n",
    "    \n",
    "    \n",
    "    display_plots(train_acc, val_acc, train_loss, val_loss)\n",
    "    print(f\"Final Test Accuracy: {test_acc}, Final Test Loss: {test_loss}\")\n",
    "    cm_display.plot(cmap='Blues')\n",
    "    plt.title(\"Confusion Matrix on Test Set\")\n",
    "    plt.show()\n",
    "    \n",
    "\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-06-28T19:27:06.098329Z",
     "iopub.status.busy": "2025-06-28T19:27:06.097621Z",
     "iopub.status.idle": "2025-06-28T19:27:06.252980Z",
     "shell.execute_reply": "2025-06-28T19:27:06.252212Z",
     "shell.execute_reply.started": "2025-06-28T19:27:06.098304Z"
    }
   },
   "outputs": [],
   "source": [
    "# To Directory\n",
    "torch.save(best_model_state, \"resnet50_6ch_feature_extractor.pth\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-07-04T00:17:45.339635Z",
     "iopub.status.busy": "2025-07-04T00:17:45.338971Z",
     "iopub.status.idle": "2025-07-04T00:17:45.344231Z",
     "shell.execute_reply": "2025-07-04T00:17:45.343508Z",
     "shell.execute_reply.started": "2025-07-04T00:17:45.339609Z"
    }
   },
   "outputs": [],
   "source": [
    "# Building Feature Extractor from Pretrained Weights #\n",
    "def prepare_feature_extractor_from_state_dict(state_dict, device='cuda'):\n",
    "    \"\"\"\n",
    "    Creating the feature extractor based on the previously trained weights\n",
    "    \n",
    "    state_dict: Previsouly trained weights, torch tensor.\n",
    "    device: device the environment is running on, str.\n",
    "    \"\"\"\n",
    "    model = models.resnet50(pretrained=False)\n",
    "    \n",
    "\n",
    "    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)\n",
    "    \n",
    "\n",
    "    in_features = model.fc.in_features\n",
    "    model.fc = nn.Linear(in_features, 2)\n",
    "    \n",
    "    # Loading saved weights\n",
    "    model.load_state_dict(state_dict)\n",
    "    \n",
    "    # Removing final classification layer to make it a feature extractor\n",
    "    feature_extractor = nn.Sequential(*list(model.children())[:-1])\n",
    "    feature_extractor = feature_extractor.to(device)\n",
    "    feature_extractor.eval()\n",
    "    \n",
    "    return feature_extractor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-07-04T00:17:48.759793Z",
     "iopub.status.busy": "2025-07-04T00:17:48.759211Z",
     "iopub.status.idle": "2025-07-04T00:17:48.766156Z",
     "shell.execute_reply": "2025-07-04T00:17:48.765386Z",
     "shell.execute_reply.started": "2025-07-04T00:17:48.759768Z"
    }
   },
   "outputs": [],
   "source": [
    "def extract_features(feature_extractor, dataloader, device='cuda'):\n",
    "    \"\"\"\n",
    "    Function to extraxt (N,2048) features from (Num_Batches, Batch_Size, 6, 224, 224) dataset\n",
    "    \n",
    "    feature_extractor: a torch model object\n",
    "    \n",
    "    dataloader: the dataset, torch dataloader object\n",
    "    \"\"\"\n",
    "    features_list = []\n",
    "    labels_list = []  \n",
    "\n",
    "    with torch.no_grad():\n",
    "        for batch in dataloader:\n",
    "            \n",
    "            #For unlabeled data (potential ruins)\n",
    "            if isinstance(batch, (tuple, list)) and len(batch) == 1:\n",
    "                x = batch[0]\n",
    "                y = None\n",
    "            #For labeled datasets x & y with or without the coords\n",
    "            elif isinstance(batch, (tuple, list)):\n",
    "                x, y = batch[0], batch[1]\n",
    "                \n",
    "            else:\n",
    "                x = batch\n",
    "                y = None\n",
    "\n",
    "            if x.dim() == 3:\n",
    "                x = x.unsqueeze(0)\n",
    "\n",
    "            x = x.to(device)\n",
    "            feats = feature_extractor(x)          # [B, 2048, 1, 1]\n",
    "            feats = feats.view(feats.size(0), -1) #Flattened to [B, 2048]\n",
    "            features_list.append(feats.cpu())\n",
    "\n",
    "            if y is not None:\n",
    "                y = y.cpu()\n",
    "                if y.dim() == 0:\n",
    "                    y = y.unsqueeze(0)\n",
    "                labels_list.append(y)\n",
    "\n",
    "    \n",
    "    if features_list and labels_list:\n",
    "        return torch.cat(features_list), torch.cat(labels_list)\n",
    "    elif features_list:\n",
    "        return torch.cat(features_list), None\n",
    "    else:\n",
    "        #Return None if blank items get passed through\n",
    "        return None, None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-07-04T00:29:03.041501Z",
     "iopub.status.busy": "2025-07-04T00:29:03.041227Z",
     "iopub.status.idle": "2025-07-04T00:29:33.476515Z",
     "shell.execute_reply": "2025-07-04T00:29:33.475682Z",
     "shell.execute_reply.started": "2025-07-04T00:29:03.041482Z"
    }
   },
   "outputs": [],
   "source": [
    "#Reloading Again to build features\n",
    "\n",
    "\n",
    "# Ensuring no Residual RAM \n",
    "keep_vars = {'torch', 'gc', 'np', 'pd','best_model_state','best_state','nne_state','rgb_state'}\n",
    "for name in dir():\n",
    "    if not name.startswith(\"_\") and name not in keep_vars:\n",
    "        del globals()[name]\n",
    "        keep_vars = {'torch', 'gc', 'np', 'pd','best_state','best_model_state','nne_state','rgb_state'}\n",
    "\n",
    "# Device \n",
    "device = 'cuda'\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()\n",
    "\n",
    "# Paths \n",
    "pt1 = '/kaggle/input/amazon-chunk-data/tile_chunk_1.pt'\n",
    "pt2 = '/kaggle/input/amazon-chunk-data/tile_chunk_2.pt'\n",
    "pt3 = '/kaggle/input/amazon-chunk-data/tile_chunk_3.pt'\n",
    "pt4 = '/kaggle/input/amazon-chunk-data/tile_chunk_4.pt'\n",
    "pt5 = '/kaggle/input/amazon-chunk-data/tile_chunk_5.pt'\n",
    "pt6 = '/kaggle/input/amazon-chunk-data/tile_chunk_6.pt'\n",
    "pt7 = '/kaggle/input/amazon-chunk-data/tile_chunk_7.pt'\n",
    "pt8 = '/kaggle/input/amazon-chunk-data/tile_chunk_8.pt'\n",
    "pt9 = '/kaggle/input/amazon-chunk-data/tile_chunk_9.pt'\n",
    "pt10 = '/kaggle/input/amazon-chunk-data/tile_chunk_10.pt'\n",
    "\n",
    "plf1 = '/kaggle/input/ruins-true-negatives/labeled_false_ruins.pt'\n",
    "plf2 = '/kaggle/input/labeled-false-ruins-2/labeled_false_ruins_2.pt'\n",
    "\n",
    "potential_ruins_1 = '/kaggle/input/potential-ruins/potential_ruins.pt'\n",
    "potential_ruins_2 = '/kaggle/input/tp-chunk-2/tile_chunk_tp_2.pt'\n",
    "\n",
    "# Loading \n",
    "t1 = torch.load(pt1,weights_only=False)\n",
    "t2 = torch.load(pt2,weights_only=False)\n",
    "t3 = torch.load(pt3,weights_only=False)\n",
    "t4 = torch.load(pt4,weights_only=False)\n",
    "t5 = torch.load(pt5,weights_only=False)\n",
    "t6 = torch.load(pt6,weights_only=False)\n",
    "t7 = torch.load(pt7,weights_only=False)\n",
    "t8 = torch.load(pt8,weights_only=False)\n",
    "t9 = torch.load(pt9,weights_only=False)\n",
    "t10 = torch.load(pt10,weights_only=False)\n",
    "f1 = torch.load(plf1,weights_only=False)\n",
    "f2 = torch.load(plf2,weights_only=False)\n",
    "\n",
    "# Features & Coordinates \n",
    "t1, t1_coord = torch.cat(t1['batches']),np.concatenate(t1['coords'])\n",
    "t2, t2_coord = torch.cat(t2['batches']),np.concatenate(t2['coords'])\n",
    "t3, t3_coord = torch.cat(t3['batches']),np.concatenate(t3['coords'])\n",
    "t4, t4_coord = torch.cat(t4['batches']),np.concatenate(t4['coords'])\n",
    "t5, t5_coord = torch.cat(t5['batches']),np.concatenate(t5['coords'])\n",
    "t6, t6_coord = torch.cat(t6['batches']),np.concatenate(t6['coords'])\n",
    "t7, t7_coord = torch.cat(t7['batches']),np.concatenate(t7['coords'])\n",
    "t8, t8_coord = torch.cat(t8['batches']),np.concatenate(t8['coords'])\n",
    "t9, t9_coord = torch.cat(t9['batches']),np.concatenate(t9['coords'])\n",
    "t10, t10_coord = torch.cat(t10['batches']),np.concatenate(t10['coords'])\n",
    "f1, f1_coord = torch.cat(f1['batches']),np.concatenate(f1['coords'])\n",
    "f2, f2_coord = torch.cat(f2['batches']),np.concatenate(f2['coords'])\n",
    "\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()\n",
    "\n",
    "\n",
    "p1 = torch.load(potential_ruins_1,weights_only=False)\n",
    "p2 = torch.load(potential_ruins_2,weights_only=False)\n",
    "\n",
    "p1, p1_coord = torch.cat(p1['batches']),np.concatenate(p1['coords'])\n",
    "p2, p2_coord = torch.cat(p2['batches']),np.concatenate(p2['coords'])\n",
    "\n",
    "pruins = torch.cat((p1,p2))\n",
    "pcoord = np.concatenate((p1_coord,p2_coord))\n",
    "\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()\n",
    "\n",
    "# Dataset \n",
    "true_ruins_batches = torch.cat((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10))\n",
    "true_ruins_coords = np.concatenate((\n",
    "                                      t1_coord, t2_coord,\n",
    "                                      t3_coord, t4_coord,\n",
    "                                      t5_coord, t6_coord,\n",
    "                                      t7_coord, t8_coord,\n",
    "                                      t9_coord, t10_coord))\n",
    "\n",
    "\n",
    "false_ruins_batches = torch.cat((f1, f2))\n",
    "false_ruins_coords = np.concatenate((f1_coord, f2_coord))\n",
    "\n",
    "\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.status.busy": "2025-07-04T00:19:24.797276Z",
     "iopub.status.idle": "2025-07-04T00:19:24.797631Z",
     "shell.execute_reply": "2025-07-04T00:19:24.797473Z",
     "shell.execute_reply.started": "2025-07-04T00:19:24.797457Z"
    }
   },
   "outputs": [],
   "source": [
    "#Restructuring coords\n",
    "true_coords = torch.tensor([[d['lat'], d['lon']] for d in true_ruins_coords])\n",
    "false_coords = torch.tensor([[d['lat'], d['lon']] for d in false_ruins_coords])\n",
    "potential_coords = torch.tensor([[d['lat'], d['lon']] for d in pcoord])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.status.busy": "2025-07-04T00:19:24.798875Z",
     "iopub.status.idle": "2025-07-04T00:19:24.799173Z",
     "shell.execute_reply": "2025-07-04T00:19:24.799020Z",
     "shell.execute_reply.started": "2025-07-04T00:19:24.799008Z"
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# Model state in this case was already defined in the environment\n",
    "best_state = best_model_state\n",
    "#sanitation check\n",
    "print(len(best_state))\n",
    "\n",
    "# Building resnet feature extractor from the traiend weights\n",
    "feature_extractor = prepare_feature_extractor_from_state_dict(best_state, device='cuda')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-06-25T00:47:33.493091Z",
     "iopub.status.busy": "2025-06-25T00:47:33.492836Z",
     "iopub.status.idle": "2025-06-25T00:48:30.715748Z",
     "shell.execute_reply": "2025-06-25T00:48:30.715010Z",
     "shell.execute_reply.started": "2025-06-25T00:47:33.493074Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_35/1568767552.py:11: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
      "  true_feat_tensor, true_lab_tensor = torch.tensor(true_features), torch.tensor(true_labels)\n",
      "/tmp/ipykernel_35/1568767552.py:12: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
      "  false_feat_tensor, false_lab_tensor = torch.tensor(false_features), torch.tensor(false_labels)\n",
      "/tmp/ipykernel_35/1568767552.py:13: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
      "  potential_feat_tensor = torch.tensor(potential_features)\n"
     ]
    }
   ],
   "source": [
    "#Building labels, creating dataset objects\n",
    "true_ruins = Tensor_Dataset(data=true_ruins_batches,labels=torch.ones(len(true_ruins_batches)),coord=true_ruins_coords)\n",
    "false_ruins = Tensor_Dataset(data=false_ruins_batches,labels=torch.zeros(len(false_ruins_batches)),coord=false_ruins_coords)\n",
    "potential_ruins = Tensor_Dataset(data=pruins,labels=None,coord=pcoord)\n",
    "\n",
    "\n",
    "\n",
    "# Feature extraction\n",
    "true_features, true_labels = extract_features(feature_extractor, true_ruins, device='cuda')\n",
    "false_features, false_labels = extract_features(feature_extractor, false_ruins, device='cuda')\n",
    "potential_features, _ = extract_features(feature_extractor, potential_ruins, device='cuda')\n",
    "true_feat_tensor, true_lab_tensor = torch.tensor(true_features), torch.tensor(true_labels)\n",
    "false_feat_tensor, false_lab_tensor = torch.tensor(false_features), torch.tensor(false_labels)\n",
    "potential_feat_tensor = torch.tensor(potential_features)\n",
    "\n",
    "# Saving\n",
    "del true_ruins, false_ruins\n",
    "if device == 'cuda':\n",
    "    torch.cuda.empty_cache()\n",
    "else: \n",
    "    gc.collect()\n",
    "    \n",
    "torch.save({\n",
    "    'features': true_feat_tensor,\n",
    "    'labels': true_lab_tensor,\n",
    "    'coords': true_coords\n",
    "}, 'resnet_true_features_train.pt')\n",
    "\n",
    "torch.save({\n",
    "    'features': false_feat_tensor,\n",
    "    'labels': false_lab_tensor,\n",
    "    'coords': false_coords\n",
    "}, 'resnet_false_features_test.pt')\n",
    "\n",
    "torch.save({\n",
    "    'features': potential_feat_tensor,\n",
    "    'coords': potential_coords,\n",
    "}, 'resnet_potential_features_val.pt')\n"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [
    {
     "datasetId": 7557028,
     "sourceId": 12012202,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7618694,
     "sourceId": 12101756,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7618912,
     "sourceId": 12102161,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7618924,
     "sourceId": 12102180,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7619186,
     "sourceId": 12102595,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7658781,
     "sourceId": 12160656,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7662310,
     "sourceId": 12165862,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7677965,
     "sourceId": 12189691,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7707990,
     "sourceId": 12233552,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31040,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
