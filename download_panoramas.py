# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import os

# %%
csv_path = "csv/pol_pd_2020_1km_ASCII_XYZ.csv"
# %%
api_key = open('api_key.txt').read()
api_key[:3]


# %%
def sample_population_points(csv_path, n_points):
    """
    Reutrns list of size n_points
    With points randomly selected according to
    WorldPop CSV distribution (X, Y, Z)
    CSV link: https://hub.worldpop.org/geodata/summary?id=43238
    """
    df = pd.read_csv(csv_path)

    df = df[df["Z"] > 0].reset_index(drop=True)

    weights = df["Z"].values
    probs = weights / weights.sum()

    idx = np.random.choice(len(df), size=n_points, p=probs, replace=True)

    xs = df.loc[idx, "X"].values
    ys = df.loc[idx, "Y"].values

    unique_x = np.sort(df["X"].unique())
    unique_y = np.sort(df["Y"].unique())
    dx = np.min(np.diff(unique_x))
    dy = np.min(np.diff(unique_y))

    xs = xs + (np.random.rand(n_points) - 0.5) * dx
    ys = ys + (np.random.rand(n_points) - 0.5) * dy

    return list(zip(ys, xs))




# %%
def snap_location(loc, api_key=api_key, radius=30000):
    """
    Snap a location to the nearest available Street View panorama.
    
    Parameters:
    -----------
    loc : tuple
        (lat, lon) coordinates to snap
    api_key : str
        Google Maps API key
    radius : int
        Search radius in meters (default: 30000)
    
    Returns:
    --------
    dict or None
        Dictionary containing:
        - 'location': tuple (lat, lon) of snapped position
        - 'pano_id': panorama ID
        - 'date': date the photo was taken (if available)
        - 'copyright': copyright information
        Returns None if no panorama found within radius
    """
    lat, lon = loc
    
    metadata_url = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={lat},{lon}"
        f"&radius={radius}"
        f"&source=outdoor"
        f"&key={api_key}"
    )
    
    try:
        response = requests.get(metadata_url, timeout=10)
        
        if response.status_code != 200:
            print(f"Metadata request failed with status {response.status_code}")
            return None
        
        data = response.json()
        
        # Check if panorama exists
        if data.get('status') != 'OK':
            print(f"No Street View imagery found within {radius}m of ({lat}, {lon})")
            return None
        
        # Extract snapped location
        snapped_location = data.get('location', {})
        snapped_lat = snapped_location.get('lat')
        snapped_lon = snapped_location.get('lng')
        
        if snapped_lat is None or snapped_lon is None:
            print("Location data missing in metadata response")
            return None
        
        result = {
            'location': (snapped_lat, snapped_lon),
            'pano_id': data.get('pano_id'),
            'date': data.get('date')
            #'copyright': data.get('copyright')
        }
        
        return result
        
    except Exception as e:
        print(f"Error retrieving metadata: {e}")
        return None


# %%
def download_streetview_image(
        loc,#(lat,lon)
        filename,
        size_px=640,
        heading=0,
        pitch=0,
        fov=90,
        api_key=api_key,
        source="outdoor",
        base_url="https://maps.googleapis.com/maps/api/streetview"
    ):
    """
    loc (lat, lon)
    filename
    size _px       – output image size e.g. 640
    heading     – 0-360
    pitch       – -90 to 90https://hub.worldpop.org/geodata/summary?id=43238
    fov         – 1 to 120
    api_key
    source - limits searches to selected source, default/outdoor
    base_url    – StreetView endpoint
    """

    lat, lon = loc

    url = (
        f"{base_url}"
        f"?size={size_px}x{size_px}"
        f"&location={lat},{lon}"
        f"&heading={heading}"
        f"&pitch={pitch}"
        f"&fov={fov}"
        f"&source={source}"
        f"&key={api_key}"
    )

    #filename = os.path.join(output_dir, f"streetview_{i}.jpg")

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"Received status {response.status_code} for point: {lat}, {lon}")
            return None

        with open(filename, "wb") as f:
            f.write(response.content)

    except Exception as e:
        print("Error code:", e)
        return None

    return filename


# %%
#headings and pitches
POLAR_NAMED = {f"{heading}_{pitch}": [heading, pitch] for heading, pitch in ([[h, 0] for h in range(0,360, 90)] + [[0,90]] + [[0,-90]])}
print(POLAR_NAMED)

# %%
P = lambda *a, **kw: print(a, kw)

# %%
import json
def download_panorama(
        loc,#(lat, lon),
        out_dir,
        ordinal=0,
        size_px=640,
        api_key=api_key,
        source="outdoor",
        base_url="https://maps.googleapis.com/maps/api/streetview"
        ):
    print(f'Download to {out_dir} sample  {ordinal} @ ',loc, end="")
    loc_metadata = snap_location(loc, api_key=api_key)
    if loc_metadata is None:
         print('\nMetadata request failed @', loc)
         return
    snapped_loc = loc_metadata['location']
    print(' -> ', snapped_loc, end="\t")

    loc_metadata['size_px'] = size_px
    
    if not os.path.isdir(out_dir) and os.path.exists(out_dir):
            raise RuntimeError("out_dir "+out_dir + " is not a directory")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(out_dir + "/" + f"stvv_{ordinal}_position.json", "w") as fp:
         json.dump(loc_metadata, fp)
         
    for name_suffix, (heading, pitch) in POLAR_NAMED.items():
        print((heading, pitch), end=",")
        
        
        download_streetview_image(
            snapped_loc,
            out_dir + "/" + f"stvv_{ordinal}_({name_suffix}).jpg",
            heading=heading,
            pitch = pitch,
            size_px=size_px,
            api_key=api_key,
            source = source,
            base_url = base_url,
            fov=90
        )
    print()


# %%
LETTERS_POLARS = {
    "F" : (0,0),
    "R" : (90,0),
    "B" : (180,0),
    "L" : (270, 0),
    "U" : (0, 90),
    "D" : (0,-90)
}
POLARS_LETTERS = {v:k for k,v in LETTERS_POLARS.items()}
POLARS_LETTERS

# %%
from PIL import Image

def get_panorama(dataset_dir, ordinal):
    with open(f"{dataset_dir}/{ordinal}/stvv_{ordinal}_position.json", "r") as fp:
         res = json.load(fp)
    res['skybox'] = {}
    for name_suffix, (heading, pitch) in POLAR_NAMED.items():
        img = Image.open(f"{dataset_dir}/{ordinal}/stvv_{ordinal}_({name_suffix}).jpg")
        np_img = np.array(img)
        res['skybox'][POLARS_LETTERS[(heading, pitch)]] = np_img
    return res
        


# %%
from PIL import Image
from py360convert import c2e, e2p
import piexif
from piexif import GPSIFD


# %%
def get_next_ordinal(dataset_dir):
    """Find the highest ordinal number in existing subdirectories"""
    if not os.path.exists(dataset_dir):
        return 0
    
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    ordinals = []
    
    for subdir in subdirs:
        try:
            ordinals.append(int(subdir))
        except ValueError:
            continue
    
    if not ordinals:
        return 0
    
    return max(ordinals) + 1

# %%
import time
def download_n_panoramas(n, dataset_dir, size_px=640, api_key=api_key):
    """Download n panoramas, continuing from the highest existing ordinal"""
    start_ordinal = get_next_ordinal(dataset_dir)
    print(f"Starting from ordinal {start_ordinal}")
    #basic throttling
    time.sleep(0.2+np.random.rand()/10)
    points = sample_population_points(csv_path, n)
    
    for i, point in enumerate(points):
        ordinal = start_ordinal + i
        subdir = os.path.join(dataset_dir, str(ordinal))
        download_panorama(point, subdir, ordinal=ordinal, size_px=size_px, api_key=api_key)
    
    print(f"\nDownloaded {n} panoramas (ordinals {start_ordinal} to {start_ordinal + n - 1})")

# %%
def decimal_to_dms(decimal):
    """Convert decimal degrees to degrees, minutes, seconds"""
    is_positive = decimal >= 0
    decimal = abs(decimal)
    
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = (decimal - degrees - minutes / 60) * 3600
    
    return (degrees, 1), (minutes, 1), (int(seconds * 100), 100)

def add_gps_exif(image_path, lat, lon):
    """Add GPS EXIF data to a JPEG image"""
    try:
        img = Image.open(image_path)
        
        # Create EXIF data
        exif_dict = {"GPS": {}}
        
        # Latitude
        exif_dict["GPS"][GPSIFD.GPSLatitude] = decimal_to_dms(abs(lat))
        exif_dict["GPS"][GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
        
        # Longitude
        exif_dict["GPS"][GPSIFD.GPSLongitude] = decimal_to_dms(abs(lon))
        exif_dict["GPS"][GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
        
        # Convert to bytes
        exif_bytes = piexif.dump(exif_dict)
        
        # Save with EXIF
        img.save(image_path, "jpeg", exif=exif_bytes)
        
    except Exception as e:
        print(f"Warning: Could not add GPS EXIF to {image_path}: {e}")

# %%
def generate_augmented_views(
        dataset_dir,
        output_dir=None,
        views_per_pano=10,
        size_px=640,
        fov=90
    ):
    """Generate augmented views from all panoramas in dataset_dir"""
    
    if output_dir is None:
        output_dir = dataset_dir + "_augmented"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all panorama ordinals
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    ordinals = []
    for subdir in subdirs:
        try:
            ordinals.append(int(subdir))
        except ValueError:
            continue
    
    ordinals.sort()
    print(f"Found {len(ordinals)} panoramas to augment")
    
    total_views = 0
    
    for ordinal in ordinals:
        print(f"Processing panorama {ordinal}...", end=" ")
        
        # Load panorama
        try:
            pano = get_panorama(dataset_dir, ordinal)
        except Exception as e:
            print(f"Error loading panorama {ordinal}: {e}")
            continue
        
        lat, lon = pano['location']
        
        # Convert to equirectangular
        equirectangular = c2e(
            pano['skybox'],
            2 * pano['size_px'],
            4 * pano['size_px'],
            cube_format='dict'
        )
        
        # Generate random views
        for view_idx in range(views_per_pano):
            # Random parameters
            u_deg = np.random.uniform(-180, 180)
            v_deg = np.random.uniform(-30, 30)
            in_rot_deg = np.random.uniform(-15, 15)
            
            # Generate perspective view
            view = e2p(
                equirectangular,
                fov_deg=fov,
                u_deg=u_deg,
                v_deg=v_deg,
                out_hw=(size_px, size_px),
                in_rot_deg=in_rot_deg,
                mode="bilinear"
            )
            
            # Save image
            filename = f"view_lat{lat:.6f}_lon{lon:.6f}_{ordinal}_{view_idx}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            Image.fromarray(view.astype(np.uint8)).save(filepath, "JPEG", quality=95)
            
            # Add GPS EXIF
            add_gps_exif(filepath, lat, lon)
            
            total_views += 1
        
        print(f"Generated {views_per_pano} views")
    
    print(f"\nTotal augmented views generated: {total_views}")
    print(f"Saved to: {output_dir}")

# %% [markdown]
# ## Usage Example

# %%
# Configuration
DATASET_DIR = "panorama_dataset"
N_PANORAMAS = 100
SIZE_PX = 640

# Step 1: Download panoramas
print("=" * 60)
print("STEP 1: Downloading panoramas")
print("=" * 60)
download_n_panoramas(N_PANORAMAS, DATASET_DIR, size_px=SIZE_PX, api_key=api_key)



# %%
# Step 2: Generate augmented views
VIEWS_PER_PANO = 10
print("\n" + "=" * 60)
print("STEP 2: Generating augmented views")
print("=" * 60)
generate_augmented_views(
    DATASET_DIR,
    output_dir=DATASET_DIR + "_augmented",
    views_per_pano=VIEWS_PER_PANO,
    size_px=SIZE_PX,
    fov=90
)

# %%
# If you want to run again (will add more panoramas without overwriting)
# download_n_panoramas(N_PANORAMAS, DATASET_DIR, size_px=SIZE_PX, api_key=api_key)
# generate_augmented_views(DATASET_DIR, views_per_pano=VIEWS_PER_PANO, size_px=SIZE_PX)
