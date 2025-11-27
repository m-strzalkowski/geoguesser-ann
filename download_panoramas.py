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
api_key = open('api_key.txt').read()
api_key[:3]


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
r=snap_location((50,20), api_key=api_key)
r


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
#points = sample_population_points(csv_path, 1)
#points[0]
# p = (50,20)
# download_panorama(p, "panorama_test/0", ordinal=0)
# p = (50,19)
# download_panorama(p, "panorama_test/1", ordinal=1)

# %% [markdown]
# ### Konwersja:

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
pano = get_panorama("panorama_test", 0)
pano

# %%
from py360convert import c2e, e2p
import matplotlib.pyplot as plt
equirectangular=c2e(pano['skybox'], 2*pano['size_px'], 4*pano['size_px'], cube_format='dict')
plt.imshow(equirectangular)

# %%

    # Parameters
    # ----------
    # e_img: ndarray
    #     Equirectangular image in shape of [H,W] or [H, W, *].
    # fov_deg: scalar or (scalar, scalar) field of view in degree
    #     Field of view given in float or tuple (h_fov_deg, v_fov_deg).
    # u_deg:   horizon viewing angle in range [-180, 180]
    #     Horizontal viewing angle in range [-pi, pi]. (- Left / + Right).
    # v_deg:   vertical viewing angle in range [-90, 90]
    #     Vertical viewing angle in range [-pi/2, pi/2]. (- Down/ + Up).
    # out_hw: tuple[int, int]
    #     Size of output perspective image.
    # in_rot_deg: float
    #     Inplane rotation.
    # mode: Literal["bilinear", "nearest"]
    #     Interpolation mode.
view_side_px = 640
view = e2p(equirectangular,90, 90,0, (view_side_px, view_side_px), 20, "bilinear")
plt.imshow(view)

# %%
# from shapely.geometry import Point
# import matplotlib.pyplot as plt
# import folium


# points = sample_population_points(csv_path, 1000)

# # center map on the mean location
# center_lat = np.mean([lat for lat, lon in points])
# center_lon = np.mean([lon for lat, lon in points])

# m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

# for lat, lon in points:
#     folium.CircleMarker(
#         location=[lat, lon],
#         radius=3,
#         color="red",
#         fill=True,
#         fill_opacity=0.7,
#     ).add_to(m)
# m

# %%

# %%
def download_streetview_images(
        points,
        output_dir="streetview_images",
        size="640x640",
        heading=0,
        pitch=0,
        fov=90,
        api_key=api_key,
        source="outdoor",
        base_url="https://maps.googleapis.com/maps/api/streetview"
    ):
    """
    points      – list (lat, lon)
    output_dir  – output directory path
    size        – output image size, e.g. "640x640"
    heading     – 0-360
    pitch       – -90 to 90https://hub.worldpop.org/geodata/summary?id=43238
    fov         – 1 to 120
    api_key
    source - limits searches to selected source, default/outdoor
    base_url    – StreetView endpoint
    """

    saved_images = []

    for i, (lat, lon) in enumerate(points):

        url = (
            f"{base_url}"
            f"?size={size}"
            f"&location={lat},{lon}"
            f"&heading={heading}"
            f"&pitch={pitch}"
            f"&fov={fov}"
            f"&source={source}"
            f"&key={api_key}"
            f"&radius=10000"
        )

        filename = os.path.join(output_dir, f"streetview_{i}.jpg")

        try:
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                print(f"Received status {response.status_code} for point {i}: {lat}, {lon}")
                continue

            with open(filename, "wb") as f:
                f.write(response.content)

            saved_images.append(filename)

        except Exception as e:
            print("Error code:", e)
            continue

    return saved_images


# %%
points = sample_population_points(csv_path, 2)

# center map on the mean location
center_lat = np.mean([lat for lat, lon in points])
center_lon = np.mean([lon for lat, lon in points])

m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

for lat, lon in points:
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color="red",
        fill=True,
        fill_opacity=0.7,
    ).add_to(m)
m

# %%
download_streetview_images(points=points, output_dir="streetview_images2", api_key=api_key)
