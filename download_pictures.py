import numpy as np
import pandas as pd
import requests
import os

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


def download_streetview_images(
        points,
        output_dir="streetview_images",
        size="640x640",
        heading=0,
        pitch=0,
        fov=90,
        api_key="",
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
        )

        filename = os.path.join(output_dir, f"streetview_{i}.jpg")

        try:
            response = requests.get("", timeout=10)

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


random_points = sample_population_points("csv/pol_pd_2020_1km_ASCII_XYZ.csv", 1)

#sometimes also signature parameter may be necessary
download_streetview_images(points=random_points, output_dir="streetview_images", api_key="")


