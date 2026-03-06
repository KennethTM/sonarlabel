# %%
# Imports
from sonarlight import Sonar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# %%
# Helper functions

EARTH_SEMI_MINOR = 6356752.3142  # WGS84 semi-minor axis


def x_to_lon(x):
    return x / EARTH_SEMI_MINOR * (180 / np.pi)


def y_to_lat(y):
    return (2 * np.arctan(np.exp(y / EARTH_SEMI_MINOR)) - np.pi / 2) * (180 / np.pi)


def pixel_to_latlon(row_idx, col_idx, df, n_cols):
    """Convert a sidescan pixel (row, col) to (lat, lon)."""
    ping = df.iloc[row_idx]
    across_dist = np.interp(col_idx, [0, n_cols - 1], [ping["min_range"], ping["max_range"]])
    mx = ping["x"] + across_dist * np.cos(ping["gps_heading"])
    my = ping["y"] - across_dist * np.sin(ping["gps_heading"])
    return y_to_lat(my), x_to_lon(mx)


def annotation_row_bounds(annotations):
    """Return (min_row, max_row) bounding all annotation polygons."""
    all_vertices = [pt for poly in annotations["polygon"] for pt in poly]
    min_row = int(np.floor(min(pt[0] for pt in all_vertices)))
    max_row = int(np.ceil(max(pt[0] for pt in all_vertices)))
    return min_row, max_row


def polygons_to_mask(polygons, height, width, row_offset=0):
    """Rasterize polygons into a binary mask (0=background, 1=foreground)."""
    mask_pil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_pil)
    for poly in polygons:
        verts = [(float(p[1]), float(p[0] - row_offset)) for p in poly]
        if len(verts) >= 3:
            draw.polygon(verts, outline=1, fill=1)
    return np.array(mask_pil, dtype=np.uint8)


def georef_polygons(annotations, sidescan_df, frame_width):
    """Convert all annotation polygons from pixel coords to lat/lon dicts."""
    geo_polygons = []
    for _, ann in annotations.iterrows():
        coords = []
        for row_px, col_px in ann["polygon"]:
            lat, lon = pixel_to_latlon(int(row_px), int(col_px), sidescan_df, frame_width)
            coords.append({"lat": lat, "lon": lon})
        geo_polygons.append(coords)
    return geo_polygons


def compute_metric_extent(df, shape):
    """Compute along-track distance and pixel resolutions from ping positions.

    Returns (extent, row_res, col_res, total_along) where extent is
    [across_min, across_max, total_along, 0] for use with imshow.
    """
    n_rows, n_cols = shape
    across_min = df["min_range"].iloc[0]
    across_max = df["max_range"].iloc[0]

    dx = np.diff(df["x"].values.astype(float))
    dy = np.diff(df["y"].values.astype(float))
    total_along = np.sum(np.sqrt(dx**2 + dy**2))

    row_res = total_along / (n_rows - 1)
    col_res = (across_max - across_min) / (n_cols - 1)
    extent = [across_min, across_max, total_along, 0]
    return extent, row_res, col_res, total_along


# %%
# Load sonar data
sonar_path = "/media/kenneth/d6c13395-8492-49ee-9c0f-6a165e34c95c/sonar_project/data/Bromme 01.sl3"
sonar = Sonar(sonar_path, clean=False)

# %%
# Extract sidescan channel
sidescan_df = sonar.df.query("survey == 'sidescan'").reset_index(drop=True)
sidescan_img = np.stack(sidescan_df["frames"])
n_pings, frame_width = sidescan_img.shape
print(f"Sidescan image: {n_pings} pings x {frame_width} samples")

# %%
# Load annotations and crop to annotated region
annotations = pd.read_json("annotations.jsonl", lines=True)

min_row, max_row = annotation_row_bounds(annotations)
crop_start = max(0, min_row)
crop_end = min(n_pings, max_row + 1)

cropped_img = sidescan_img[crop_start:crop_end, :]
cropped_df = sidescan_df.iloc[crop_start:crop_end].reset_index(drop=True)

# %%
# Build binary mask
mask = polygons_to_mask(annotations["polygon"], *cropped_img.shape, row_offset=crop_start)
print(f"Mask shape: {mask.shape}, values: {np.unique(mask)}")

# %%
# Plot cropped sidescan with annotation outlines
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cropped_img, cmap="copper", interpolation="nearest")
for _, ann in annotations.iterrows():
    poly = ann["polygon"]
    ax.plot([p[1] for p in poly], [p[0] - crop_start for p in poly], color="red")
ax.set_aspect("equal")
ax.set_title("Cropped Sidescan with Annotations")
ax.set_xlabel("Range (pixels)")
ax.set_ylabel("Along-track (pixels)")
plt.show()

# %%
# Plot cropped sidescan with mask overlay (green = plants)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cropped_img, cmap="copper", interpolation="nearest")
ax.imshow(
    np.ma.masked_where(mask == 0, mask),
    cmap="Greens",
    alpha=0.45,
    interpolation="nearest",
)
ax.set_aspect("equal")
ax.set_title("Cropped Sidescan with Plant Mask Overlay")
ax.set_xlabel("Range (pixels)")
ax.set_ylabel("Along-track (pixels)")
plt.show()

# %%
# Convert annotation polygons to lat/lon
annotations["geo_polygon"] = georef_polygons(annotations, sidescan_df, frame_width)
print(annotations[["polygon", "geo_polygon"]].head())

# %%
# --- Interactive Leaflet map (folium) ---
import folium

route_coords = list(zip(cropped_df["latitude"], cropped_df["longitude"]))
center_lat = cropped_df["latitude"].mean()
center_lon = cropped_df["longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")

# Vessel track centerline
folium.PolyLine(route_coords, color="blue", weight=2, opacity=0.7, tooltip="Vessel track").add_to(m)

# Annotation polygons
for _, ann in annotations.iterrows():
    poly_coords = [(pt["lat"], pt["lon"]) for pt in ann["geo_polygon"]]
    folium.Polygon(
        locations=poly_coords,
        color="red",
        weight=2,
        fill=True,
        fill_color="red",
        fill_opacity=0.3,
        tooltip=ann.get("label", "annotation"),
    ).add_to(m)

m

# %%
# --- Metric-scaled sidescan plot ---
extent, row_res, col_res, total_along = compute_metric_extent(cropped_df, cropped_img.shape)
print(f"Along-track: {total_along:.1f} m | Row res: {row_res:.4f} m/px | Col res: {col_res:.4f} m/px")

fig, ax = plt.subplots(figsize=(10, 20))
ax.imshow(cropped_img, cmap="copper", interpolation="nearest", extent=extent, aspect="equal")

across_min, across_max = extent[0], extent[1]
crop_h, crop_w = cropped_img.shape
for _, ann in annotations.iterrows():
    poly = ann["polygon"]
    xs = [np.interp(p[1], [0, crop_w - 1], [across_min, across_max]) for p in poly]
    ys = [(p[0] - crop_start) * row_res for p in poly]
    ax.plot(xs, ys, color="red", linewidth=1.5)

ax.set_xlabel("Across-track (m)")
ax.set_ylabel("Along-track (m)")
ax.set_title("Sidescan Image (metric scale)")
plt.show()

# %%
