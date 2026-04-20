"""
LangChain tools for the Regio-AI strandskydd violation agent.

All tools share a mutable session dict injected via make_tools().
Results (numpy arrays, GeoDataFrames, map HTML) are stored in the session
so that each tool can build on the previous one's output.
"""
from __future__ import annotations

import os
import time
import tempfile

import numpy as np
import requests as http_requests
import geopandas as gpd
import folium
from folium import plugins
from shapely.geometry import shape
import rasterio.features
from rasterio.features import shapes as rasterio_shapes
from rasterio.enums import Resampling
from affine import Affine
from pyproj import Transformer
from langchain.tools import tool as lc_tool

# ── Fixed AOI — Orust, Bohuslän ──────────────────────────────────────────────
AOI_BBOX        = {"west": 11.65, "south": 58.18, "east": 11.91, "north": 58.35}
BBOX_LIST       = [AOI_BBOX["west"], AOI_BBOX["south"], AOI_BBOX["east"], AOI_BBOX["north"]]
POI_LAT         = 58.26599
POI_LON         = 11.77902
PRITHVI_BANDS   = ["B02", "B03", "B04", "B8A", "B11", "B12"]
INVALID_SCL     = {0, 1, 3, 8, 9, 10, 11}
PATCH_SIZE      = 224
STRANDSKYDD_M   = 100
NDBI_THRESHOLD  = 0.05
PRITHVI_MEANS   = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
PRITHVI_STDS    = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])


def make_tools(session: dict, prithvi_url: str) -> list:
    """Return a list of LangChain tools bound to *session* and *prithvi_url*."""

    # ── Tool 1 ────────────────────────────────────────────────────────────────
    @lc_tool
    def search_and_fetch_scenes(date_before: str, date_after: str) -> str:
        """
        Search Planetary Computer for the best cloud-free Sentinel-2 scenes for
        Orust island in the two date ranges, download them, and extract a
        224×224 px patch centred on the Point of Interest.

        Args:
            date_before: ISO date range for the 'before' epoch,
                         e.g. '2017-01-01/2018-12-31'
            date_after:  ISO date range for the 'after'  epoch,
                         e.g. '2022-06-01/2023-09-30'
        """
        import pystac_client
        import planetary_computer
        import stackstac

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        def best_scene(temporal: str):
            items = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=BBOX_LIST,
                datetime=temporal,
                query={"eo:cloud_cover": {"lt": 20}},
            ).item_collection()
            return min(items, key=lambda x: x.properties["eo:cloud_cover"])

        best_bef = best_scene(date_before)
        best_aft = best_scene(date_after)

        def make_stack(item):
            return stackstac.stack(
                [item], assets=PRITHVI_BANDS, bounds_latlon=BBOX_LIST,
                resolution=10, dtype="float64", fill_value=np.nan,
                rescale=False, epsg=32633,
            )

        def make_scl(item):
            return stackstac.stack(
                [item], assets=["SCL"], bounds_latlon=BBOX_LIST,
                resolution=10, dtype="float64", fill_value=0,
                rescale=False, epsg=32633, resampling=Resampling.nearest,
            )

        data_bef = make_stack(best_bef).squeeze("time").compute()
        data_aft = make_stack(best_aft).squeeze("time").compute()
        scl_bef  = make_scl(best_bef).squeeze("time").isel(band=0).compute().values
        scl_aft  = make_scl(best_aft).squeeze("time").isel(band=0).compute().values

        # Project POI → UTM 33N and find patch bounds
        to_utm   = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        poi_x, poi_y = to_utm.transform(POI_LON, POI_LAT)
        x_coords = data_aft.x.values
        y_coords = data_aft.y.values
        poi_col  = int(np.argmin(np.abs(x_coords - poi_x)))
        poi_row  = int(np.argmin(np.abs(y_coords - poi_y)))
        half     = PATCH_SIZE // 2
        r0 = max(0, poi_row - half)
        c0 = max(0, poi_col - half)
        r1 = min(r0 + PATCH_SIZE, data_aft.shape[-2])
        c1 = min(c0 + PATCH_SIZE, data_aft.shape[-1])

        def extract(data):
            bands = [data.sel(band=b).values[r0:r1, c0:c1] for b in PRITHVI_BANDS]
            p = np.stack(bands, axis=0).astype(np.float32)
            ph = PATCH_SIZE - p.shape[1]
            pw = PATCH_SIZE - p.shape[2]
            if ph > 0 or pw > 0:
                p = np.pad(p, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            return p

        patch_bef = extract(data_bef)
        patch_aft = extract(data_aft)

        dx = float(x_coords[1] - x_coords[0])
        dy = float(y_coords[1] - y_coords[0])
        patch_transform = Affine(dx, 0, float(x_coords[c0]) - dx / 2,
                                  0, dy, float(y_coords[r0]) - dy / 2)

        def normalize(p):
            m = PRITHVI_MEANS[:, None, None].astype(np.float32)
            s = PRITHVI_STDS[:, None, None].astype(np.float32)
            return (p - m) / s

        def pad_scl(scl):
            ph = PATCH_SIZE - scl.shape[0]
            pw = PATCH_SIZE - scl.shape[1]
            if ph > 0 or pw > 0:
                scl = np.pad(scl, ((0, ph), (0, pw)), mode="edge")
            return scl

        session.update({
            "patch_before":      patch_bef,
            "patch_after":       patch_aft,
            "patch_before_norm": normalize(patch_bef),
            "patch_after_norm":  normalize(patch_aft),
            "scl_before":        pad_scl(scl_bef[r0:r1, c0:c1]),
            "scl_after":         pad_scl(scl_aft[r0:r1, c0:c1]),
            "patch_transform":   patch_transform,
            "date_before":       best_bef.datetime.strftime("%Y-%m-%d"),
            "date_after":        best_aft.datetime.strftime("%Y-%m-%d"),
        })

        cloud_bef = best_bef.properties["eo:cloud_cover"]
        cloud_aft = best_aft.properties["eo:cloud_cover"]
        return (
            f"Downloaded scenes: "
            f"before={session['date_before']} ({cloud_bef:.1f}% cloud), "
            f"after={session['date_after']} ({cloud_aft:.1f}% cloud). "
            f"Patch: {PATCH_SIZE}×{PATCH_SIZE} px centred on POI (2.24 km²)."
        )

    # ── Tool 2 ────────────────────────────────────────────────────────────────
    @lc_tool
    def run_prithvi_water_detection(dummy: str = "") -> str:
        """
        Call the Prithvi-EO-2.0-300M KServe endpoint to segment water bodies in
        the before and after patches. Derives the strandskydd 100 m buffer from
        the after-scene water mask.
        Must be called after search_and_fetch_scenes.
        """
        if "patch_after_norm" not in session:
            return "Error: call search_and_fetch_scenes first."

        def call_endpoint(patch_norm: np.ndarray, label: str):
            # Add batch dimension: (6,224,224) → (1,6,224,224)
            inp = patch_norm[None].astype(np.float32)
            payload = {
                "inputs": [{
                    "name": "patch",
                    "shape": list(inp.shape),
                    "datatype": "FP32",
                    "data": inp.flatten().tolist(),
                }]
            }
            t0   = time.time()
            resp = http_requests.post(prithvi_url, json=payload, timeout=300)
            resp.raise_for_status()
            elapsed = time.time() - t0
            outs  = {o["name"]: o for o in resp.json()["outputs"]}
            preds = np.array(outs["preds"]["data"], dtype=np.int64).reshape(outs["preds"]["shape"])
            probs = np.array(outs["probs"]["data"], dtype=np.float32).reshape(outs["probs"]["shape"])
            water_pct = (preds == 1).mean() * 100
            return preds, probs, elapsed, water_pct

        preds_aft, probs_aft, t_aft, w_aft = call_endpoint(session["patch_after_norm"],  "after")
        preds_bef, probs_bef, t_bef, w_bef = call_endpoint(session["patch_before_norm"], "before")

        # Vectorise after-scene water mask → 100 m strandskydd buffer
        water_uint8 = (preds_aft == 1).astype(np.uint8)
        pt          = session["patch_transform"]
        water_geoms = [shape(g) for g, v in rasterio_shapes(water_uint8, transform=pt) if v == 1]

        strandskydd_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
        strandskydd_gdf  = None
        water_gdf_wgs84  = None

        if water_geoms:
            wgdf  = gpd.GeoDataFrame(geometry=water_geoms, crs="EPSG:32633")
            wdiss = wgdf.dissolve()
            water_gdf_wgs84 = (
                wdiss.simplify(20).to_frame("geometry")
                .set_crs("EPSG:32633").to_crs("EPSG:4326")
            )
            sbuf = gpd.GeoDataFrame(geometry=wdiss.buffer(STRANDSKYDD_M), crs="EPSG:32633")
            strandskydd_gdf = sbuf.to_crs("EPSG:4326")
            rast = rasterio.features.rasterize(
                [(g, 1) for g in sbuf.geometry],
                out_shape=(PATCH_SIZE, PATCH_SIZE),
                transform=pt, fill=0, dtype=np.uint8,
            )
            strandskydd_mask = rast == 1

        session.update({
            "preds_after":      preds_aft,
            "probs_after":      probs_aft,
            "preds_before":     preds_bef,
            "probs_before":     probs_bef,
            "water_after":      preds_aft == 1,
            "water_before":     preds_bef == 1,
            "strandskydd_mask": strandskydd_mask,
            "strandskydd_gdf":  strandskydd_gdf,
            "water_gdf_wgs84":  water_gdf_wgs84,
        })

        return (
            f"Prithvi water detection complete. "
            f"After ({session['date_after']}): {w_aft:.1f}% water ({t_aft:.0f}s). "
            f"Before ({session['date_before']}): {w_bef:.1f}% water ({t_bef:.0f}s). "
            f"Strandskydd zone: {strandskydd_mask.sum():,} px."
        )

    # ── Tool 3 ────────────────────────────────────────────────────────────────
    @lc_tool
    def compute_ndbi_change(dummy: str = "") -> str:
        """
        Compute NDBI (Normalised Difference Built-up Index) change between the
        before and after scenes to identify new built-up surfaces such as new
        buildings or paved areas. Intersects with strandskydd zone to flag
        potential violations.
        Must be called after run_prithvi_water_detection.
        """
        if "patch_after" not in session:
            return "Error: call search_and_fetch_scenes first."

        def ndbi(patch):
            swir1, nir = patch[4].astype(float), patch[3].astype(float)
            denom = swir1 + nir
            denom[denom == 0] = np.nan
            return (swir1 - nir) / denom

        ndbi_aft = ndbi(session["patch_after"])
        ndbi_bef = ndbi(session["patch_before"])

        # Apply SCL cloud masks
        ndbi_aft[np.isin(session["scl_after"],  list(INVALID_SCL))] = np.nan
        ndbi_bef[np.isin(session["scl_before"], list(INVALID_SCL))] = np.nan

        ndbi_change = ndbi_aft - ndbi_bef
        new_buildup = (ndbi_change > NDBI_THRESHOLD) & np.isfinite(ndbi_change)

        strandskydd_mask = session.get("strandskydd_mask",
                                       np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool))
        violation_mask   = new_buildup & strandskydd_mask

        session.update({
            "ndbi_after":     ndbi_aft,
            "ndbi_before":    ndbi_bef,
            "ndbi_change":    ndbi_change,
            "new_buildup":    new_buildup,
            "violation_mask": violation_mask,
        })

        return (
            f"NDBI change detection complete. "
            f"New built-up pixels (Δ > {NDBI_THRESHOLD}): {new_buildup.sum():,} "
            f"(~{new_buildup.sum() * 100:.0f} m²). "
            f"Potential violations inside strandskydd zone: {violation_mask.sum():,} px "
            f"(~{violation_mask.sum() * 100:.0f} m²)."
        )

    # ── Tool 4 ────────────────────────────────────────────────────────────────
    @lc_tool
    def generate_violation_map(dummy: str = "") -> str:
        """
        Generate an interactive Folium map showing the Prithvi water mask,
        strandskydd 100 m protection zone, NDBI new built-up areas, and
        potential violation pixels. Includes the official Naturvårdsverket WMS overlay.
        Must be called after compute_ndbi_change.
        Returns a summary with violation pixel count.
        """
        if "violation_mask" not in session:
            return "Error: call compute_ndbi_change first."

        m = folium.Map(
            location=[POI_LAT, POI_LON], zoom_start=15,
            tiles=None, control_scale=True,
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Satellite (Esri)", overlay=False, control=True,
        ).add_to(m)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(m)

        water_wgs84 = session.get("water_gdf_wgs84")
        if water_wgs84 is not None:
            folium.GeoJson(
                water_wgs84.__geo_interface__,
                name=f"Water bodies — Prithvi ({session.get('date_after', '')})",
                style_function=lambda x: {
                    "fillColor": "#3399FF", "color": "#0066CC",
                    "weight": 1, "fillOpacity": 0.55,
                },
                tooltip="Water body — Prithvi-EO-2.0 detection",
            ).add_to(m)

        strand_gdf = session.get("strandskydd_gdf")
        if strand_gdf is not None:
            folium.GeoJson(
                strand_gdf.__geo_interface__,
                name=f"Strandskydd zone ({STRANDSKYDD_M} m buffer)",
                style_function=lambda x: {
                    "fillColor": "#FF4444", "color": "#CC0000",
                    "weight": 1, "fillOpacity": 0.2,
                },
                tooltip=f"Protected zone — {STRANDSKYDD_M} m from water",
            ).add_to(m)

        violation_mask = session["violation_mask"]
        if violation_mask.any():
            viol_geoms = [
                shape(g)
                for g, v in rasterio_shapes(
                    violation_mask.astype(np.uint8), transform=session["patch_transform"]
                )
                if v == 1
            ]
            if viol_geoms:
                viol_gdf = gpd.GeoDataFrame(
                    geometry=viol_geoms, crs="EPSG:32633"
                ).to_crs("EPSG:4326")
                folium.GeoJson(
                    viol_gdf.__geo_interface__,
                    name="⚠ Potential violations",
                    style_function=lambda x: {
                        "fillColor": "#FF6600", "color": "#CC3300",
                        "weight": 2, "fillOpacity": 0.85,
                    },
                    tooltip="Potential strandskydd violation",
                ).add_to(m)

        folium.Marker(
            location=[POI_LAT, POI_LON],
            popup=folium.Popup(
                "Summer house POI<br>Suspected new structure post-2019", max_width=260
            ),
            tooltip="Point of Interest",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(m)
        folium.Circle(
            location=[POI_LAT, POI_LON], radius=200,
            color="red", weight=1.5, fill=False, tooltip="200 m context radius",
        ).add_to(m)
        folium.WmsTileLayer(
            url="https://nvpub.vic-metria.nu/arcgis/services/Strandskydd/MapServer/WmsServer",
            name="Strandskydd — Naturvårdsverket (official WMS)",
            layers="0", fmt="image/png", transparent=True,
            overlay=True, control=True, opacity=0.6, attr="Naturvårdsverket",
        ).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen().add_to(m)

        session["map_html"] = m._repr_html_()

        n = violation_mask.sum()
        return (
            f"Map generated. "
            + (f"POTENTIAL VIOLATION: {n} px (~{n * 100:.0f} m²) detected inside "
               f"the strandskydd zone near the POI."
               if n > 0 else
               "No violation signal detected in this scene combination.")
        )

    return [
        search_and_fetch_scenes,
        run_prithvi_water_detection,
        compute_ndbi_change,
        generate_violation_map,
    ]
