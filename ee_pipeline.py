# -*- coding: utf-8 -*-
"""
ee_pipeline.py
==============
ชั้นเชื่อมต่อ (Integration Layer) ระหว่าง Web App กับโค้ด Earth Engine เดิม

>>> จุดสำคัญ <<<
ไฟล์นี้คือ "ที่เดียว" ที่คุณต้องเอาโค้ด EE เดิมจาก ml_corn.py มาวาง
- ฟังก์ชันในไฟล์นี้ต้องคืนค่าเป็นชนิดที่ pickle ได้ (string / number / dict)
  เท่านั้น เพื่อให้ @st.cache_data ใน app.py ทำงานได้
- ห้ามคืนค่าเป็น ee.Image / ee.FeatureCollection ออกไปนอกไฟล์นี้โดยตรง
  เพราะ Streamlit cache จะ serialize ไม่ได้ -> ให้แปลงเป็น tile URL / getInfo() ก่อน
"""

import ee


# ---------------------------------------------------------------------------
# 1) เริ่มต้น Earth Engine
# ---------------------------------------------------------------------------
def initialize_ee(project: str = "ee-aum121236"):
    """
    เริ่มต้น Earth Engine

    บนเครื่อง local:      ee.Authenticate() ครั้งแรกก่อน แล้ว ee.Initialize()
    บน Streamlit Cloud:   ใช้ Service Account (ดูหมายเหตุท้ายไฟล์)
    """
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


# ---------------------------------------------------------------------------
# 2) แปลง ee.Image -> tile URL template (สิ่งที่ทำให้ cache ได้)
# ---------------------------------------------------------------------------
def get_tile_url(ee_image: "ee.Image", vis_params: dict) -> str:
    """
    รับ ee.Image + vis_params แล้วคืน URL template ของ XYZ tiles (เป็น string)
    string นี้เอาไปใส่ folium.TileLayer ได้ และ pickle ได้ -> cache ได้
    """
    map_id = ee.Image(ee_image).getMapId(vis_params)
    return map_id["tile_fetcher"].url_format


# ---------------------------------------------------------------------------
# 3) >>> วางโค้ด EE เดิมของคุณตรงนี้ <<<
#    ฟังก์ชันเหล่านี้ตอนนี้เป็น placeholder — ก็อปโค้ดจาก ml_corn.py มาวาง
# ---------------------------------------------------------------------------

def build_aoi(bounds: list) -> "ee.Geometry":
    """
    สร้าง Area of Interest

    >>> วางโค้ด AOI เดิม (จาก ml_corn.py บรรทัด 29) <<<
        aoi = ee.Geometry.Rectangle([...])
    """
    # bounds = [xmin, ymin, xmax, ymax]
    return ee.Geometry.Rectangle(bounds)


def build_composite(aoi, crop_season_start: str, crop_season_end: str) -> "ee.Image":
    """
    สร้าง Sentinel-1/2 monthly composite แบบรวม band ทั้งหมด (reprojected)

    >>> วางโค้ด composite เดิม (จาก ml_corn.py) <<<
        - get_sentinel1_data(...)          -> มีอยู่แล้วใน ml_corn_function.py
        - get_sentinel2_data(...)          -> มีอยู่แล้วใน ml_corn_function.py
        - create_monthly_composites(...)   -> มีอยู่แล้วใน ml_corn_function.py
        - process_month_s2(...) / s2_composite
        - monthly_composite_all = s1_composite.addBands(s2_composite)
                                   .reproject(crs='EPSG:32647', scale=10)
        return monthly_composite_all

    เวอร์ชันนี้เติมให้ครบแล้ว โดยเรียกใช้ฟังก์ชันเดิมจาก ml_corn_function.py
    (S1) และ implement S2 composite ตาม ml_corn.py บรรทัด 193-266
    """
    from ml_corn_function import (
        get_sentinel1_data, get_sentinel2_data, create_monthly_composites,
    )

    months = ee.List(list(range(5, 13))).cat(ee.List(list(range(1, 3))))

    # --- Sentinel-1 monthly composite ---
    s1_raw = get_sentinel1_data(crop_season_start, crop_season_end, aoi)
    s2_raw = get_sentinel2_data(crop_season_start, crop_season_end, aoi)
    s1_monthly = create_monthly_composites(s1_raw, s2_raw, months)
    s1_composite = (
        ee.ImageCollection.fromImages(s1_monthly)
        .filter(ee.Filter.notNull(["system:band_names"]))
        .toBands()
    )

    # --- Sentinel-2 monthly composite + vegetation indices ---
    s2_monthly = months.map(lambda m: _process_month_s2(m, s2_raw))
    s2_composite = (
        ee.ImageCollection.fromImages(s2_monthly)
        .filter(ee.Filter.notNull(["system:band_names"]))
        .toBands()
    )

    # --- Combine + reproject (EPSG:32647, 10m) ---
    return s1_composite.addBands(s2_composite).reproject(crs="EPSG:32647", scale=10)


def _process_month_s2(m, s2_raw):
    """S2 monthly median + NDVI/SAVI/NDMI (อ้างอิง ml_corn.py บรรทัด 193-245)"""
    month = ee.Number(m)
    year = ee.Algorithms.If(month.lte(2), 2025, 2024)
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    img = s2_raw.filterDate(start, end).median()
    month_str = month.format("%02d")
    band_names = img.bandNames()

    required_bands = ee.List(["B4", "B8", "B11"])

    def check_band(b, prev):
        return ee.Algorithms.If(prev, band_names.contains(b), False)

    has_all_bands = required_bands.iterate(check_band, True)

    def create_indices():
        ndvi = img.normalizedDifference(["B8", "B4"]).rename(
            ee.String("NDVI_").cat(month_str)
        )
        ndmi = img.normalizedDifference(["B8", "B11"]).rename(
            ee.String("NDMI_").cat(month_str)
        )
        savi = img.expression(
            "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
            {"NIR": img.select("B8"), "RED": img.select("B4"), "L": 0.5},
        ).rename(ee.String("SAVI_").cat(month_str))

        def rename_band(band):
            return ee.String("S2_").cat(band).cat("_").cat(month_str)

        renamed = img.rename(img.bandNames().map(rename_band))
        return renamed.addBands(ndvi).addBands(savi).addBands(ndmi)

    return ee.Algorithms.If(has_all_bands, create_indices(), None)


def run_clustering(composite: "ee.Image", aoi, n_clusters: int,
                   num_pixels: int = 5000, seed: int = 42) -> "ee.Image":
    """
    KMeans clustering บน composite

    >>> วางโค้ด clustering เดิม (จาก ml_corn.py บรรทัด 291-302) <<<
        sample = composite.sample(region=aoi, scale=10, numPixels=..., seed=..., geometries=True)
        clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(sample)
        return composite.cluster(clusterer)
    """
    sample = composite.sample(
        region=aoi, scale=10, numPixels=num_pixels, seed=seed, geometries=True
    )
    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(sample)
    return composite.cluster(clusterer)


def cluster_histogram(cluster_image: "ee.Image", aoi) -> dict:
    """
    สรุปการกระจายตัวของแต่ละ cluster -> คืน dict (pickle ได้)

    >>> อ้างอิงโค้ดเดิม ml_corn.py บรรทัด 615-623 <<<
    """
    hist = cluster_image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi,
        scale=10,
        maxPixels=int(1e9),
        bestEffort=True,
    )
    return hist.getInfo().get("cluster", {})


def cluster_area_stats(cluster_image: "ee.Image", aoi) -> dict:
    """
    คำนวณ 'พื้นที่จริง' (ตร.ม.) ของแต่ละ cluster ในครั้งเดียว
    ใช้ ee.Image.pixelArea() รวมแบบ group by cluster -> คืน {cluster_id: area_sqm}
    (อ้างอิงวิธีคำนวณพื้นที่จาก ml_corn.py บรรทัด 478-488)
    """
    area_img = ee.Image.pixelArea().addBands(cluster_image)
    result = area_img.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="cluster"),
        geometry=aoi,
        scale=10,
        maxPixels=int(1e10),
        bestEffort=True,
    )
    groups = result.getInfo().get("groups", [])
    return {int(g["cluster"]): float(g["sum"]) for g in groups}


def corn_mask_tile(cluster_image: "ee.Image", corn_cluster_id: int) -> str:
    """
    สร้าง tile URL ของ 'หน้ากากข้าวโพด' (สีเหลือง) จาก cluster ที่เลือก
    corn_mask = cluster_image.eq(corn_cluster_id)  (อ้างอิง ml_corn.py บรรทัด 316-318)
    """
    corn = ee.Image(cluster_image).eq(corn_cluster_id).selfMask()
    return get_tile_url(corn, {"min": 1, "max": 1, "palette": ["#FFEB00"]})


# ---------------------------------------------------------------------------
# หมายเหตุ: การ deploy บน Streamlit Cloud ด้วย Service Account
# ---------------------------------------------------------------------------
# import json
# from google.oauth2 import service_account
# key_dict = json.loads(st.secrets["EE_SERVICE_ACCOUNT"])
# creds = service_account.Credentials.from_service_account_info(
#     key_dict, scopes=["https://www.googleapis.com/auth/earthengine"]
# )
# ee.Initialize(creds, project=key_dict["project_id"])
