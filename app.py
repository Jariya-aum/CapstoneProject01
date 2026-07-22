# -*- coding: utf-8 -*-
"""
app.py — Web App สำหรับ Corn Classification (Sentinel-1/2 + KMeans)
==================================================================
รัน:  streamlit run app.py

โครงสร้าง:
  Sidebar  -> รับ parameter (วันที่ / จำนวน cluster / จำนวน sample)
  Main     -> แผนที่ interactive (folium) + สถิติ cluster
  Caching  -> @st.cache_resource สำหรับ EE init, @st.cache_data สำหรับผลลัพธ์
"""

import streamlit as st
import folium
from streamlit_folium import st_folium

import ee_pipeline as pipe


# ---------------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Corn Classification (GEE)", layout="wide")


# ---------------------------------------------------------------------------
# CACHING LAYER
# ---------------------------------------------------------------------------
# @st.cache_resource  -> ของที่ pickle ไม่ได้ / อยากให้มีตัวเดียวทั้งแอป (EE session)
@st.cache_resource(show_spinner="กำลังเชื่อมต่อ Earth Engine...")
def init_earth_engine():
    pipe.initialize_ee()
    return True


# @st.cache_data      -> ผลลัพธ์ที่ pickle ได้ (string / number / dict)
#   คีย์ของ cache = ค่าพารามิเตอร์ทั้งหมด -> เปลี่ยนพารามิเตอร์เมื่อไรจึงคำนวณใหม่
#   ขยับ/ซูมแผนที่เฉย ๆ พารามิเตอร์ไม่เปลี่ยน -> ใช้ค่าจาก cache ทันที (ไม่เรียก EE ซ้ำ)
@st.cache_data(show_spinner=False)
def compute_layers(bounds, start, end, n_clusters, num_pixels):
    """
    รัน pipeline ทั้งหมดแล้วคืน 'ผลลัพธ์ที่ serialize ได้' เท่านั้น
    (tile URL templates + สถิติ) เพื่อให้ cache ได้

    สำคัญ: สร้าง tile ข้าวโพด (สีเหลือง) ไว้ล่วงหน้า "ทุก cluster"
    -> ผู้ใช้สลับว่า cluster ไหนคือข้าวโพดได้ทันที โดยไม่ต้องประมวลผล EE ใหม่
    """
    aoi = pipe.build_aoi(bounds)
    composite = pipe.build_composite(aoi, start, end)
    cluster_img = pipe.run_clustering(composite, aoi, n_clusters, num_pixels)

    palette = ["red", "blue", "green", "orange", "purple",
               "yellow", "cyan", "magenta", "lime", "brown"]
    cluster_vis = {"min": 0, "max": n_clusters - 1, "palette": palette[:n_clusters]}

    return {
        "cluster_tiles": pipe.get_tile_url(cluster_img, cluster_vis),
        # tile ข้าวโพดสีเหลือง แยกตาม cluster id (คีย์เป็น str เพื่อความชัวร์เวลา cache)
        "corn_tiles": {str(cid): pipe.corn_mask_tile(cluster_img, cid)
                       for cid in range(n_clusters)},
        # พื้นที่จริง (ตร.ม.) ของแต่ละ cluster
        "areas": {str(k): v for k, v in pipe.cluster_area_stats(cluster_img, aoi).items()},
        "center": [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
        "n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------------
# SIDEBAR — พารามิเตอร์
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ พารามิเตอร์")

st.sidebar.subheader("พื้นที่ (AOI)")
xmin = st.sidebar.number_input("Lon min", value=98.03724813148085, format="%.6f")
ymin = st.sidebar.number_input("Lat min", value=18.32740564041369, format="%.6f")
xmax = st.sidebar.number_input("Lon max", value=98.36821126624648, format="%.6f")
ymax = st.sidebar.number_input("Lat max", value=18.67642400421453, format="%.6f")

st.sidebar.subheader("ช่วงฤดูเพาะปลูก")
start = st.sidebar.text_input("เริ่ม (YYYY-MM-DD)", "2024-05-01")
end = st.sidebar.text_input("สิ้นสุด (YYYY-MM-DD)", "2025-02-28")

st.sidebar.subheader("Clustering")
n_clusters = st.sidebar.slider("จำนวน Cluster", 2, 10, 5)
num_pixels = st.sidebar.select_slider(
    "จำนวน Sample pixels", options=[1000, 2000, 5000, 10000], value=5000
)

run = st.sidebar.button("▶️ ประมวลผล", type="primary", use_container_width=True)

# เลือกว่า cluster ไหน = ข้าวโพด (ค่าเริ่มต้น 2 ตามโค้ดเดิม corn_cluster_id = 2)
# ปรับค่านี้ได้ทันทีหลังประมวลผล โดยไม่ต้องรัน EE ใหม่
st.sidebar.subheader("🌽 พื้นที่ข้าวโพด")
default_corn = 2 if n_clusters > 2 else 0
corn_cluster_id = st.sidebar.selectbox(
    "Cluster ที่เป็นข้าวโพด",
    options=list(range(n_clusters)),
    index=default_corn,
    help="ดูสัดส่วน/สีของแต่ละ cluster บนแผนที่ แล้วเลือกอันที่เป็นข้าวโพด",
)


# ---------------------------------------------------------------------------
# MAIN — หัวข้อ + แผนที่
# ---------------------------------------------------------------------------
st.title("🌽 Corn Classification — Sentinel-1/2 + KMeans")

init_earth_engine()  # เชื่อม EE (มี spinner จาก cache_resource)

bounds = [xmin, ymin, xmax, ymax]

# เก็บผลลัพธ์ไว้ใน session_state เพื่อให้ยังอยู่ตอน re-render (เช่น ขยับแผนที่)
if run:
    with st.spinner("⏳ กำลังประมวลผล Earth Engine (composite + clustering)..."):
        try:
            st.session_state["result"] = compute_layers(
                tuple(bounds), start, end, n_clusters, num_pixels
            )
        except NotImplementedError as e:
            st.warning(f"ยังไม่ได้วางโค้ด EE เดิม: {e}")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

# วาดแผนที่ (ใช้ folium — โหลด tile จาก URL ที่ cache ไว้ ไม่คำนวณ EE ซ้ำ)
result = st.session_state.get("result")
center = result["center"] if result else [(ymin + ymax) / 2, (xmin + xmax) / 2]

# basemap = ภาพดาวเทียม (Google Satellite) เพื่อให้ overlay ข้าวโพดสีเหลืองเด่นเหมือนภาพตัวอย่าง
fmap = folium.Map(location=center, zoom_start=12, control_scale=True, tiles=None)
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google Satellite", name="🛰️ ดาวเทียม", control=True,
).add_to(fmap)
folium.TileLayer("OpenStreetMap", name="🗺️ แผนที่ถนน", control=True).add_to(fmap)
folium.Rectangle([[ymin, xmin], [ymax, xmax]], color="red", fill=False).add_to(fmap)

if result:
    # 1) ชั้น cluster ทั้งหมด (ปิดไว้ก่อน เปิดดูได้จาก LayerControl)
    folium.TileLayer(
        tiles=result["cluster_tiles"], attr="Google Earth Engine",
        name="ทุก Cluster", overlay=True, control=True, show=False,
    ).add_to(fmap)
    # 2) ชั้นข้าวโพด (สีเหลือง) ของ cluster ที่เลือก — เปิดไว้เป็นค่าเริ่มต้น
    folium.TileLayer(
        tiles=result["corn_tiles"][str(corn_cluster_id)], attr="Google Earth Engine",
        name=f"🌽 ข้าวโพด (Cluster {corn_cluster_id})", overlay=True, control=True, show=True,
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

col_map, col_stats = st.columns([3, 1])
with col_map:
    st_folium(fmap, width=None, height=560, returned_objects=[])

with col_stats:
    if result:
        areas = result["areas"]  # {cluster_id(str): area_sqm}
        total = sum(areas.values()) or 1

        # ---- พื้นที่ข้าวโพดที่เลือก ----
        st.subheader("🌽 พื้นที่ข้าวโพด")
        corn_sqm = areas.get(str(corn_cluster_id), 0.0)
        st.metric("ไร่", f"{corn_sqm / 1600:,.0f}")   # 1 ไร่ = 1,600 ตร.ม.
        st.caption(
            f"{corn_sqm / 10000:,.1f} เฮกตาร์  ·  "
            f"{corn_sqm / 1_000_000:,.2f} ตร.กม.  ·  "
            f"{corn_sqm / total * 100:.1f}% ของ AOI"
        )
        st.divider()

        # ---- สัดส่วนแต่ละ cluster (ช่วยเลือกว่าอันไหนคือข้าวโพด) ----
        st.subheader("📊 ทุก Cluster")
        for cid in sorted(areas, key=int):
            pct = areas[cid] / total * 100
            label = f"Cluster {cid}" + ("  🌽" if int(cid) == corn_cluster_id else "")
            st.write(f"**{label}** — {pct:.1f}%  ({areas[cid] / 1600:,.0f} ไร่)")
            st.progress(min(pct / 100, 1.0))
    else:
        st.info("กด ▶️ ประมวลผล เพื่อเริ่ม")
