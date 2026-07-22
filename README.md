# 🌽 Corn Classification Web App — Sentinel-1/2 + KMeans

เว็บแอปจำแนกพื้นที่ปลูก **ข้าวโพด** จากภาพดาวเทียม **Sentinel-1 (SAR)** และ **Sentinel-2 (Optical)**
โดยใช้ **Google Earth Engine** ทำ Monthly Composite + Vegetation Indices แล้วแบ่งกลุ่มพื้นที่ด้วย
**KMeans Clustering** พร้อมแสดงผลบนแผนที่ interactive และคำนวณพื้นที่ข้าวโพดเป็น "ไร่"

> Capstone Project — Satellite-based Crop Classification using Google Earth Engine Python API

---

## 🚀 Live Demo

เปิดใช้งานบนเว็บ (Streamlit Cloud): **https://capstoneproject01.streamlit.app**

> 📱 บนมือถือกด *เพิ่มไปยังหน้าจอโฮม / เพิ่มลงในหน้าจอหลัก* เพื่อใช้เหมือนแอปจริง
> _(หากลิงก์ไม่ตรง ให้แก้เป็น URL จริงจากหน้า Streamlit ของคุณ)_

---

## ✨ ฟีเจอร์

- 🛰️ ดึงและรวมภาพ **Sentinel-1** (VV, VH, Diff, Ratio) และ **Sentinel-2** (+ NDVI, SAVI, NDMI) แบบรายเดือน
- 🧩 **KMeans Clustering** แบ่งพื้นที่ตามลักษณะสเปกตรัม (เลือกจำนวน cluster ได้)
- 🌽 ระบุ **cluster ที่เป็นข้าวโพด** และไฮไลต์บนภาพดาวเทียม (สลับ cluster ได้ทันที)
- 📐 คำนวณ **พื้นที่ข้าวโพด** เป็น ไร่ / เฮกตาร์ / % ของพื้นที่
- 🗺️ แผนที่ interactive (Folium) — เลื่อน/ซูมได้ พร้อม LayerControl
- ⚡ **Caching** — ขยับแผนที่ไม่ต้องประมวลผล Earth Engine ซ้ำ
- 📱 UI รองรับจอมือถือ

---

## 🧱 Tech Stack

| ส่วน | เทคโนโลยี |
|------|-----------|
| Frontend / UI | Streamlit |
| แผนที่ | Folium + streamlit-folium |
| ประมวลผลภาพดาวเทียม | Google Earth Engine (`earthengine-api`, `geemap`) |
| Deploy | Streamlit Community Cloud (ฟรี) |
| Auth บนคลาวด์ | GEE Service Account (ผ่าน Streamlit Secrets) |

---

## 📂 โครงสร้างโปรเจกต์

```
CapstoneProject01/
├── app.py                  # เว็บแอป Streamlit (UI, caching, render แผนที่)
├── ee_pipeline.py          # ชั้นเชื่อม Earth Engine (composite, clustering, พื้นที่ข้าวโพด)
├── ml_corn.py              # สคริปต์ EE ต้นฉบับ (เต็ม: composite → clustering → RF/SVM)
├── ml_corn_function.py     # ฟังก์ชัน EE ที่นำกลับมาใช้ซ้ำ
├── architecture.drawio     # ไดอะแกรมสถาปัตยกรรม (เปิดใน draw.io)
├── requirements.txt        # dependencies
├── run.ps1                 # สคริปต์รันแอปในเครื่อง (Windows)
├── .streamlit/config.toml  # ตั้งค่า Streamlit
├── notebooks/              # Jupyter notebook ต้นฉบับ
└── tests/                  # unit tests
```

---

## 🔄 ขั้นตอนการทำงาน (Pipeline)

```
AOI (พื้นที่สนใจ)
   │
   ├─ Sentinel-1 ─┐
   │              ├─ Monthly Composite + Indices ─┐
   ├─ Sentinel-2 ─┘                               │
   │                                              ▼
   │                          Combine + Reproject (EPSG:32647, 10m)
   │                                              │
   │                                              ▼
   │                                  KMeans Clustering (n clusters)
   │                                              │
   │                     ┌────────────────────────┤
   │                     ▼                        ▼
   │            เลือก cluster = ข้าวโพด    คำนวณพื้นที่รายคลัสเตอร์
   │                     │                        │
   ▼                     ▼                        ▼
แผนที่ดาวเทียม  +  overlay ข้าวโพด (เหลือง)  +  พื้นที่ (ไร่)
```

หัวใจของการทำงาน: Earth Engine คืนค่าเป็น **XYZ tile URL** (string) → เบราว์เซอร์โหลด tile เอง
ทำให้ผลลัพธ์ **cache ได้** และ **ขยับแผนที่ไม่ต้องเรียก EE ซ้ำ**

---

## 💻 รันในเครื่อง (Local)

**ต้องมี:** Python 3.11+ และบัญชี Google Earth Engine

```bash
# 1) ติดตั้ง dependencies
pip install -r requirements.txt

# 2) auth Earth Engine (ครั้งแรกครั้งเดียว)
earthengine authenticate

# 3) รันแอป
streamlit run app.py
```

Windows (PowerShell) รันสั้น ๆ ได้ด้วย:
```powershell
.\run.ps1
```

เปิดเบราว์เซอร์ที่ `http://localhost:8501`

---

## ☁️ Deploy บน Streamlit Cloud

1. Push โค้ดขึ้น GitHub (public repo)
2. สร้าง **GEE Service Account** ใน Google Cloud Console (project ของคุณ) และให้ role:
   - **Earth Engine Resource Writer** (จำเป็นสำหรับสร้าง map tile)
   - **Service Usage Consumer**
   จากนั้นดาวน์โหลด **JSON key**
3. ไปที่ [share.streamlit.io](https://share.streamlit.io/) → Deploy จาก repo → เลือก `app.py`
4. ใส่ **Secrets** (Advanced settings) โดยวางเนื้อ JSON key ทั้งก้อน:
   ```toml
   EE_SERVICE_ACCOUNT = '''
   {
     "type": "service_account",
     "project_id": "your-ee-project",
     ...
   }
   '''
   ```
5. กด **Deploy**

> โค้ด `initialize_ee()` ใน [ee_pipeline.py](ee_pipeline.py) จะตรวจอัตโนมัติ:
> ใช้ Service Account เมื่ออยู่บนคลาวด์ / ใช้ auth ปกติเมื่อรันในเครื่อง

⚠️ **อย่า commit ไฟล์ JSON key ขึ้น Git เด็ดขาด** — `.gitignore` กันไว้แล้ว key เก็บใน Streamlit Secrets เท่านั้น

---

## 🗺️ ปรับพารามิเตอร์

ปรับได้จาก Sidebar ในแอป:
- **AOI** (พิกัดขอบเขตพื้นที่)
- **ช่วงฤดูเพาะปลูก** (วันเริ่ม–สิ้นสุด)
- **จำนวน Cluster** และ **จำนวน Sample pixels**
- **Cluster ที่เป็นข้าวโพด**

---

## 📜 License

Educational / research use — Capstone Project.
Earth Engine ใช้ได้ฟรีสำหรับงานวิจัย/การศึกษา (noncommercial)
