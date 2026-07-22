# tests/test_indices.py
import pytest

def calculate_ndvi(nir, red):
    """คำนวณ NDVI = (NIR - Red) / (NIR + Red)"""
    if nir + red == 0:
        return 0
    return (nir - red) / (nir + red)

def calculate_savi(nir, red, L=0.5):
    """คำนวณ SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)"""
    if nir + red + L == 0:
        return 0
    return ((nir - red) / (nir + red + L)) * (1 + L)

def calculate_ndmi(nir, swir):
    """คำนวณ NDMI = (NIR - SWIR) / (NIR + SWIR)"""
    if nir + swir == 0:
        return 0
    return (nir - swir) / (nir + swir)

# --- Tests ---

def test_ndvi_healthy_vegetation():
    """ทดสอบ NDVI สำหรับพืชสมบูรณ์ (ควรได้ค่าใกล้ 1)"""
    nir = 0.8  # ค่า NIR สูง
    red = 0.1  # ค่า Red ต่ำ (พืชดูดกลืน)
    
    ndvi = calculate_ndvi(nir, red)
    
    assert 0.7 <= ndvi <= 1.0, f"Healthy vegetation NDVI should be 0.7-1.0, got {ndvi:.2f}"
    print(f"✅ NDVI Healthy Test Passed: {ndvi:.4f}")

def test_ndvi_bare_soil():
    """ทดสอบ NDVI สำหรับดินเปล่า (ควรได้ค่าใกล้ 0)"""
    nir = 0.3
    red = 0.3
    
    ndvi = calculate_ndvi(nir, red)
    
    assert -0.1 <= ndvi <= 0.2, f"Bare soil NDVI should be near 0, got {ndvi:.2f}"
    print(f"✅ NDVI Bare Soil Test Passed: {ndvi:.4f}")

def test_savi_calculation():
    """ทดสอบ SAVI (Soil-Adjusted Vegetation Index)"""
    nir = 0.7
    red = 0.2
    
    savi = calculate_savi(nir, red, L=0.5)
    
    # SAVI ควรอยู่ระหว่าง -1 ถึง +1
    assert -1 <= savi <= 1, f"SAVI should be in range [-1, 1], got {savi:.2f}"
    print(f"✅ SAVI Test Passed: {savi:.4f}")

def test_ndmi_wet_vegetation():
    """ทดสอบ NDMI สำหรับพืชมีความชื้นสูง"""
    nir = 0.8
    swir = 0.2  # SWIR ต่ำ = ความชื้นสูง
    
    ndmi = calculate_ndmi(nir, swir)
    
    assert ndmi > 0.5, f"Wet vegetation NDMI should be > 0.5, got {ndmi:.2f}"
    print(f"✅ NDMI Wet Test Passed: {ndmi:.4f}")

def test_ndmi_dry_vegetation():
    """ทดสอบ NDMI สำหรับพืชแห้ง"""
    nir = 0.3
    swir = 0.7  # SWIR สูง = แห้ง
    
    ndmi = calculate_ndmi(nir, swir)
    
    assert ndmi < 0, f"Dry vegetation NDMI should be negative, got {ndmi:.2f}"
    print(f"✅ NDMI Dry Test Passed: {ndmi:.4f}")