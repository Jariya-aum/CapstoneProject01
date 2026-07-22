# ml_corn_function.py
"""
Helper functions for corn classification
Extracted from ml_corn.py for reusable imports
"""

import ee
import pandas as pd
import numpy as np

def get_sentinel1_data(crop_season_start, crop_season_end, aoi):
    """ดึงข้อมูล Sentinel-1"""
    sen1_raw = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filterDate(crop_season_start, crop_season_end) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH'])

    print(f"Sentinel1_image: {sen1_raw.size().getInfo()}")
    return sen1_raw

def get_sentinel2_data(crop_season_start, crop_season_end, aoi):
    """ดึงข้อมูล Sentinel-2"""
    sen2_raw = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(crop_season_start, crop_season_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])

    print(f"Sentinel2_image: {sen2_raw.size().getInfo()}")
    return sen2_raw

def create_monthly_composites(s1_raw, s2_raw, months_list):
    """สร้าง monthly composites"""
    
    def create_s1_monthly(month):
        month = ee.Number(month)
        year = ee.Algorithms.If(month.lte(2), 2025, 2024)
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')

        img = ee.Image(s1_raw.filterDate(start, end).mean())
        band_names = img.bandNames()
        month_str = month.format('%02d')

        def create_bands():
            vv = img.select('VV')
            vh = img.select('VH')
            diff = vh.subtract(vv)
            ratio = vh.divide(vv)

            name_vv = ee.String('S1VV_').cat(month_str)
            name_vh = ee.String('S1VH_').cat(month_str)
            name_diff = ee.String('S1Diff_').cat(month_str)
            name_ratio = ee.String('S1Ratio_').cat(month_str)

            return vv.rename([name_vv]) \
                .addBands(vh.rename([name_vh])) \
                .addBands(diff.rename([name_diff])) \
                .addBands(ratio.rename([name_ratio]))

        return ee.Algorithms.If(
            band_names.contains('VV'),
            ee.Algorithms.If(
                band_names.contains('VH'),
                create_bands(),
                None
            ),
            None
        )

    s1_monthly = months_list.map(create_s1_monthly)
    s1_monthly = ee.List(s1_monthly).removeAll([None])

    return s1_monthly