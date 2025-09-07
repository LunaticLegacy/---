# -*- coding: utf-8 -*-
# GenerateSyntheticTerrain.pyt
# ArcGIS Pro Python Toolbox: Generate synthetic terrain (DEM) using interpolation with random noise
# 一个用于模拟地形的简单工具。
# 
# Requirements:
# - ArcGIS Pro 2.8+
# - Spatial Analyst extension (for IDW/Spline interpolation)
#
# Base Author: ChatGPT (GPT-5 Thinking)
# Co-Author: 月と猫 - LunaNeko
# Date: 2025-09-04
#

import arcpy as ap
import math
import random
import os

import numpy as np
import noise

from typing import List, Dict, Tuple, Any, Optional, TypeAlias
from typing_extensions import TypedDict
from dataclasses import dataclass

# 来自arcpy的导入项
from arcpy import env
from arcpy.sa import Idw, Spline, ExtractByMask

# 规定基础类型格式。
@dataclass
class Line(TypedDict):            
    geometry: ap.Polyline
    width: float
    level: int

Point: TypeAlias = Tuple[Tuple[float, float], float]

# 从这里开始写内容。
class Toolbox(object):
    def __init__(self):
        self.label: str = "地形生成工具箱"
        self.alias: str = "syntheticTerrain"
        self.tools: List = [
            GenerateTerrain, 
        ]

class GenerateTerrain:
    def __init__(self) -> None:
        self.label: str = "生成地形"
        self.description: str = "利用带有随机噪声的插值法生成DEM；支持设置高程范围、空间分辨率和地形大致走向（方位角）。"
        self.canRunInBackground: bool = False

    def getParameterInfo(self) -> List[ap.Parameter]:
        params: List[ap.Parameter] = []

        # 输出栅格
        p_out_ws: ap.Parameter = ap.Parameter(
            displayName="输出工作空间（文件夹或地理数据库）",
            name="out_workspace",
            datatype=["DEWorkspace", "DEFolder"],
            parameterType="Required",
            direction="Input"
        )

        p_out_name: ap.Parameter = ap.Parameter(
            displayName="输出DEM名称（不含扩展名）",
            name="out_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )

        # 范围
        p_extent: ap.Parameter = ap.Parameter(
            displayName="生成范围 (xmin ymin xmax ymax)；留空则使用当前地图范围",
            name="extent_str",
            datatype="GPExtent",
            parameterType="Required",
            direction="Input"
        )

        p_clip_fc: ap.Parameter = ap.Parameter(
            displayName="可选裁剪面",
            name="clip_fc",
            datatype=["GPFeatureLayer", "DEFeatureClass"],
            parameterType="Optional",
            direction="Input"
        )

        # 高程范围
        p_minz: ap.Parameter = ap.Parameter(
            displayName="最低高程",
            name="min_z",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p_minz.value = 0.0

        p_maxz: ap.Parameter = ap.Parameter(
            displayName="最高高程",
            name="max_z",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p_maxz.value = 1000.0

        # 分辨率
        p_cellsize: ap.Parameter = ap.Parameter(
            displayName="空间分辨率（像元大小，投影单位）",
            name="cell_size",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p_cellsize.value = 30.0

        # 走向（方位角）
        p_azimuth: ap.Parameter = ap.Parameter(
            displayName="地形大致走向（方位角，度；0=北，90=东）",
            name="azimuth_deg",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p_azimuth.value = 90.0

        # 点数 & 噪声强度
        p_npts: ap.Parameter = ap.Parameter(
            displayName="用于插值的随机采样点数",
            name="n_points",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p_npts.value = 2000

        p_noise: ap.Parameter = ap.Parameter(
            displayName="噪声强度（0-1，建议0.1~0.5）",
            name="noise_strength",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p_noise.value = 0.25

        p_method: ap.Parameter = ap.Parameter(
            displayName="插值方法",
            name="interp_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p_method.filter.type = "ValueList"
        p_method.filter.list = ["IDW", "Spline"]
        p_method.value = "IDW"

        p_seed: ap.Parameter = ap.Parameter(
            displayName="随机数种子（可重复实验，留空为随机）",
            name="random_seed",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )

        p_projection: ap.Parameter = ap.Parameter(
            displayName="投影",
            name="projection",
            datatype="GPSpatialReference",
            parameterType="Required",
            direction="Input"
        )

        p_reference_feature_valley: ap.Parameter = ap.Parameter(
            displayName="局部低点参考要素（点要素或线要素）",
            name="referencing_feature_hill",
            datatype=["DEFeatureClass", "GPFeatureLayer"],
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )

        p_reference_feature_hill: ap.Parameter = ap.Parameter(
            displayName="局部高点参考要素（点要素或线要素）",
            name="referencing_feature_hill",
            datatype=["DEFeatureClass", "GPFeatureLayer"],
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )

        p_output: ap.Parameter = ap.Parameter(
            displayName="输出DEM（结果）",
            name="out_dem",
            datatype="DERasterDataset",
            parameterType="Derived",
            direction="Output"
        )

        params = [
            p_out_ws, p_out_name, p_extent, p_clip_fc,
            p_minz, p_maxz, p_cellsize, p_azimuth,
            p_npts, p_noise, p_method, p_seed, 
            p_projection, p_reference_feature_valley, p_reference_feature_hill, 
            p_output
        ]
        return params

    def isLicensed(self) -> bool:
        try:
            available: bool = ap.CheckExtension("Spatial") == "Available"
            if not available:
                return False
            ap.CheckOutExtension("Spatial")
            return True
        except Exception:
            return False

    def updateParameters(self, parameters: List[ap.Parameter]) -> None:
        if parameters[4].value and parameters[5].value:
            try:
                if float(parameters[5].value) <= float(parameters[4].value):
                    parameters[5].setErrorMessage("最高高程必须大于最低高程。")
                else:
                    parameters[5].clearMessage()
            except Exception:
                pass

        if parameters[13].value:
            try:
                if parameters[13].valueAsText == "山丘" or parameters[13].valueAsText == "山谷":
                    parameters[14].enabled = True
                    # 处理多值输入。
                    if parameters[14].value:
                    # 如果有输入值，清除错误信息
                        parameters[14].clearMessage()
                    elif not (parameters[14].value or parameters[14].valueAsText):
                        parameters[14].setErrorMessage("如果选择山丘或山谷模式，则必须选择有效的点要素或线要素。")
                else:
                    parameters[14].enabled = False
            except Exception:
                pass
        

    def updateMessages(self, parameters: List[ap.Parameter]) -> None:
        return

    def execute(self, parameters: List[ap.Parameter], messages: Any) -> None:
        # 解包参数
        out_ws: str = parameters[0].valueAsText
        out_name: str = parameters[1].valueAsText
        extent_val: ap.Extent = parameters[2].value
        clip_fc: Optional[str] = parameters[3].valueAsText
        min_z: float = float(parameters[4].valueAsText)
        max_z: float = float(parameters[5].valueAsText)
        cell_size: float = float(parameters[6].valueAsText)
        azimuth_deg: float = float(parameters[7].valueAsText)
        n_points: int = int(parameters[8].valueAsText)
        noise_strength: float = float(parameters[9].valueAsText)
        method: str = parameters[10].valueAsText
        seed: Optional[int] = int(parameters[11].valueAsText) if parameters[11].value else None
        projection: ap.SpatialReference = parameters[12].value

        # Seed RNG
        if seed is not None:
            random.seed(seed)

        # 设置环境
        env.workspace = out_ws
        env.overwriteOutput = True

        # 决定范围
        if extent_val:
            extent: ap.Extent = extent_val
        else:
            curr_extent: ap.Extent = env.extent
            if not curr_extent:
                raise ap.ExecuteError("未提供范围，且当前环境未设置env.extent。请提供范围。")
            extent = curr_extent

        xmin: float = extent.XMin
        ymin: float = extent.YMin
        xmax: float = extent.XMax
        ymax: float = extent.YMax
        ap.AddMessage(f"地形生成范围：x in: {xmin, xmax}, y in: {ymin, ymax}")

        width: float = xmax - xmin
        height: float = ymax - ymin
        if width <= 0 or height <= 0:
            raise ap.ExecuteError("范围无效（宽或高<=0）。")

        # 大致趋势向量
        theta: float = math.radians(90.0 - azimuth_deg)
        ux: float = math.cos(theta)
        uy: float = math.sin(theta)

        # 设置点位数据集，并加入点位
        temp_gdb: str = ap.management.CreateFileGDB(env.scratchFolder, f"syntTerrain_{random.randint(1000,9999)}").getOutput(0)
        ap.AddMessage(f"临时GDB: {temp_gdb}")
        env.scratchWorkspace = temp_gdb

        pts_fc: str = os.path.join(temp_gdb, "pts")
        ap.management.CreateFeatureclass(temp_gdb, "pts", "POINT", spatial_reference=projection)
        ap.management.AddField(pts_fc, "z", "DOUBLE")

        # 随机生成采样点。
        with ap.da.InsertCursor(pts_fc, ["SHAPE@XY", "z"]) as icur:
            for _ in range(n_points):
                x: float = random.random() * width + xmin
                y: float = random.random() * height + ymin

                cx: float = (xmin + xmax) / 2.0
                cy: float = (ymin + ymax) / 2.0
                rx: float = x - cx
                ry: float = y - cy
                t: float = rx * ux + ry * uy
                half_len: float = abs((width/2.0) * ux) + abs((height/2.0) * uy)
                nt: float = 0.5 + (t / (2.0 * half_len)) if half_len != 0 else 0.5
                nt = max(0.0, min(1.0, nt))

                base_z: float = min_z + nt * (max_z - min_z)
                noise: float = random.gauss(0.0, 1.0) * noise_strength * (max_z - min_z) * 0.15 * (cell_size / 1000)
                z: float = max(min_z, min(max_z, base_z + noise))
                icur.insertRow(((x, y), z))

        ap.AddMessage(f"已生成随机采样点，数量：{n_points}")

        # 保存采样点要素。
        points_path: str = os.path.join(out_ws, "points_debug")
        try:
            ap.management.CopyFeatures(pts_fc, points_path)
            ap.AddMessage(f"采样点要素已保存至: {points_path}")
        except Exception as e:
            ap.AddWarning(f"保存采样点要素失败: {str(e)}")


        # 设置范围
        env.extent = ap.Extent(xmin, ymin, xmax, ymax)
        env.cellSize = cell_size

        # 插值
        if method.upper() == "IDW":
            dem_tmp: ap.Raster = Idw(pts_fc, "z", cell_size)
        else:
            dem_tmp: ap.Raster = Spline(pts_fc, "z", weight="0.1", spline_type="REGULARIZED", cell_size=cell_size)

        # 可选裁剪
        if clip_fc:
            dem_clipped: ap.Raster = ExtractByMask(dem_tmp, clip_fc)
        else:
            dem_clipped = dem_tmp

        ap.AddMessage("地形生成并剪裁完毕，正在保存。")

        # 保存输出
        out_path: str = os.path.join(out_ws, out_name)
        out_path_unicode: str = out_path.encode("utf-8", errors="ignore").decode("utf-8")    # 防御
        dem_clipped.save(out_path_unicode)
        ap.AddMessage(f"输出DEM: {out_path_unicode}")

        parameters[-1].value = out_path_unicode
        return
    
