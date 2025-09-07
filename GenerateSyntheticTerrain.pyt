# -*- coding: utf-8 -*-
# GenerateSyntheticTerrain.pyt
# WIP
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
        self.description: str = "一个用于生成地形的工具，可预先设置山脉走向和水系。"
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

        p_reference_feature_river: ap.Parameter = ap.Parameter(
            displayName="水域参考要素（线要素或面要素）",
            name="referencing_feature_water",
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
            # 0~4
            p_out_ws, p_out_name, p_extent, p_clip_fc, p_minz,
            # 5~9
            p_maxz, p_cellsize, p_noise, p_method, p_seed, 
            # 10~14
            p_projection, p_reference_feature_valley, p_reference_feature_hill, p_reference_feature_river, p_output
            # 15+
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
        noise_strength: float = float(parameters[7].valueAsText)
        method: str = parameters[8].valueAsText
        seed: Optional[int] = int(parameters[9].valueAsText) if parameters[9].value else None
        projection: ap.SpatialReference = parameters[10].value
        ref_feature_valley: ap.FeatureSet = parameters[11].value
        ref_feature_hill: ap.FeatureSet = parameters[12].value

