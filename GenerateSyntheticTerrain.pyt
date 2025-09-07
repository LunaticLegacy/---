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
from numpy.typing import NDArray

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
        p_cellsize.value = 1000.0

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
            p_maxz, p_cellsize, p_noise, p_seed, p_projection, 
            # 10~14
            p_reference_feature_valley, p_reference_feature_hill, p_reference_feature_river, p_output
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
    
    callback_count: int = 0
    @staticmethod
    def _perlin_callback(value: float) -> None:
        GenerateTerrain.callback_count += 1
        if GenerateTerrain.callback_count % 800 == 0:    
            ap.AddMessage(f"-- 地形生成进度：{value * 100 :.6f}% --")
            GenerateTerrain.callback_count = 0

    def execute(self, parameters: List[ap.Parameter], messages: Any) -> None:
        # 解包参数
        # 0~4
        out_ws: str = parameters[0].valueAsText
        out_name: str = parameters[1].valueAsText
        extent_val: ap.Extent = parameters[2].value
        clip_fc: Optional[str] = parameters[3].valueAsText
        min_z: float = float(parameters[4].valueAsText)
        # 5~9
        max_z: float = float(parameters[5].valueAsText)
        cell_size: float = float(parameters[6].valueAsText)
        noise_strength: float = float(parameters[7].valueAsText)
        seed: Optional[int] = int(parameters[8].valueAsText) if parameters[8].value else None
        projection: ap.SpatialReference = parameters[9].value
        # 10+
        ref_feature_valley: ap.FeatureSet = parameters[10].value
        ref_feature_hill: ap.FeatureSet = parameters[11].value
        ref_feature_river: ap.FeatureSet = parameters[12].value

        ap.AddMessage("数据解包完毕。")
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 解析生成范围
        if extent_val is None:
            # 如果没有指定范围，需要从当前地图获取
            raise NotImplementedError("当前版本需要指定生成范围")
        
        xmin = extent_val.XMin
        ymin = extent_val.YMin
        xmax = extent_val.XMax
        ymax = extent_val.YMax
        ap.AddMessage("开始使用柏林噪声生成基础地形...")
        
        # 通过柏林噪声生成基础地形，生成一次。
        perlin_data_base_terrain: NDArray[np.float32] = noise.batch_pnoise2(
            min_x=xmin, 
            min_y=ymin,
            max_x=xmax,
            max_y=ymax,
            resolution=cell_size,  # 每像元采样
            base=float(noise_strength) if noise_strength else 0.0,
            callback=self._perlin_callback    
        )
        # 映射到范围内。
        # perlin_data_elev: NDArray[np.float32] = min_z + (perlin_data_base_terrain + 1.0) * 0.5 * (max_z - min_z)

        ap.AddMessage("基础地形生成完毕。")

        # 然后转化为地形。
        dem_terrain: ap.Raster = ap.NumPyArrayToRaster(
            in_array=perlin_data_base_terrain,
            lower_left_corner=ap.Point(xmin, ymin),
            x_cell_size=cell_size,
            y_cell_size=cell_size,
        )

        # 设置路径，并保存
        out_path: str = os.path.join(out_ws, out_name)
        out_path = out_path.encode("utf-8", errors="ignore").decode("utf-8")
        dem_terrain.save(out_path)
        # 设置投影
        ap.DefineProjection_management(out_path, projection)

        ap.AddMessage(f"地形已保存到：{out_path}。")

        parameters[-1].value = out_path
        
        return
