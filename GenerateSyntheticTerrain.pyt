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

from typing import List, Any, Dict, Optional

# 来自arcpy的导入项
from arcpy import env
from arcpy.sa import Idw, Spline, ExtractByMask

class Toolbox(object):
    def __init__(self):
        self.label: str = "地形生成工具箱"
        self.alias: str = "syntheticTerrain"
        self.tools: List = [
            GenerateTerrain, 
            GenerateMountain
        ]

class GenerateTerrain:
    def __init__(self) -> None:
        self.label: str = "步骤1：生成基础地形"
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
            p_projection, p_output
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
        azimuth_deg: float = float(parameters[7].valueAsText)
        n_points: int = int(parameters[8].valueAsText)
        noise_strength: float = float(parameters[9].valueAsText)
        method: str = parameters[10].valueAsText
        seed: Optional[int] = int(parameters[11].valueAsText) if parameters[11].value else None
        projection: ap.SpatialReference = parameters[12].value

        ap.AddMessage(f"投影变量：{projection}")

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
        ap.AddMessage(f"地形生成范围：{xmin, xmax, ymin, ymax}")

        width: float = xmax - xmin
        height: float = ymax - ymin
        if width <= 0 or height <= 0:
            raise ap.ExecuteError("范围无效（宽或高<=0）。")

        # 大致趋势向量
        theta: float = math.radians(90.0 - azimuth_deg)
        ux: float = math.cos(theta)
        uy: float = math.sin(theta)

        # 内存点位
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
                noise: float = random.gauss(0.0, 1.0) * noise_strength * (max_z - min_z) * 0.15
                z: float = max(min_z, min(max_z, base_z + noise))
                icur.insertRow(((x, y), z))

        ap.AddMessage(f"已生成随机采样点，数量：{n_points}")

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

        # 保存输出
        out_path: str = os.path.join(out_ws, out_name)
        out_path = out_path.encode("utf-8", errors="ignore").decode("utf-8")
        try:
            dem_clipped.save(out_path)
        except UnicodeDecodeError as e:
            ap.AddMessage("错误：{e}，可能因为文件已存在导致。")
        ap.AddMessage(f"输出DEM: {out_path}")

        parameters[-1].value = out_path
        return


class GenerateMountain(object):
    """
    生成算法：
    1. 先基于原始基线，生成不等长度的法向量，作为1级基线。然后以此类推生成3级基线。（数字越大，等级越低）
        - 对非原始基线而言，同级基线不得有交点，且同级基线只能与上级基线之间有1个焦点。
        - 下级基线的长度一般是上级基线的μ(0.2, 0.03)，遵循正态分布。
    2. 通过3级基线生成对应的山脉范围，使用缓冲区，以所有的基线为范围（等级越低，缓冲区范围越小），确定山脉范围。
    3. 以基线作为山脊线，每一点均必须具有头坐标及尾坐标。
        3.1 先基于所有的基线生成点位，下级基线的点位高程比上级基线更低。
        3.2 主要基线中，选择位于最中央的节点作为山脉最高点。（可偏移最多3个点的距离）

    """
    def __init__(self):
        self.label = "步骤2：生成山脉"
        self.description = "使用线要素规定山脉走向，并根据山脉走向，在步骤1的基础上，生成山脉。"
        self.canRunInBackground = False

    def isLicensed(self):
        return ap.CheckExtension("Spatial") == "Available"
    
    def getParameterInfo(self):
        params: List[ap.Parameter] = []

        # 输出工作空间
        p_out_ws = ap.Parameter(
            displayName="输出工作空间（文件夹或地理数据库）",
            name="out_workspace",
            datatype=["DEWorkspace", "DEFolder"],
            parameterType="Required",
            direction="Input"
        )
        p_out_ws.help = ["选择输出的工作空间位置。"]

        p_out_name = ap.Parameter(
            displayName="输出DEM名称（不含扩展名）",
            name="out_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )

        # 规定山脊线要素
        p_ridge_lines = ap.Parameter(
            displayName="山脊线要素",
            name="ridge_lines",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )

        # DEM来源选项
        p_dem_source = ap.Parameter(
            displayName="DEM来源",
            name="dem_source",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p_dem_source.filter.type = "ValueList"
        p_dem_source.filter.list = ["手动设置分辨率", "基于现有DEM"]
        p_dem_source.value = "手动设置分辨率"

        # 手动设置的DEM分辨率
        p_cellsize = ap.Parameter(
            displayName="DEM分辨率（米）",
            name="cell_size",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p_cellsize.value = 30.0
        p_cellsize.enabled = True

        # 基于现有DEM
        p_reference_dem = ap.Parameter(
            displayName="参考DEM（用于获取分辨率）",
            name="reference_dem",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )
        p_reference_dem.enabled = False

        # 最大噪声
        p_noise = ap.Parameter(
            displayName="随机噪声强度",
            name="noise_strength",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        p_noise.value = 0.1

        # 基线密度
        p_baseline_density = ap.Parameter(
            displayName="基线密度（每条上级基线上生成多少个下级基线，最多生成3级）",
            name="baseline_density",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p_baseline_density.value = 5

        # 基线点密度
        p_baseline_point_density = ap.Parameter(
            displayName="基线点密度（每条基线上生成多少个点）",
            name="baseline_point_density",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p_baseline_point_density.value = 5

        # 山脊宽度字段
        p_width = ap.Parameter(
            displayName="山脊宽度字段",
            name="width_field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        p_width.parameterDependencies = ["ridge_lines"]

        # 在参数列表中添加新参数
        params = [
            p_out_ws, p_out_name, p_ridge_lines,
            p_dem_source, p_cellsize, p_reference_dem,
            p_noise,
            p_baseline_density, p_baseline_point_density,
            p_width
        ]
        return params

    def updateParameters(self, parameters: List[ap.Parameter]):
        # 控制DEM分辨率相关参数的启用状态
        if parameters[3].value:  # dem_source参数
            if parameters[3].valueAsText == "手动设置分辨率":
                parameters[4].enabled = True   # cell_size
                parameters[5].enabled = False  # reference_dem
            elif parameters[3].valueAsText == "基于现有DEM":
                parameters[4].enabled = False  # cell_size
                parameters[5].enabled = True   # reference_dem
        return
    
    def updateMessages(self, parameters):
        # 验证基线密度参数
        if parameters[7].value is not None:
            try:
                density = int(parameters[7].value)
                if density <= 0:
                    parameters[7].setErrorMessage("基线密度必须大于0")
                elif density > 20:
                    parameters[7].setWarningMessage("密度过高可能导致处理时间显著增加")
            except ValueError:
                parameters[7].setErrorMessage("请输入有效的整数值")
        
        # 验证基线点密度参数
        if parameters[8].value is not None:
            try:
                point_density = int(parameters[8].value)
                if point_density <= 0:
                    parameters[8].setErrorMessage("点密度必须大于0")
                elif point_density > 50:
                    parameters[8].setWarningMessage("密度过高可能导致处理时间显著增加")
            except ValueError:
                parameters[8].setErrorMessage("请输入有效的整数值")
        
        # 验证噪声强度参数
        if parameters[6].value is not None:
            try:
                noise = float(parameters[6].value)
                if noise < 0:
                    parameters[6].setErrorMessage("噪声强度不能为负值")
                elif noise > 1:
                    parameters[6].setWarningMessage("噪声强度过高可能导致地形失真")
            except ValueError:
                parameters[6].setErrorMessage("请输入有效的数值")
        
        # 验证DEM分辨率参数
        if parameters[3].valueAsText == "手动设置分辨率":
            if parameters[4].value is not None:
                try:
                    cell_size = float(parameters[4].value)
                    if cell_size <= 0:
                        parameters[4].setErrorMessage("分辨率必须大于0")
                    elif cell_size > 1000:
                        parameters[4].setWarningMessage("分辨率值较大，可能导致处理时间过长")
                except ValueError:
                    parameters[4].setErrorMessage("请输入有效的数值")
        
        # 验证参考DEM
        elif parameters[3].valueAsText == "基于现有DEM":
            if parameters[5].value is None:
                parameters[5].setErrorMessage("请选择参考DEM")
            else:
                # 检查参考DEM是否存在
                try:
                    desc = ap.Describe(parameters[5].value)
                    if not hasattr(desc, "meanCellWidth"):
                        parameters[5].setErrorMessage("选择的图层不是有效的栅格数据集")
                except Exception:
                    parameters[5].setErrorMessage("无法读取选择的栅格数据集")
        
        return

    def execute(self, parameters: List[ap.Parameter], messages):
        """
        生成算法：
        1. 先基于原始基线，生成不等长度的法向量，作为1级基线。然后以此类推生成3级基线。（数字越大，等级越低）
            - 对非原始基线而言，同级基线不得有交点，且同级基线只能与上级基线之间有1个焦点。
            - 下级基线的长度一般是上级基线的μ(0.2, 0.03)，遵循正态分布。
            - 基线可以不必与原始线互相垂直，但必须至少有15°的夹角。
        2. 生成对应的山脉范围：使用缓冲区以所有的基线为范围（等级越低，缓冲区范围越小），确定山脉范围。
        3. 以基线作为山脊线，每一点均必须具有头坐标及尾坐标。
            3.1 先基于所有的基线生成点位，下级基线的点位高程比上级基线更低。
            3.2 主要基线中，选择位于最中央的节点作为山脉最高点。（可偏移最多3个点的距离）
        """
        
        # 解包参数
        out_ws: str = parameters[0].valueAsText
        out_name: str = parameters[1].valueAsText
        ridge_lines: ap.FeatureSet = parameters[2].value
        dem_source: str = parameters[3].valueAsText
        cell_size: float = parameters[4].value
        reference_dem: float = parameters[5].value
        noise_strength: float = parameters[6].value
        baseline_density: int = parameters[7].value
        baseline_point_density: float = parameters[8].value
        width_field: str = parameters[9].valueAsText
        
        # 设置环境
        ap.env.workspace = out_ws
        ap.env.overwriteOutput = True
        
        # 根据DEM来源确定分辨率
        if dem_source == "基于现有DEM":
            desc = ap.Describe(reference_dem)
            cell_size = desc.meanCellWidth
        
        # 获取山脊线范围
        desc = ap.Describe(ridge_lines)
        sr = desc.spatialReference
        extent = desc.extent
        
        # 创建临时工作空间
        temp_gdb_name = "mountainTemp_{}".format(random.randint(1000, 9999))
        temp_gdb = ap.management.CreateFileGDB(ap.env.scratchFolder, temp_gdb_name).getOutput(0)
        ap.AddMessage("临时GDB: {}".format(temp_gdb))
        ap.env.scratchWorkspace = temp_gdb
        
        try:
            # 步骤1: 生成基线系统
            # 读取原始山脊线
            baselines: List[Dict[str, Any]] = []
            with ap.da.SearchCursor(ridge_lines, ["SHAPE@", width_field]) as cursor:
                for row in cursor:
                    # 处理宽度字段可能为空的情况
                    width: float = row[1] if row[1] is not None else 100  # 默认宽度100米
                    baseline: Dict[str, Any] = {
                        "geometry": row[0],
                        "width": width,
                        "level": 0  # 原始基线为0级
                    }
                    baselines.append(baseline)
            
            ap.AddMessage(f"读取到{len(baselines)}条原始山脊线")
            
            # 生成下级基线 (1级到3级)
            all_baselines: List[List[Dict[str, Any]]] = [baselines[:]]  # 包含所有级别的基线
            # 基线索引使用operator[]进行。

            # 为每个级别生成下级基线
            for level in range(1, 4):  # 生成1, 2, 3级基线
                new_baselines: List[Dict[str, Any]] = []
                for baseline in all_baselines[-1]:  # 选取all_baselines的最后一个元素。
                    new_baseline_list: List[Dict[str, Any]] = self._generate_sub_baseline(baseline, baseline_density, level)

                all_baselines.extend(new_baselines)
                ap.AddMessage(f"生成{len(new_baselines)}条{level}级基线")
            
            # 步骤2: 生成山脉范围
            # 为不同级别的基线创建缓冲区
            buffered_baselines = []
            for baseline in all_baselines:
                buffer_fc = os.path.join(temp_gdb, f"buffer_{random.randint(1000, 9999)}")
                # 确保宽度不为None
                width = baseline["width"] if baseline["width"] is not None else 100
                # 缓冲区大小随级别递减
                buffer_distance = width * (0.5 ** baseline["level"])
                ap.Buffer_analysis(
                    in_features=baseline["geometry"],
                    out_feature_class=buffer_fc,
                    buffer_distance_or_field=f"{buffer_distance} Meters",
                    dissolve_option="ALL"
                )
                buffered_baselines.append(buffer_fc)
            
            # 合并所有缓冲区以确定山脉范围
            mountain_extent_fc = os.path.join(temp_gdb, "mountain_extent")
            ap.Merge_management(buffered_baselines, mountain_extent_fc)
            
            # 步骤3: 生成点位和高程
            # 创建点要素类
            pts_fc = os.path.join(temp_gdb, "mountain_points")
            ap.CreateFeatureclass_management(temp_gdb, "mountain_points", "POINT", spatial_reference=sr)
            ap.AddField_management(pts_fc, "z", "DOUBLE")
            
            # 为所有基线生成点位
            with ap.da.InsertCursor(pts_fc, ["SHAPE@XY", "z"]) as cursor:
                for baseline in all_baselines:
                    # 在基线上按点密度生成点
                    points = self._generate_points_on_baseline(
                        baseline, 
                        baseline_point_density, 
                        noise_strength
                    )
                    for point in points:
                        cursor.insertRow(point)
            
            ap.AddMessage(f"已生成{int(ap.GetCount_management(pts_fc).getOutput(0))}个点。")

            # 设置范围和像元大小
            ap.env.extent = extent
            ap.env.cellSize = cell_size
            
            # 插值生成DEM
            ap.AddMessage(f"正在通过IDW法生成山区DEM...请在山区DEM生成结束后，将该DEM和背景DEM间相加（如果你有）。")
            dem_tmp = ap.sa.Idw(pts_fc, "z", cell_size)
            
            # 保存输出
            out_path = os.path.join(out_ws, out_name)
            out_path = out_path.encode('utf-8', errors='ignore').decode('utf-8')
            dem_tmp.save(out_path)
            ap.AddMessage(f"输出DEM: {out_path}")
            
            # 设置输出位置
            parameters[-1].value = out_path
            
        except Exception as e:
            ap.AddError(f"执行过程中出现错误: {str(e)}")
            raise
        
        return
    
    def _generate_sub_baseline(
            self, 
            parent_baseline: Dict[str, Any], 
            sub_line_num: int,
            level: int
        ) -> List[Dict[str, Any]]:
        """
        生成下级基线。规则：
        1. 在上级基线上生成点。
        2. 基于该点，生成长度随机的线。

        Args:
            parent_baseline (Dict[str, Any]): 用于生成本级基线的上级基线。
            sub_line_num (int): 生成的子基线数量。
            level (int): 基线等级。
        
        Returns:
            (List[Dict[str, Any]]): 新的基线列表。
        """
        try:
            # 获取基线几何
            geom: str = parent_baseline["geometry"]
            width: float = parent_baseline["width"] if parent_baseline["width"] is not None else 100

            for _ in range(0, sub_line_num):
                # 生成下级基线长度（遵循正态分布）
                sub_length_ratio: float = random.gauss(0.3, 0.075)
                # 然后，生成对应的基线内容。
                sub_baseline = {
                    "geometry": geom,  # 实际这里应创建一个新的几何要素。
                    "width": width * sub_length_ratio,
                    "level": level
                }
            
            return sub_baseline
        except Exception:
            return None
    
    def _generate_points_on_baseline(self, baseline, point_density, noise_strength):
        """
        在基线上生成点位
        """
        points = []
        try:
            geom = baseline["geometry"]
            level = baseline["level"]
            
            # 根据基线级别确定点的数量
            num_points = point_density * (4 - level)  # 级别越高点越少
            
            # 获取基线的起点和终点
            first_point = geom.firstPoint
            last_point = geom.lastPoint
            
            # 在基线上均匀分布点
            for i in range(int(num_points)):
                # 计算点的位置比例
                ratio = i / (num_points - 1) if num_points > 1 else 0.5
                
                # 线性插值计算点坐标
                x = first_point.X + (last_point.X - first_point.X) * ratio
                y = first_point.Y + (last_point.Y - first_point.Y) * ratio
                
                # 基础高程（简化模型：起点高程到终点高程的线性变化）
                # 实际应用中应该从字段中读取起点和终点高程
                base_z = 1000 - (level * 200)  # 级别越高高程越低
                
                # 添加噪声
                if noise_strength:
                    noise = random.gauss(0, noise_strength * 100)
                    z = base_z + noise
                else:
                    z = base_z
                
                points.append(((x, y), z))
                
        except Exception as e:
            ap.AddWarning(f"生成基线点时出错: {str(e)}")
        
        return points