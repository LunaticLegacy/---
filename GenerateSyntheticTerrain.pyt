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

        p_terrain_type: ap.Parameter = ap.Parameter(
            displayName="地形类型",
            name="terrain_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p_terrain_type.filter.type = "ValueList"
        p_terrain_type.filter.list = ["坡状", "山丘", "山谷"]
        p_terrain_type.value = "坡状"

        p_reference_feature: ap.Parameter = ap.Parameter(
            displayName="参考要素（点或线要素）",
            name="referencing_feature",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Input"
        )
        p_reference_feature.enabled = False

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
            p_projection, p_terrain_type, p_reference_feature, 
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
    

class GenerateMountain(object):
    """
    生成算法：
    1. 先基于原始基线，生成不等长度的法向量，作为1级基线。然后以此类推生成3级基线。（数字越大，等级越低）
        1.1 先按照基点密度，在线上生成特定数量的基点。
        1.2 随后，对每一个基点而言，基点作为原点，按照如下规则拉出射线（并将其截断在下级基线长度中）：
            - 对非原始基线而言，同级基线不得有交点，且同级基线只能与上级基线之间有1个焦点。
            - 下级基线的长度一般是上级基线的μ(0.2, 0.03)，遵循正态分布。
            - 基线可以不必与原始线互相垂直，但必须至少保证在该点上与基线呈15°的夹角。
            - 基线的远端距离不得长于基线长度。
            - 最低级基线不再执行拉射线操作。
    2. 生成对应的山脉范围：
        2.1 使用缓冲区以所有的基线为范围（等级越低，缓冲区范围越小），确定山脉范围。
    3. 以基线作为山脊线，对各点位赋值。遵循如下规则：
        - 更高级基线上基点的高度均值必须整体大于下级基线。
        - 距山脊线越高，点位高度越高。但点位高度不得高于最近的山脊点。
    4. 插值。
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
            parameterType="Optional",
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
            displayName="基线点密度（在每条基线的泰森多边形区域内附近生成多少个点）",
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
        执行算法。
        """
        
        # 解包参数
        out_ws: str = parameters[0].valueAsText
        out_name: str = parameters[1].valueAsText
        ridge_lines: str = parameters[2].valueAsText
        dem_source: str = parameters[3].valueAsText
        cell_size: float = parameters[4].value
        reference_dem: str = parameters[5].value
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

        # 步骤1: 生成基线系统
        # 读取原始山脊线
        baselines: List[List[Line]] = []    # 一个用于读取山脊线的程序。
        basepoints: List[List[Point]] = []

        with ap.da.SearchCursor(ridge_lines, ["SHAPE@", width_field]) as cursor:
            init_baseline: List[Line] = []
            for row in cursor:
                # 处理宽度字段可能为空的情况
                width: float = row[1] if row[1] is not None else 100  # 默认宽度100米
                now_baseline: Line = {
                    "geometry": row[0],
                    "width": width,
                    "level": 0  # 原始基线为0级
                }
                init_baseline.append(now_baseline)
            baselines.append(init_baseline)

        ap.AddMessage(f"读取到{len(baselines)}条原始山脊线，正在生成下级基线...")

        # 第一步：开始生成各级基线。
        for level in range(1, 1+3):
            # 开始生成基线，每次取最终元素。
            new_baseline: List[Line] = []
            new_points: List[List[Point]] = []
            for line in baselines[-1]:
                now_lines, init_points = self._generate_sub_baseline(
                    parent_baseline=line, 
                    sub_line_num=baseline_density, 
                    level=level, 
                    noice_strength=noise_strength
                )
                new_baseline.extend(now_lines),
                new_points.extend(init_points)
            # 加入点。
            baselines.append(new_baseline)
            basepoints.append(new_points)

            ap.AddMessage(f"{level}级基线生成完毕，共计生成{len(new_baseline)}条，及{len(new_points)}个点。")

        # 第二步：生成山脉范围（缓冲区）
        buffers: List[str] = []  # 存储各级缓冲结果的路径

        ap.AddMessage("正在生成山脉范围...")

        for level, lines in enumerate(baselines):
            if not lines:
                continue

            # 生成该级别的要素类
            fc_name: str = f"baseline_level{level}_{random.randint(100,999)}"
            fc_path: str = os.path.join(temp_gdb, fc_name)
            ap.CreateFeatureclass_management(
                out_path=temp_gdb,
                out_name=fc_name,
                geometry_type="POLYLINE",
                spatial_reference=sr
            )
            ap.AddField_management(fc_path, "WIDTH", "DOUBLE")

            # 批量写入该级别的基线
            with ap.da.InsertCursor(fc_path, ["SHAPE@", "WIDTH"]) as cursor:
                for line in lines:
                    width: float = line["width"] * (1 - level * 0.1)  # 级别越低，缓冲区越窄
                    cursor.insertRow([line["geometry"], width])

            # 一次性生成该级别的缓冲区
            buf_fc: str = os.path.join(temp_gdb, f"buffer_level{level}")
            ap.Buffer_analysis(fc_path, buf_fc, "WIDTH")
            buffers.append(buf_fc)

            ap.AddMessage(f"{level}级山脉缓冲区生成完毕，共 {len(lines)} 条基线。输出到：{buf_fc}")

        # 合并所有级别缓冲区
        mountain_area: str = os.path.join(temp_gdb, "mountain_area")
        merged_fc: str = os.path.join(temp_gdb, "all_buffers")
        ap.Merge_management(buffers, merged_fc)
        ap.Dissolve_management(merged_fc, mountain_area)

        ap.AddMessage(f"所有缓冲区合并完成，山脉范围输出：{mountain_area}")

        # 第三步：山脊点赋值，并准备进行插值。
        # 收集所有点
        ap.AddMessage("正在为山脊点赋值...")
        all_points = []
        for level, pts_group in enumerate(basepoints):
            for pt in pts_group:
                all_points.append([pt[0][0], pt[0][1], pt[1]])

        # 存储为点要素类
        sr = desc.spatialReference
        point_fc = os.path.join(temp_gdb, "ridge_points")
        ap.CreateFeatureclass_management(temp_gdb, "ridge_points", "POINT", spatial_reference=sr)
        ap.AddField_management(point_fc, "Z", "DOUBLE")

        with ap.da.InsertCursor(point_fc, ["SHAPE@XY", "Z"]) as cursor:
            for x, y, z in all_points:
                cursor.insertRow(((x, y), z))

        # 插值
        ap.AddMessage("插值中...")
        dem_final = Spline(point_fc, "Z", cell_size)

        # 保存 DEM
        out_dem: str = os.path.join(out_ws, out_name)
        dem_final.save(out_dem)
        ap.AddMessage(f"DEM已生成在: {out_dem}")
            
        parameters[-1] = out_dem

        return
    
    def _generate_sub_baseline(
            self, 
            parent_baseline: Line, 
            sub_line_num: int,
            level: int,
            noice_strength: float
        ) -> Tuple[List[Line], List[Point]]:
        """
        生成下级基线。规则：
        - 对非原始基线而言，同级基线不得有交点，且同级基线只能与上级基线之间有1个焦点。
        - 下级基线的长度一般是上级基线的μ(0.2, 0.03)，遵循正态分布。
        - 基线可以不必与原始线互相垂直，但必须至少保证在该点上与基线呈15°的夹角。
        - 基线的远端距离不得长于基线长度。
        - 最低级基线不再执行拉射线操作。

        Args:
            parent_baseline (Line): 用于生成本级基线的上级基线。
            sub_line_num (int): 生成的子基线数量。
            level (int): 基线等级。
        
        Returns:
            (Tuple[List[Line], List[Point]]): 新的基线列表，和对应的基点列表。
        """
        geom = parent_baseline["geometry"]
        width = parent_baseline["width"] if parent_baseline["width"] is not None else 100

        # 在基线上生成基点
        points: List[Point] = self._generate_points_on_baseline(
            parent_baseline, sub_line_num, noice_strength
        )
        lines: List[Line] = []

        # 遍历基点，生成子线
        for pt, z in points:
            x0, y0 = pt

            # 基线方向（取线段首尾点方向）
            start, end = geom.firstPoint, geom.lastPoint
            dx, dy = end.X - start.X, end.Y - start.Y
            base_angle = math.atan2(dy, dx)

            # 随机角度偏移，必须保证与原始线夹角 >= 15°
            offset_angle = math.radians(random.uniform(30, 150))
            if random.random() < 0.5:
                offset_angle = -offset_angle
            new_angle = base_angle + offset_angle

            # 下级基线长度（高斯分布，限制不超过父线长度）
            parent_length = geom.length
            sub_length = min(
                parent_length, 
                abs(random.gauss(0.2, 0.03)) * parent_length
            )

            # 计算终点
            x1 = x0 + sub_length * math.cos(new_angle)
            y1 = y0 + sub_length * math.sin(new_angle)

            # 构建新的 polyline
            array = ap.Array([ap.Point(x0, y0), ap.Point(x1, y1)])
            new_geom = ap.Polyline(array, geom.spatialReference)

            sub_baseline: Line = {
                "geometry": new_geom,
                "width": width * (sub_length / parent_length),
                "level": level
            }
            lines.append(sub_baseline)

        return (lines, points)

    def _generate_points_on_baseline(
            self, 
            baseline: Line, 
            point_density: float, 
            noise_strength: float
        ) -> List[Point]:
        """
        在基线上生成点位。
        Args:
            baseline (Line): 目标基线。
            point_density (float): 该基线上生成的点数量。
            noise_strength (float): 噪音强度。
        """
        points: List[Point] = []

        geom = baseline["geometry"]
        level = baseline["level"]
        
        # 根据基线级别确定点的数量
        num_points: int = point_density * (4 - level)  # 级别越高点越少 
        
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
        
        return points