import os
import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
                            QDoubleSpinBox, QProgressBar, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入各个模块
from material.material import Material
from meshing.mesher import Mesher
from analysis.boundary import BoundaryConditions
from analysis.solver import FEMSolver
from visualization.plotter import ResultsPlotter

class AnalysisThread(QThread):
    """后台分析线程，避免界面卡顿"""
    progress_updated = pyqtSignal(int)
    analysis_finished = pyqtSignal(dict)
    
    def __init__(self, mesh, material, boundary_conditions):
        super().__init__()
        self.mesh = mesh
        self.material = material
        self.boundary_conditions = boundary_conditions
        
    def run(self):
        try:
            solver = FEMSolver(self.mesh, self.material, self.boundary_conditions)
            self.progress_updated.emit(30)
            
            # 组装刚度矩阵
            solver.assemble_stiffness_matrix()
            self.progress_updated.emit(60)
            
            # 求解
            results = solver.solve()
            self.progress_updated.emit(100)
            
            self.analysis_finished.emit(results)
        except Exception as e:
            self.analysis_finished.emit({"error": str(e)})

class MainWindow(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("弹簧有限元分析软件")
        self.setGeometry(100, 100, 1000, 700)

        # 数据存储
        self.step_file = None
        self.mesh = None
        self.material = None
        self.boundary_conditions = None
        self.results = None

        # 创建主部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 创建标签页控件
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # 创建各个标签页
        self.create_import_tab()
        self.create_material_tab()
        self.create_meshing_tab()
        self.create_boundary_tab()
        self.create_analysis_tab()
        self.create_results_tab()

        # 创建导航按钮
        self.nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一步")
        self.next_btn = QPushButton("下一步")
        self.nav_layout.addWidget(self.prev_btn)
        self.nav_layout.addWidget(self.next_btn)
        self.main_layout.addLayout(self.nav_layout)

        # 连接信号和槽
        self.prev_btn.clicked.connect(self.prev_tab)
        self.next_btn.clicked.connect(self.next_tab)
        self.import_btn.clicked.connect(self.import_stl)
        self.generate_mesh_btn.clicked.connect(self.generate_mesh)
        self.set_boundary_btn.clicked.connect(self.set_boundary_conditions)
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.plot_mesh_btn.clicked.connect(self.plot_mesh)
        self.plot_displacement_btn.clicked.connect(self.plot_displacement)
        self.plot_stress_btn.clicked.connect(self.plot_stress)
        self.plot_stress_disp_btn.clicked.connect(self.plot_stress_displacement)

        # 初始化按钮状态
        self.update_nav_buttons()
        
    def create_import_tab(self):
        """创建模型导入标签页"""
        self.import_tab = QWidget()
        layout = QVBoxLayout(self.import_tab)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.import_btn = QPushButton("导入STEP文件")

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.import_btn)

        self.model_info = QLabel("未导入模型")
        self.model_info.setAlignment(Qt.AlignCenter)
        self.model_info.setStyleSheet("font-size: 14px; margin-top: 20px;")

        layout.addLayout(file_layout)
        layout.addWidget(self.model_info)
        layout.addStretch()

        self.tabs.addTab(self.import_tab, "1. STEP模型导入")
        
    def create_material_tab(self):
        """创建材料属性标签页"""
        self.material_tab = QWidget()
        layout = QVBoxLayout(self.material_tab)
        
        group = QGroupBox("材料属性")
        group_layout = QVBoxLayout(group)
        
        # 弹性模量
        em_layout = QHBoxLayout()
        em_layout.addWidget(QLabel("弹性模量 (Pa):"))
        self.elastic_modulus = QDoubleSpinBox()
        self.elastic_modulus.setRange(1e9, 1e12)
        self.elastic_modulus.setValue(2e11)  # 钢的典型值
        self.elastic_modulus.setSuffix(" Pa")
        self.elastic_modulus.setDecimals(2)
        em_layout.addWidget(self.elastic_modulus)
        group_layout.addLayout(em_layout)
        
        # 泊松比
        pr_layout = QHBoxLayout()
        pr_layout.addWidget(QLabel("泊松比:"))
        self.poisson_ratio = QDoubleSpinBox()
        self.poisson_ratio.setRange(0.0, 0.5)
        self.poisson_ratio.setValue(0.3)  # 钢的典型值
        pr_layout.addWidget(self.poisson_ratio)
        group_layout.addLayout(pr_layout)
        
        # 密度
        den_layout = QHBoxLayout()
        den_layout.addWidget(QLabel("密度 (kg/m³):"))
        self.density = QDoubleSpinBox()
        self.density.setRange(1000, 10000)
        self.density.setValue(7850)  # 钢的典型值
        den_layout.addWidget(self.density)
        group_layout.addLayout(den_layout)
        
        # 屈服强度
        ys_layout = QHBoxLayout()
        ys_layout.addWidget(QLabel("屈服强度 (Pa):"))
        self.yield_strength = QDoubleSpinBox()
        self.yield_strength.setRange(1e6, 1e9)
        self.yield_strength.setValue(250e6)  # 钢的典型值
        self.yield_strength.setSuffix(" Pa")
        group_layout.addLayout(ys_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        
        self.tabs.addTab(self.material_tab, "2. 材料属性")
        
    def create_meshing_tab(self):
        """创建网格划分标签页"""
        self.meshing_tab = QWidget()
        layout = QVBoxLayout(self.meshing_tab)

        mesh_param_layout = QVBoxLayout()

        # 网格类型
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("网格类型:"))
        self.mesh_type = QComboBox()
        self.mesh_type.addItems(["triangle", "tetrahedron"])
        self.mesh_type.setCurrentText("tetrahedron")
        type_layout.addWidget(self.mesh_type)
        mesh_param_layout.addLayout(type_layout)

        # 网格大小
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("网格大小 (m):"))
        self.mesh_size = QDoubleSpinBox()
        self.mesh_size.setRange(0.001, 1.0)  # 最大网格大小提升到1米
        self.mesh_size.setValue(0.05)        # 默认值可适当调大
        self.mesh_size.setDecimals(4)
        size_layout.addWidget(self.mesh_size)
        mesh_param_layout.addLayout(size_layout)

        # 生成网格按钮
        self.generate_mesh_btn = QPushButton("生成网格")
        mesh_param_layout.addWidget(self.generate_mesh_btn)

        # 网格信息
        self.mesh_info = QLabel("未生成网格")
        mesh_param_layout.addWidget(self.mesh_info)

        layout.addLayout(mesh_param_layout)
        layout.addStretch()

        self.tabs.addTab(self.meshing_tab, "3. 网格划分")
        
    def create_boundary_tab(self):
        """创建边界条件标签页"""
        self.boundary_tab = QWidget()
        layout = QVBoxLayout(self.boundary_tab)
        
        # 载荷设置
        load_group = QGroupBox("载荷设置")
        load_layout = QVBoxLayout(load_group)
        
        # 载荷大小
        mag_layout = QHBoxLayout()
        mag_layout.addWidget(QLabel("载荷大小 (N):"))
        self.load_magnitude = QDoubleSpinBox()
        self.load_magnitude.setRange(1, 1e6)
        self.load_magnitude.setValue(1000)
        mag_layout.addWidget(self.load_magnitude)
        load_layout.addLayout(mag_layout)
        
        # 载荷方向
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("载荷方向:"))
        self.load_direction = QComboBox()
        self.load_direction.addItems(["X轴", "Y轴", "Z轴"])
        self.load_direction.setCurrentText("Z轴")
        dir_layout.addWidget(self.load_direction)
        load_layout.addLayout(dir_layout)
        
        load_group.setLayout(load_layout)
        
        # 设置边界条件按钮
        self.set_boundary_btn = QPushButton("应用边界条件")
        
        layout.addWidget(load_group)
        layout.addWidget(self.set_boundary_btn)
        layout.addStretch()
        
        self.tabs.addTab(self.boundary_tab, "4. 边界条件")
        
    def create_analysis_tab(self):
        """创建分析求解标签页"""
        self.analysis_tab = QWidget()
        layout = QVBoxLayout(self.analysis_tab)
        
        self.run_analysis_btn = QPushButton("运行有限元分析")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.analysis_status = QLabel("等待分析...")
        
        layout.addWidget(self.run_analysis_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.analysis_status)
        layout.addStretch()
        
        self.tabs.addTab(self.analysis_tab, "5. 分析求解")
        
    def create_results_tab(self):
        """创建结果可视化标签页"""
        self.results_tab = QWidget()
        layout = QVBoxLayout(self.results_tab)
        
        # 结果按钮
        btn_layout = QHBoxLayout()
        
        self.plot_mesh_btn = QPushButton("显示网格")
        self.plot_displacement_btn = QPushButton("显示位移分布")
        self.plot_stress_btn = QPushButton("显示应力云图")
        self.plot_stress_disp_btn = QPushButton("应力-位移关系")
        
        btn_layout.addWidget(self.plot_mesh_btn)
        btn_layout.addWidget(self.plot_displacement_btn)
        btn_layout.addWidget(self.plot_stress_btn)
        btn_layout.addWidget(self.plot_stress_disp_btn)
        
        # 结果信息
        self.results_info = QLabel("尚未进行分析，无结果可显示")
        
        layout.addLayout(btn_layout)
        layout.addWidget(self.results_info)
        layout.addStretch()
        
        self.tabs.addTab(self.results_tab, "6. 结果可视化")
    
    def prev_tab(self):
        """切换到上一个标签页"""
        current_index = self.tabs.currentIndex()
        if current_index > 0:
            self.tabs.setCurrentIndex(current_index - 1)
            self.update_nav_buttons()
    
    def next_tab(self):
        """切换到下一个标签页"""
        current_index = self.tabs.currentIndex()
        if current_index < self.tabs.count() - 1:
            self.tabs.setCurrentIndex(current_index + 1)
            self.update_nav_buttons()
    
    def update_nav_buttons(self):
        """更新导航按钮状态"""
        current_index = self.tabs.currentIndex()
        self.prev_btn.setEnabled(current_index > 0)
        self.next_btn.setEnabled(current_index < self.tabs.count() - 1)
    
    def import_stl(self):
        """导入STL文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择STEP文件", "", "STEP Files (*.step *.stp)"
        )
        if file_path:
            self.step_file = file_path
            self.file_path_edit.setText(file_path)
            self.model_info.setText(f"STEP模型导入成功: {os.path.basename(file_path)}")
            QMessageBox.information(self, "成功", "STEP模型导入成功")
    
    def generate_mesh(self):
        """生成网格"""
        if not self.step_file:
            QMessageBox.warning(self, "警告", "请先导入STEP模型")
            return
        try:
            element_type = self.mesh_type.currentText()
            mesh_size = self.mesh_size.value()
            self.mesher = Mesher()
            self.mesh = self.mesher.generate_mesh(
                self.step_file,
                element_type=element_type,
                mesh_size=mesh_size
            )
            self.mesh_info.setText(
                f"网格生成成功: 节点数={len(self.mesh['nodes'])}, "
                f"单元数={len(self.mesh['elements'])}, 类型={element_type}"
            )
            QMessageBox.information(self, "成功", "网格生成成功")
        except Exception as e:
            self.mesh_info.setText(f"网格生成失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"网格生成失败: {str(e)}")
    
    def set_boundary_conditions(self):
        """设置边界条件"""
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先生成网格")
            return
        
        try:
            # 获取载荷方向向量
            dir_text = self.load_direction.currentText()
            if dir_text == "X轴":
                direction = [1, 0, 0]
            elif dir_text == "Y轴":
                direction = [0, 1, 0]
            else:  # Z轴
                direction = [0, 0, 1]
                
            # 创建材料对象
            self.material = Material(
                e=self.elastic_modulus.value(),
                nu=self.poisson_ratio.value(),
                rho=self.density.value(),
                sigma_y=self.yield_strength.value()
            )
            
            # 创建边界条件
            self.boundary_conditions = BoundaryConditions(
                self.mesh,
                load_magnitude=self.load_magnitude.value(),
                load_direction=direction
            )
            
            # 自动检测固定面和载荷面
            self.boundary_conditions.auto_detect_fixed_and_load_faces()
            
            QMessageBox.information(self, "成功", "边界条件设置成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"边界条件设置失败: {str(e)}")
    
    def run_analysis(self):
        """运行有限元分析"""
        if not all([self.mesh, self.material, self.boundary_conditions]):
            QMessageBox.warning(self, "警告", "请先完成前面的步骤")
            return
        
        try:
            # 禁用按钮
            self.run_analysis_btn.setEnabled(False)
            self.analysis_status.setText("正在进行有限元分析...")
            self.progress_bar.setValue(0)
            
            # 创建并启动分析线程
            self.analysis_thread = AnalysisThread(
                self.mesh, self.material, self.boundary_conditions
            )
            self.analysis_thread.progress_updated.connect(self.update_progress)
            self.analysis_thread.analysis_finished.connect(self.on_analysis_finished)
            self.analysis_thread.start()
        except Exception as e:
            self.analysis_status.setText(f"分析失败: {str(e)}")
            self.run_analysis_btn.setEnabled(True)
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def on_analysis_finished(self, results):
        """分析完成处理"""
        self.run_analysis_btn.setEnabled(True)
        
        if "error" in results:
            self.analysis_status.setText(f"分析失败: {results['error']}")
            QMessageBox.critical(self, "错误", f"分析失败: {results['error']}")
        else:
            self.results = results
            self.analysis_status.setText("分析完成成功")
            # 判断 von_mises 是否为空或无效
            von_mises = results.get('von_mises', None)
            if von_mises is None or len(von_mises) == 0:
                self.results_info.setText("分析完成，但未计算出有效应力结果。请检查网格和边界条件设置。")
                QMessageBox.warning(self, "警告", "分析完成，但未计算出有效应力结果。请检查网格和边界条件设置。")
            else:
                max_disp = np.max(np.linalg.norm(results['displacement'].reshape(-1,3), axis=1))
                max_stress = np.max(von_mises)
                self.results_info.setText(
                    f"分析完成: 最大位移={max_disp:.6f}m, 最大应力={max_stress:.2f}Pa"
                )
                QMessageBox.information(self, "成功", "有限元分析完成")
            # 自动切换到结果标签页
            self.tabs.setCurrentIndex(5)
    
    def plot_mesh(self):
        """显示网格"""
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先生成网格")
            return
        
        try:
            plotter = ResultsPlotter(self.mesh, {})
            plotter.plot_mesh()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示网格: {str(e)}")
    
    def plot_displacement(self):
        """显示位移分布"""
        if not self.results:
            QMessageBox.warning(self, "警告", "请先完成分析")
            return
        
        try:
            plotter = ResultsPlotter(self.mesh, self.results)
            plotter.plot_displacement()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示位移分布: {str(e)}")
    
    def plot_stress(self):
        """显示应力云图"""
        if not self.results:
            QMessageBox.warning(self, "警告", "请先完成分析")
            return
        
        try:
            plotter = ResultsPlotter(self.mesh, self.results)
            plotter.plot_stress(self.results['von_mises'])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示应力云图: {str(e)}")
    
    def plot_stress_displacement(self):
        """显示应力-位移关系"""
        if not self.results:
            QMessageBox.warning(self, "警告", "请先完成分析")
            return
        
        try:
            plotter = ResultsPlotter(self.mesh, self.results)
            plotter.plot_stress_vs_displacement()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示应力-位移关系: {str(e)}")
