import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

class ResultsPlotter:
    """结果可视化工具，用于展示有限元分析结果"""
    
    def __init__(self, mesh, results):
        """
        初始化结果可视化工具
        
        参数:
            mesh: 网格数据
            results: 有限元分析结果
        """
        self.mesh = mesh
        self.results = results
        
        self.nodes = mesh['nodes']
        self.elements = mesh['elements']
        self.element_type = mesh['type']
        
        if 'displacement' in results:
            self.displacement = results['displacement'].reshape(-1, 3)
        else:
            self.displacement = None
        
        # 创建PyVista网格用于3D可视化
        self.pv_mesh = self._create_pyvista_mesh()
        
    def _create_pyvista_mesh(self):
        """创建PyVista网格对象"""
        if self.element_type == 'triangle':
            # 创建表面网格
            faces = np.hstack([3 * np.ones((len(self.elements), 1), dtype=int), 
                              self.elements]).flatten()
            return pv.PolyData(self.nodes, faces)
        elif self.element_type == 'tetrahedron':
            # 创建体积网格
            return pv.UnstructuredGrid({pv.CellType.TETRA: self.elements}, self.nodes)
            
    def plot_mesh(self, title="有限元网格"):
        """绘制网格"""
        try:
            plotter = pv.Plotter()
            plotter.add_mesh(self.pv_mesh, show_edges=True, color='white')
            plotter.add_title(title)
            plotter.show_grid()
            plotter.show(auto_close=False)
        except Exception as e:
            print(f"网格可视化窗口关闭或异常: {e}")

    def plot_displacement(self, scale_factor=None, title="位移分布"):
        """
        绘制位移分布

        参数:
            scale_factor: 位移放大因子 (默认自动计算)
            title: 图表标题
        """
        if self.displacement is None:
            raise ValueError("结果中不包含位移数据")
        try:
            # 最大位移
            max_disp = np.max(np.linalg.norm(self.displacement, axis=1))
            # 模型尺寸（对角线长度）
            bbox_min = np.min(self.nodes, axis=0)
            bbox_max = np.max(self.nodes, axis=0)
            model_size = np.linalg.norm(bbox_max - bbox_min)

            # 自动推荐放大因子
            if scale_factor is None:
                if max_disp > 0:
                    scale_factor = max(1.0, 0.1 * model_size / max_disp)
                else:
                    scale_factor = 1.0
                print(f"[提示] 自动选择位移放大因子: {scale_factor:.2f}")

            # 计算位移后的节点位置
            displaced_nodes = self.nodes + self.displacement * scale_factor

            # 更新网格节点位置
            displaced_mesh = self.pv_mesh.copy()
            displaced_mesh.points = displaced_nodes

            # 计算位移大小作为颜色映射
            disp_magnitude = np.linalg.norm(self.displacement, axis=1)
            displaced_mesh['位移大小'] = disp_magnitude

            # 绘制
            plotter = pv.Plotter()
            # 添加原始网格作为参考
            plotter.add_mesh(self.pv_mesh, style='wireframe', color='gray', opacity=0.3)
            # 添加位移后的网格
            plotter.add_mesh(displaced_mesh, scalars='位移大小', cmap='jet',
                             show_edges=True, scalar_bar_args={'title': '位移大小 (m)'})
            plotter.add_title(title)
            plotter.show_grid()
            plotter.show(auto_close=False)
        except Exception as e:
            print(f"位移可视化窗口关闭或异常: {e}")

    def plot_stress(self, stress_values, title="应力分布"):
        """
        绘制应力分布
        
        参数:
            stress_values: 应力值数组
            title: 图表标题
        """
        try:
            # 将单元应力分配给节点
            node_stresses = np.zeros(len(self.nodes))
            node_counts = np.zeros(len(self.nodes), dtype=int)
            for i, element in enumerate(self.elements):
                for node in element:
                    node_stresses[node] += stress_values[i]
                    node_counts[node] += 1
            # 计算平均值
            for i in range(len(node_stresses)):
                if node_counts[i] > 0:
                    node_stresses[i] /= node_counts[i]
            # 将应力值添加到网格
            stress_mesh = self.pv_mesh.copy()
            stress_mesh['应力值'] = node_stresses
            # 绘制
            plotter = pv.Plotter()
            plotter.add_mesh(stress_mesh, scalars='应力值', cmap='jet', 
                             show_edges=True, scalar_bar_args={'title': '应力值 (Pa)'})
            plotter.add_title(title)
            plotter.show_grid()
            plotter.show(auto_close=False)
        except Exception as e:
            print(f"应力可视化窗口关闭或异常: {e}")
        
    def plot_stress_vs_displacement(self):
        """绘制应力-位移关系图"""
        if self.displacement is None:
            raise ValueError("结果中不包含位移数据")
            
        # 计算每个节点的位移大小
        disp_magnitude = np.linalg.norm(self.displacement, axis=1)
        
        # 计算Von Mises应力并插值到节点
        von_mises = self.results.get('von_mises', None)
        if von_mises is None:
            von_mises = self.results.get('stresses', np.zeros(len(self.elements)))
            
        node_stresses = np.zeros(len(self.nodes))
        node_counts = np.zeros(len(self.nodes), dtype=int)
        
        for i, element in enumerate(self.elements):
            for node in element:
                node_stresses[node] += von_mises[i]
                node_counts[node] += 1
                
        for i in range(len(node_stresses)):
            if node_counts[i] > 0:
                node_stresses[i] /= node_counts[i]
        
        # 绘制散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(disp_magnitude, node_stresses, alpha=0.6)
        plt.xlabel('位移大小 (m)')
        plt.ylabel('Von Mises应力 (Pa)')
        plt.title('节点应力-位移关系')
        plt.grid(True)
        plt.show()
        
    def show(self):
        """显示所有图形（如果使用matplotlib）"""
        plt.show()
