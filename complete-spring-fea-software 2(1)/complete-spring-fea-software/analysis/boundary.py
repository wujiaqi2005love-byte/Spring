import numpy as np

class BoundaryConditions:
    """边界条件管理类，处理约束和载荷"""
    
    def __init__(self, mesh, load_magnitude=1000.0, load_direction=[0, 0, 1]):
        """
        初始化边界条件
        参数:
            mesh: 网格数据
            load_magnitude: 载荷大小
            load_direction: 载荷方向向量（X, Y, Z）
        """
        self.mesh = mesh
        self.nodes = mesh['nodes']
        self.elements = mesh['elements']
        
        # 载荷设置
        self.load_magnitude = load_magnitude
        self.load_direction = np.array(load_direction, dtype=np.float64)
        self.load_direction /= np.linalg.norm(self.load_direction)  # 归一化
        
        # 约束设置
        self.fixed_dofs = []  # 固定的自由度
        self.loaded_nodes = []  # 受载节点
        self.node_forces = np.zeros(3 * len(self.nodes))  # 节点力向量
    
    def auto_detect_fixed_and_load_faces(self):
        """自动检测并设置固定面（底部）和载荷面（顶部），严格选取主方向最大/最小坐标的所有节点"""
        main_axis = np.argmax(np.abs(self.load_direction))
        node_coords = self.nodes[:, main_axis]
        min_coord = np.min(node_coords)
        max_coord = np.max(node_coords)
        # 固定点：选取最小坐标的单个节点
        fixed_node = np.argmin(node_coords)
        for dof in range(3):
            self.fixed_dofs.append(3 * fixed_node + dof)
        # 载荷点：选取最大坐标的单个节点
        loaded_node = np.argmax(node_coords)
        self.loaded_nodes = np.array([loaded_node])
        force_per_node = self.load_magnitude
        for i in range(3):
            self.node_forces[3 * loaded_node + i] = force_per_node * self.load_direction[i]
    
    def get_stiffness_matrix_mask(self):
        """生成刚度矩阵的掩码，用于处理固定自由度"""
        n_dofs = 3 * len(self.nodes)
        mask = np.ones(n_dofs, dtype=bool)
        mask[self.fixed_dofs] = False
        return mask
    
    def apply_boundary_conditions(self, stiffness_matrix, force_vector):
        """
        应用边界条件到刚度矩阵和力向量
        
        参数:
            stiffness_matrix: 刚度矩阵
            force_vector: 力向量
            
        返回:
            处理后的刚度矩阵和力向量
        """
        # 创建掩码，掩码是一个布尔数组，表示哪些自由度是未固定的
        mask = self.get_stiffness_matrix_mask()
        
        # 应用掩码到刚度矩阵
        reduced_stiffness = stiffness_matrix[mask][:, mask]
        
        # 应用掩码到力向量
        reduced_force = force_vector[mask]
        
        return reduced_stiffness, reduced_force
    
    def expand_displacement(self, reduced_displacement):
        """
        将简化的位移向量扩展为完整的位移向量
        
        参数:
            reduced_displacement: 简化的位移向量
            
        返回:
            完整的位移向量
        """
        mask = self.get_stiffness_matrix_mask()
        full_displacement = np.zeros(3 * len(self.nodes))
        full_displacement[mask] = reduced_displacement
        return full_displacement
