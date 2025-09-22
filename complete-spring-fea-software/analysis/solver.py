import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class FEMSolver:
    """有限元求解器，用于求解弹簧的应力和位移"""
    
    def __init__(self, mesh, material, boundary_conditions):
        """
        初始化有限元求解器
        
        参数:
            mesh: 网格数据
            material: 材料属性
            boundary_conditions: 边界条件
        """
        self.mesh = mesh
        self.material = material
        self.bc = boundary_conditions
        
        self.nodes = mesh['nodes']
        self.elements = mesh['elements']
        self.element_type = mesh['type']
        
        self.stiffness_matrix = None
        self.force_vector = None
        self.displacement = None
        self.stresses = None
        
        # 获取弹性矩阵
        self.d_matrix = material.get_elastic_matrix()
    
    def assemble_stiffness_matrix(self):
        """组装整体刚度矩阵和力向量"""
        n_nodes = len(self.nodes)
        n_dofs = 3 * n_nodes  # 每个节点3个自由度
        
        # 初始化刚度矩阵（稀疏矩阵）
        self.stiffness_matrix = lil_matrix((n_dofs, n_dofs), dtype=np.float64)
        
        # 初始化力向量
        self.force_vector = self.bc.node_forces.copy()
        
        # 遍历所有单元，计算单元刚度矩阵并组装到整体刚度矩阵
        for element_id, element_nodes in enumerate(self.elements):
            # 获取单元节点坐标
            coords = self.nodes[element_nodes]
            
            # 计算单元刚度矩阵
            if self.element_type == 'triangle':
                ke = self.calculate_triangle_stiffness(coords)
                dofs = np.array([[3*i, 3*i+1, 3*i+2] for i in element_nodes]).flatten()
            else:  # tetrahedron
                ke = self.calculate_tetrahedron_stiffness(coords)
                dofs = np.array([[3*i, 3*i+1, 3*i+2] for i in element_nodes]).flatten()
            
            # 组装到整体刚度矩阵
            for i in range(len(dofs)):
                for j in range(len(dofs)):
                    self.stiffness_matrix[dofs[i], dofs[j]] += ke[i, j]
    
    def calculate_triangle_stiffness(self, coords):
        """计算三角形单元的刚度矩阵（平面应力）"""
        # 提取节点坐标
        x1, y1, z1 = coords[0]
        x2, y2, z2 = coords[1]
        x3, y3, z3 = coords[2]
        
        # 计算B矩阵（应变-位移矩阵）
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
        
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        
        b_matrix = (1/(2*area)) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])
        
        # 计算单元刚度矩阵 (面积 * B^T * D * B)
        return area * b_matrix.T @ self.d_matrix[:3, :3] @ b_matrix
    
    def calculate_tetrahedron_stiffness(self, coords):
        """计算四面体单元的刚度矩阵"""
        # 提取节点坐标
        p1, p2, p3, p4 = coords
        
        # 计算体积和B矩阵
        v = self.calculate_tetrahedron_volume(p1, p2, p3, p4)
        
        # 构建矩阵
        a = np.array([
            [1, p1[0], p1[1], p1[2]],
            [1, p2[0], p2[1], p2[2]],
            [1, p3[0], p3[1], p3[2]],
            [1, p4[0], p4[1], p4[2]]
        ])
        
        # 计算伴随矩阵
        m = np.zeros((3, 4))
        for i in range(4):
            m[:, i] = np.linalg.det(self.minor(a, 0, i))
        
        # 构建B矩阵
        b_matrix = np.zeros((6, 12))
        for i in range(4):
            b_matrix[0, 3*i] = m[0, i]
            b_matrix[1, 3*i+1] = m[1, i]
            b_matrix[2, 3*i+2] = m[2, i]
            b_matrix[3, 3*i] = m[1, i]
            b_matrix[3, 3*i+1] = m[0, i]
            b_matrix[4, 3*i+1] = m[2, i]
            b_matrix[4, 3*i+2] = m[1, i]
            b_matrix[5, 3*i] = m[2, i]
            b_matrix[5, 3*i+2] = m[0, i]
        
        b_matrix /= (6 * v)
        
        # 计算单元刚度矩阵 (体积 * B^T * D * B)
        return v * b_matrix.T @ self.d_matrix @ b_matrix
    
    def calculate_tetrahedron_volume(self, p1, p2, p3, p4):
        """计算四面体体积"""
        return abs(np.dot(np.cross(p2-p1, p3-p1), p4-p1)) / 6
    
    def minor(self, matrix, i, j):
        """计算矩阵的余子式"""
        return matrix[np.array(list(range(i)) + list(range(i+1, matrix.shape[0])))[:, np.newaxis],
                      np.array(list(range(j)) + list(range(j+1, matrix.shape[1])))]
    
    def solve(self):
        """求解有限元方程，得到位移和应力"""
        if self.stiffness_matrix is None:
            self.assemble_stiffness_matrix()
        
        # 应用边界条件
        reduced_k, reduced_f = self.bc.apply_boundary_conditions(
            self.stiffness_matrix.tocsr(), 
            self.force_vector
        )
        
        # 求解线性方程组
        reduced_u = spsolve(reduced_k, reduced_f)
        
        # 扩展到位移向量
        self.displacement = self.bc.expand_displacement(reduced_u)
        
        # 计算应力
        self.calculate_stresses()
        
        # 计算Von Mises应力
        von_mises = self.calculate_von_mises_stress()
        
        return {
            'displacement': self.displacement,
            'stresses': self.stresses,
            'von_mises': von_mises
        }
    
    def calculate_stresses(self):
        """计算每个单元的应力"""
        n_elements = len(self.elements)
        self.stresses = np.zeros((n_elements, 6))  # 存储每个单元的6个应力分量
        
        for element_id, element_nodes in enumerate(self.elements):
            coords = self.nodes[element_nodes]
            u_element = np.array([self.displacement[3*i:3*i+3] for i in element_nodes]).flatten()
            
            # 计算B矩阵和应力
            if self.element_type == 'triangle':
                # 提取节点坐标
                x1, y1, z1 = coords[0]
                x2, y2, z2 = coords[1]
                x3, y3, z3 = coords[2]
                
                # 计算面积和B矩阵
                area = 0.5 * abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
                
                b1 = y2 - y3
                b2 = y3 - y1
                b3 = y1 - y2
                c1 = x3 - x2
                c2 = x1 - x3
                c3 = x2 - x1
                
                b_matrix = (1/(2*area)) * np.array([
                    [b1, 0, b2, 0, b3, 0],
                    [0, c1, 0, c2, 0, c3],
                    [c1, b1, c2, b2, c3, b3]
                ])
                
                # 计算应力
                stress = self.d_matrix[:3, :3] @ b_matrix @ u_element
                # 补全6个分量（平面应力假设）
                self.stresses[element_id] = np.array([
                    stress[0], stress[1], 0, stress[2], 0, 0
                ])
                
            else:  # tetrahedron
                # 提取节点坐标
                p1, p2, p3, p4 = coords
                
                # 计算体积和B矩阵
                v = self.calculate_tetrahedron_volume(p1, p2, p3, p4)
                
                # 构建矩阵
                a = np.array([
                    [1, p1[0], p1[1], p1[2]],
                    [1, p2[0], p2[1], p2[2]],
                    [1, p3[0], p3[1], p3[2]],
                    [1, p4[0], p4[1], p4[2]]
                ])
                
                # 计算伴随矩阵
                m = np.zeros((3, 4))
                for i in range(4):
                    m[:, i] = np.linalg.det(self.minor(a, 0, i))
                
                # 构建B矩阵
                b_matrix = np.zeros((6, 12))
                for i in range(4):
                    b_matrix[0, 3*i] = m[0, i]
                    b_matrix[1, 3*i+1] = m[1, i]
                    b_matrix[2, 3*i+2] = m[2, i]
                    b_matrix[3, 3*i] = m[1, i]
                    b_matrix[3, 3*i+1] = m[0, i]
                    b_matrix[4, 3*i+1] = m[2, i]
                    b_matrix[4, 3*i+2] = m[1, i]
                    b_matrix[5, 3*i] = m[2, i]
                    b_matrix[5, 3*i+2] = m[0, i]
                
                b_matrix /= (6 * v)
                
                # 计算应力
                self.stresses[element_id] = self.d_matrix @ b_matrix @ u_element
    
    def calculate_von_mises_stress(self):
        """计算每个单元的Von Mises应力"""
        if self.stresses is None:
            return None
            
        von_mises = np.zeros(len(self.stresses))
        for i, stress in enumerate(self.stresses):
            s11, s22, s33, s12, s23, s13 = stress
            von_mises[i] = np.sqrt(
                0.5 * ((s11 - s22)**2 + (s22 - s33)** 2 + (s33 - s11)**2 +
                       6 * (s12**2 + s23**2 + s13**2))
            )
        return von_mises
