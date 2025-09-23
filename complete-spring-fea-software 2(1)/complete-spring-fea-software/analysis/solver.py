import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class FEMSolver:
    """有限元求解器，用于求解弹簧的应力和位移"""

    def __init__(self, mesh, material, boundary_conditions):
        """
        参数:
            mesh: {'nodes': ndarray (n_nodes,3), 'elements': list/ndarray of index lists, 'type': 'triangle'/'tetrahedron'}
            material: 对象，必须实现 get_elastic_matrix() 返回 6x6 (弹性矩阵 D) 或至少适用的子矩阵
            boundary_conditions: 对象，需包含:
                - node_forces: ndarray (3*n_nodes,) 总力向量
                - apply_boundary_conditions(K_csr, f) -> reduced_K, reduced_f
                - expand_displacement(reduced_u) -> full_u (3*n_nodes,)
        """
        self.mesh = mesh
        self.material = material
        self.bc = boundary_conditions

        self.nodes = np.asarray(mesh['nodes'], dtype=np.float64)
        self.elements = list(mesh['elements'])
        self.element_type = mesh['type']

        self.stiffness_matrix = None
        self.force_vector = None
        self.displacement = None
        self.stresses = None

        # 获取弹性矩阵（假设返回 6x6 全矩阵）
        self.d_matrix = np.asarray(material.get_elastic_matrix(), dtype=np.float64)

    # ---------------- 组装 ----------------
    def assemble_stiffness_matrix(self):
        """组装整体刚度矩阵和力向量"""
        n_nodes = len(self.nodes)
        n_dofs = 3 * n_nodes  # 每节点3个自由度

        # 初始化刚度矩阵（稀疏矩阵）
        self.stiffness_matrix = lil_matrix((n_dofs, n_dofs), dtype=np.float64)

        # 初始化力向量，需要与 n_dofs 大小一致
        if getattr(self.bc, 'node_forces', None) is None:
            self.force_vector = np.zeros(n_dofs, dtype=np.float64)
        else:
            self.force_vector = np.asarray(self.bc.node_forces, dtype=np.float64).copy()
            if self.force_vector.size != n_dofs:
                raise ValueError(f"bc.node_forces 大小应为 {n_dofs}，但得到 {self.force_vector.size}")

        # 遍历单元并装配
        for element_id, element_nodes in enumerate(self.elements):
            element_nodes = np.asarray(element_nodes, dtype=int)
            coords = self.nodes[element_nodes]

            if self.element_type == 'triangle':
                ke = self.calculate_triangle_stiffness(coords)  # 返回 9x9（3 nodes * 3 DOF）
            else:  # tetrahedron
                ke = self.calculate_tetrahedron_stiffness(coords)  # 返回 12x12

            dofs = np.array([[3*i, 3*i+1, 3*i+2] for i in element_nodes]).flatten()
            # 组装
            for i_local in range(len(dofs)):
                for j_local in range(len(dofs)):
                    self.stiffness_matrix[dofs[i_local], dofs[j_local]] += ke[i_local, j_local]

    # ---------------- 三角形单元（平面应力） ----------------
    def calculate_triangle_stiffness(self, coords):
        """
        coords: shape (3,3) (x,y,z)，使用 x,y 计算平面单元刚度（平面应力），
        返回 9x9（每个节点3 DOF，映射 ux,uy 到 2D 刚度分量，uz 不参与）
        """
        # 只用 x,y 分量
        x1, y1 = coords[0, 0], coords[0, 1]
        x2, y2 = coords[1, 0], coords[1, 1]
        x3, y3 = coords[2, 0], coords[2, 1]

        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
        if area <= 1e-14:
            raise ValueError("三角形单元面积接近 0，检查网格质量或节点顺序。")

        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        B2d = (1.0/(2.0*area)) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])  # 3x6

        # 使用 d_matrix 的 2D 部分（假设弹性矩阵 D 支持索引前 3x3）
        D2d = self.d_matrix[:3, :3]

        ke2d = area * (B2d.T @ D2d @ B2d)  # 6x6

        # map 2D DOF indices (ux,uy per node) 至 3D DOF 排列 (ux,uy,uz)
        # local 3*3 = 9 主自由度顺序: [u0x,u0y,u0z,u1x,u1y,u1z,u2x,u2y,u2z]
        idx2d = [0, 1, 3, 4, 6, 7]  # 对应 2D 索引
        ke9 = np.zeros((9, 9), dtype=np.float64)
        # 将 6x6 嵌入到 9x9
        ke9[np.ix_(idx2d, idx2d)] = ke2d

        # 注意：这里 uz 与 ux,uy完全解耦（ke9 中对应 uz 行列仍为 0）。
        # 如果你希望考虑板/壳的厚度或 z 方向刚度，需要替换为 3D 壳/实体单元模型。
        return ke9

    # ---------------- 四面体单元 ----------------
    def calculate_tetrahedron_stiffness(self, coords):
        """
        coords: shape (4,3)
        返回 12x12 刚度矩阵
        使用标准线性四面体：构造矩阵 A = [[1 x y z]...], inv(A) 得到形函数系数
        """
        p1, p2, p3, p4 = coords
        v = self.calculate_tetrahedron_volume(p1, p2, p3, p4)
        if v <= 1e-18:
            raise ValueError("四面体体积接近 0，检查网格质量或节点顺序。")

        A = np.array([
            [1.0, p1[0], p1[1], p1[2]],
            [1.0, p2[0], p2[1], p2[2]],
            [1.0, p3[0], p3[1], p3[2]],
            [1.0, p4[0], p4[1], p4[2]]
        ], dtype=np.float64)

        invA = np.linalg.inv(A)  # 4x4
        # 形函数 N_i = invA[0,i] + invA[1,i]*x + invA[2,i]*y + invA[3,i]*z
        # grad N_i = [invA[1,i], invA[2,i], invA[3,i]]
        grads = invA[1:4, :]  # 3x4，列 i 为节点 i 的梯度 (b_i, c_i, d_i)

        # 构建 B 矩阵 (6 x 12)
        B = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            bi, ci, di = grads[0, i], grads[1, i], grads[2, i]
            B[0, 3*i + 0] = bi
            B[1, 3*i + 1] = ci
            B[2, 3*i + 2] = di
            B[3, 3*i + 0] = ci
            B[3, 3*i + 1] = bi
            B[4, 3*i + 1] = di
            B[4, 3*i + 2] = ci
            B[5, 3*i + 0] = di
            B[5, 3*i + 2] = bi

        # 刚度矩阵 Ke = volume * B^T * D * B
        ke = v * (B.T @ self.d_matrix @ B)
        return ke

    def calculate_tetrahedron_volume(self, p1, p2, p3, p4):
        return abs(np.dot(np.cross(p2 - p1, p3 - p1), (p4 - p1))) / 6.0

    # ---------------- 辅助（删除：原 minor） ----------------
    def minor(self, matrix, i, j):
        """返回矩阵删除第 i 行第 j 列后的子矩阵（备用，但主实现已改用 np.delete）"""
        # 兼容地实现
        return np.delete(np.delete(matrix, i, axis=0), j, axis=1)

    # ---------------- 求解 ----------------
    def solve(self):
        """求解有限元方程，得到位移和应力"""
        if self.stiffness_matrix is None:
            self.assemble_stiffness_matrix()

        # 应用边界条件：这里假设 bc.apply_boundary_conditions 返回稀疏矩阵和载荷
        reduced_k, reduced_f = self.bc.apply_boundary_conditions(
            self.stiffness_matrix.tocsr(),
            self.force_vector
        )

        # 求解线性方程组
        reduced_u = spsolve(reduced_k, reduced_f)

        # 扩展到全位移向量（3*n_nodes）
        self.displacement = self.bc.expand_displacement(reduced_u)

        # 计算应力
        self.calculate_stresses()

        # 计算 Von Mises 应力
        von_mises = self.calculate_von_mises_stress()

        return {
            'displacement': self.displacement,
            'stresses': self.stresses,
            'von_mises': von_mises
        }

    # ---------------- 计算单元应力 ----------------
    def calculate_stresses(self):
        """计算每个单元的应力（返回每个单元 6 分量：s11,s22,s33,s12,s23,s13）"""
        n_elements = len(self.elements)
        self.stresses = np.zeros((n_elements, 6), dtype=np.float64)

        for element_id, element_nodes in enumerate(self.elements):
            element_nodes = np.asarray(element_nodes, dtype=int)
            coords = self.nodes[element_nodes]
            # 组装单元位移向量（3* n_nodes_per_element）
            u_element = np.concatenate([self.displacement[3*i:3*i+3] for i in element_nodes])

            if self.element_type == 'triangle':
                # 计算与刚度一致的 B 矩阵（同 calculate_triangle_stiffness 中）
                x1, y1 = coords[0, 0], coords[0, 1]
                x2, y2 = coords[1, 0], coords[1, 1]
                x3, y3 = coords[2, 0], coords[2, 1]
                area = 0.5 * abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
                if area <= 1e-14:
                    raise ValueError("三角形单元面积接近 0，检查网格质量或节点顺序。")
                b1 = y2 - y3
                b2 = y3 - y1
                b3 = y1 - y2
                c1 = x3 - x2
                c2 = x1 - x3
                c3 = x2 - x1
                B2d = (1.0/(2.0*area)) * np.array([
                    [b1, 0, b2, 0, b3, 0],
                    [0, c1, 0, c2, 0, c3],
                    [c1, b1, c2, b2, c3, b3]
                ])
                # 取 u_element 的 ux,uy 分量
                u2d = u_element[[0,1,3,4,6,7]]
                stress2d = self.d_matrix[:3, :3] @ (B2d @ u2d)
                # expand to 6 comps: s11,s22,s33(=0), s12, s23=0, s13=0
                self.stresses[element_id, :] = np.array([stress2d[0], stress2d[1], 0.0, stress2d[2], 0.0, 0.0])
            else:
                # 四面体：用 invA 方法构建 B
                p1, p2, p3, p4 = coords
                v = self.calculate_tetrahedron_volume(p1, p2, p3, p4)
                if v <= 1e-18:
                    raise ValueError("四面体体积接近 0，检查网格质量或节点顺序。")
                A = np.array([
                    [1.0, p1[0], p1[1], p1[2]],
                    [1.0, p2[0], p2[1], p2[2]],
                    [1.0, p3[0], p3[1], p3[2]],
                    [1.0, p4[0], p4[1], p4[2]]
                ], dtype=np.float64)
                invA = np.linalg.inv(A)
                grads = invA[1:4, :]  # 3x4
                B = np.zeros((6, 12), dtype=np.float64)
                for i in range(4):
                    bi, ci, di = grads[0, i], grads[1, i], grads[2, i]
                    B[0, 3*i + 0] = bi
                    B[1, 3*i + 1] = ci
                    B[2, 3*i + 2] = di
                    B[3, 3*i + 0] = ci
                    B[3, 3*i + 1] = bi
                    B[4, 3*i + 1] = di
                    B[4, 3*i + 2] = ci
                    B[5, 3*i + 0] = di
                    B[5, 3*i + 2] = bi
                stress = self.d_matrix @ (B @ u_element)
                self.stresses[element_id, :] = stress

    # ---------------- Von Mises ----------------
    def calculate_von_mises_stress(self):
        """计算每个单元的Von Mises应力"""
        if self.stresses is None:
            return None

        von_mises = np.zeros(len(self.stresses), dtype=np.float64)
        for i, stress in enumerate(self.stresses):
            s11, s22, s33, s12, s23, s13 = stress
            von_mises[i] = np.sqrt(
                0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) +
                3.0 * (s12**2 + s23**2 + s13**2)
            )
        return von_mises
