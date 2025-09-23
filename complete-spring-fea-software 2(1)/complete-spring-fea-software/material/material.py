class Material:
    """材料属性类，存储和管理材料的物理属性"""
    
    def __init__(self, e, nu, rho, sigma_y):
        """
        初始化材料属性
        
        参数:
            e: 弹性模量 (Pa)
            nu: 泊松比
            rho: 密度 (kg/m³)
            sigma_y: 屈服强度 (Pa)
        """
        self.e = e          # 弹性模量
        self.nu = nu        # 泊松比
        self.rho = rho      # 密度
        self.sigma_y = sigma_y  # 屈服强度
    
    def get_elastic_matrix(self):
        """
        计算并返回弹性矩阵 (3D)
        
        返回:
            弹性矩阵
        """
        c = self.e / ((1 + self.nu) * (1 - 2 * self.nu))
        return c * np.array([
            [1 - self.nu, self.nu, self.nu, 0, 0, 0],
            [self.nu, 1 - self.nu, self.nu, 0, 0, 0],
            [self.nu, self.nu, 1 - self.nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * self.nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * self.nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * self.nu) / 2]
        ])

# 导入numpy用于矩阵运算
import numpy as np
