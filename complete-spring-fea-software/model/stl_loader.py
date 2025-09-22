import numpy as np
from stl import mesh

class STLLoader:
    
    def __init__(self):
        self.vertices = None
        self.triangles = None
    
    def load(self, file_path):

        try:
            stl_mesh = mesh.Mesh.from_file(file_path)
            self.vertices = stl_mesh.vectors.reshape(-1, 3)
            # 去除重复顶点
            self.vertices, unique_indices = np.unique(
                self.vertices, axis=0, return_inverse=True
            )
            
            # 获取三角形索引
            self.triangles = unique_indices.reshape(-1, 3)
            
            return {
                'vertices': self.vertices,
                'triangles': self.triangles
            }
        except Exception as e:
            raise RuntimeError(f"无法加载STL文件: {str(e)}")
        """
        这个代码用来夹在stl文件,输入的参数为file_path: STL文件路径
        返回包含顶点和三角形数据的字典
        """