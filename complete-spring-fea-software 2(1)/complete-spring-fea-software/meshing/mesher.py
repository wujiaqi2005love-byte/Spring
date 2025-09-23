import gmsh
import numpy as np

class Mesher:
    """网格划分工具类，用于生成有限元网格"""
    
    def __init__(self):
        """初始化网格划分器"""
        # 初始化Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        self.model_name = "spring_mesh"
        gmsh.model.add(self.model_name)
        
    def generate_mesh(self, stl_file, element_type='tetrahedron', mesh_size=0.01):
        try:
            # 判断文件类型
            if stl_file.lower().endswith(('.step', '.stp')):
                # STEP文件导入
                gmsh.model.occ.importShapes(stl_file)
                gmsh.model.occ.synchronize()
                # 获取所有体积标签
                volumes = gmsh.model.getEntities(dim=3)
                volume_tags = [v[1] for v in volumes]
                if volume_tags:
                    gmsh.model.addPhysicalGroup(3, volume_tags)
                # 也可添加表面物理组
                surfaces = gmsh.model.getEntities(dim=2)
                surface_tags = [s[1] for s in surfaces]
                if surface_tags:
                    gmsh.model.addPhysicalGroup(2, surface_tags)
            else:
                # STL文件导入（仅表面网格）
                gmsh.merge(stl_file)
                gmsh.model.geo.synchronize()
                surfaces = gmsh.model.getEntities(dim=2)
                surface_tags = [s[1] for s in surfaces]
                if surface_tags:
                    gmsh.model.addPhysicalGroup(2, surface_tags)
            # 设置网格大小
            gmsh.option.setNumber('Mesh.CharacteristicLengthMax', mesh_size)
            # 生成网格
            dim = 3 if stl_file.lower().endswith(('.step', '.stp')) or element_type == 'tetrahedron' else 2
            gmsh.model.mesh.generate(dim)
            # 获取节点
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes = np.reshape(node_coords, (int(len(node_coords)/3), 3))
            # 获取单元
            element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
            elements = []
            # 筛选出我们需要的单元类型
            for i in range(len(element_types)):
                if (dim == 2 and element_types[i] == 3):
                    element_nodes = np.reshape(element_node_tags[i], (-1, 3))
                    elements = element_nodes - 1
                    break
                elif (dim == 3 and element_types[i] == 4):
                    element_nodes = np.reshape(element_node_tags[i], (-1, 4))
                    elements = element_nodes - 1
                    break
            nodes = np.reshape(node_coords, (int(len(node_coords)/3), 3))
            
            # 获取单元
            element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
            elements = []
            
            # 筛选出我们需要的单元类型
            for i in range(len(element_types)):
                # 三角形单元类型为3，四面体为4（gmsh官方编号）
                if (element_type == 'triangle' and element_types[i] == 3):
                    element_nodes = np.reshape(element_node_tags[i], (-1, 3))
                    elements = element_nodes - 1  # Gmsh使用1基索引
                    break
                elif (element_type == 'tetrahedron' and element_types[i] == 4):
                    element_nodes = np.reshape(element_node_tags[i], (-1, 4))
                    elements = element_nodes - 1
                    break
            
            # 保存网格（可选）
            gmsh.write("spring_mesh.msh")
            
            return {
                'nodes': nodes,
                'elements': elements,
                'type': element_type
            }
            
        except Exception as e:
            raise RuntimeError(f"网格生成失败: {str(e)}")
        finally:
            # 清理Gmsh
            gmsh.finalize()
