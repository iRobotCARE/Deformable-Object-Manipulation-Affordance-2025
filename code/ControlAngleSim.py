"""使用Sofa环境,控制变形到期望角度
created at 2025-03-10 by hsy
"""
import Sofa
import SofaRuntime
import Sofa.Gui
import Sofa.SofaGL
import os, sys
import numpy as np
import numpy.typing as npt
from scipy import sparse
import copy
import meshio
import taichi as ti
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)                                            # 改变当前工作目录到脚本文件所在目录

root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from DiffPD2d import SoftObject2D, compress_vectors
from Utilize.GenMsh import read_mshv2_triangle, mesh_obj_tri, write_msh2_tri


def createScene(root, contact:list):
    root.addObject('RequiredPlugin', pluginName=['Sofa.Component',
                                                 'Sofa.Component.Collision',
                                                 'Sofa.Component.Constraint.Projective',
                                                 'Sofa.Component.IO.Mesh',
                                                 'Sofa.Component.LinearSolver',
                                                 'Sofa.GL.Component.Rendering3D'])
    
    root.dt = 0.01
    root.bbox = [[-0.1, -0.1, 0.], [0.2, 0.2, 0.1]]
    root.gravity = [0., 0., 0.]
    root.addObject('VisualStyle', displayFlags='showBehaviorModels showVisual showForceFields showInteractionForceFields showWireframe')

    root.addObject('DefaultAnimationLoop', )
    root.addObject('CollisionPipeline', depth="6", verbose="0", draw="0")
    root.addObject('BruteForceBroadPhase', )
    root.addObject('BVHNarrowPhase', )
    root.addObject('NewProximityIntersection', name="Proximity", alarmDistance="0.5", contactDistance="0.2")
    root.addObject('CollisionResponse', name="Response", response="PenalityContactForceField")

    obj = root.addChild('object')
    # Rayleigh阻尼影响了软体振动
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    obj.addObject('MeshGmshLoader', name='loader', filename='Mesh/shape_split.msh', scale='1', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TriangleSetTopologyModifier', name='modifier')
    obj.addObject('TriangleSetGeometryAlgorithms', name='geomalgo')#, tempate='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.1')#, massDensity='0.1')

    X_EPS = 5.e-3
    obj.addObject('BoxROI', name='box', box=f"-0.1 {-X_EPS} -0.1 0.11 {X_EPS} 0.1")
    obj_fixed = obj.addObject('FixedConstraint', name='fixed', indices='@box.indices')

    obj.addObject('MeshSpringForceField', name="springs", trianglesStiffness=90, trianglesDamping=0.3)
    obj.addObject('TriangleCollisionModel')

    obj_move_list = []
    for q_i in contact:
        obj_move_list.append(obj.addObject('LinearMovementConstraint', name='cnt'+str(q_i), template="Vec3", indices=[q_i]))

    return obj, obj_move_list


def add_move(handle_list:list, dt:float, movement:npt.NDArray):
    """ Use `LinearMovemetConstraint` to add a simulation step-wise movement
    Args:
        handle: The node of the object
        dt: The time step
        movement: The additional movement
    """
    if movement.shape[1] == 2:
        movement = np.concatenate((movement, np.zeros((movement.shape[0], 1))), axis=1)
    for i, handle in enumerate(handle_list):
        times_array = handle.findData('keyTimes').value
        movements_array = handle.findData('movements').value

        last_time = times_array[-1]
        last_movement = movements_array[-1, :]

        handle.findData('keyTimes').value = np.append(times_array, last_time + dt)
        handle.findData('movements').value = np.append(movements_array, [movement[i,:] + last_movement], axis=0)


def get_marker_pos(handle, marker_idx:list)->npt.NDArray:
    """从sofa中获取指定节点的位置
    """
    marker_pos = np.zeros((len(marker_idx), 3))
    # node_pos = handle.findData('position').value
    for i, idx in enumerate(marker_idx):
        pos_tmp = copy.deepcopy(handle.findData('position').value[idx])
        marker_pos[i] = pos_tmp
    return marker_pos


def save_vtu(mesh_file:str, pos:npt.NDArray, write_name:str):
    """Save the node position to a .vtu file

    Args:
        mesh_file (str): The initial mesh file name
        pos (npt.NDArray): The node position
        write_name (istrnt): The write file name
    """
    _, triangles = read_mshv2_triangle(mesh_file)

    cells_write = [("triangle", triangles)]
    mesh = meshio.Mesh(points=pos, cells=cells_write)
    mesh.write(f"data/{write_name}")


class MyObject(SoftObject2D):
    def __init__(self, shape, fix, contact, dots_list, E, nu, dt, density, **kwargs):
        super().__init__(shape, fix, contact, E, nu, dt, density, **kwargs)
        self.loss = 0.
        self.dots_idx = dots_list
        dots_num = len(dots_list)
        self.dot_pos = ti.Vector.field(2, dtype=ti.f64, shape=dots_num)
        self.dot_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=dots_num)

        print(f"Marker index: {self.dots_idx}")

        self.construct_dot_pos()

    
    def construct_dot_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos_init[i] = self.node_pos_init[idx]
    

    def update_dot_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos[i] = self.node_pos[idx]

        
    def construct_L_sofa(self, dot_sofa:npt.NDArray, angle_desired:float):
        """将指定角度变形到期望角度
        """
        self.dL_dq_contact.fill(0.)
        if len(self.dots_idx) != 3:
            raise ValueError("The number of marker is not 3")
        else:
            idx1, idx2, idx3 = self.dots_idx
        pos1, pos2, pos3 = dot_sofa[:,:2]         # sofa中是三维坐标

        v1, v2 = pos2 - pos1, pos3 - pos1
        angle_tmp = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        loss = (angle_tmp - angle_desired) ** 2
        dloss = 2 * (angle_tmp - angle_desired)

        A, B, C = v1.dot(v2), np.linalg.norm(v1), np.linalg.norm(v2)
        dv1 = v2 / (B*C) - v1 * A / (B**3 * C)
        dv2 = v1 / (B*C) - v2 * A / (B * C**3)
        dv1 *= dloss
        dv2 *= dloss

        dpos2 = dv1 @ np.eye(2)
        dpos3 = dv2 @ np.eye(2)
        dpos1 = dv1 @ (-np.eye(2)) + dv2 @ (-np.eye(2))

        self.dL_dq_contact[idx1*2]     = dpos1[0]
        self.dL_dq_contact[idx1*2 + 1] = dpos1[1]
        self.dL_dq_contact[idx2*2]     = dpos2[0]
        self.dL_dq_contact[idx2*2 + 1] = dpos2[1]
        self.dL_dq_contact[idx3*2]     = dpos3[0]
        self.dL_dq_contact[idx3*2 + 1] = dpos3[1]

        # Vector norm constraint
        v1_norm_init = np.linalg.norm(self.dot_pos_init[1] - self.dot_pos_init[0])
        v2_norm_init = np.linalg.norm(self.dot_pos_init[2] - self.dot_pos_init[0])

        loss_norm = (B - v1_norm_init) ** 2 + (C - v2_norm_init) ** 2
        print(f"Distance constraint: {loss_norm}")

        dv1norm = 2 * (B - v1_norm_init) * v1 / B
        dv2norm = 2 * (C - v2_norm_init) * v2 / C
        dv1norm *= 1.e3                 # 用于增加Distance constraint的权重, 需要根据情况调整
        dv2norm *= 1.e3                 
        
        dpos2norm = dv1norm @ np.eye(2)
        dpos3norm = dv2norm @ np.eye(2)
        dpos1norm = dv1norm @ (-np.eye(2)) + dv2norm @ (-np.eye(2))

        self.dL_dq_contact[idx1*2]     += dpos1norm[0]
        self.dL_dq_contact[idx1*2 + 1] += dpos1norm[1]
        self.dL_dq_contact[idx2*2]     += dpos2norm[0]
        self.dL_dq_contact[idx2*2 + 1] += dpos2norm[1]
        self.dL_dq_contact[idx3*2]     += dpos3norm[0]
        self.dL_dq_contact[idx3*2 + 1] += dpos3norm[1]

        return angle_tmp, loss + loss_norm
        

    def compute_dcontact(self, dot_sofa:npt.NDArray):
        """ \partial L / \partial y with contact action, 计算关于contact的导数, 用于控制
        """
        angle_tmp, loss_tmp = self.construct_L_sofa(dot_sofa, 0.78)
        self.construct_g_hessian()
        self.compute_z(10)

        print(f"Angle Cosine: {angle_tmp}; Loss: {loss_tmp}")

        z_np = self.z.to_numpy()
        self.dy_contact = np.multiply(z_np, self.dx_const.to_numpy())


@ti.data_oriented
class SofaObject:
    def __init__(self, points:npt.NDArray, triangles:npt.NDArray):
        self.points_num = points.shape[0]
        self.triangles_num = triangles.shape[0]

        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.points_num)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.points_num)
        self.node_pos.from_numpy(points[:,:2])
        self.node_pos_init.from_numpy(points[:,:2])

        self.triangles = ti.Vector.field(3, dtype=ti.i32, shape=self.triangles_num)
        self.Xg_inv = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.triangles_num)
        self.stretch_energy = ti.field(dtype=ti.f64, shape=self.triangles_num)
        self.triange_area = ti.field(dtype=ti.f64, shape=self.triangles_num)
        self.triangles.from_numpy(triangles)

        self.compute_area()
        self.construct_Xg_inv()

    
    @ti.kernel
    def compute_area(self):
        for f_i in range(self.triangles_num):
            ia, ib, ic = self.triangles[f_i]
            qa, qb, qc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            self.triange_area[f_i] = 0.5 * ti.abs(((qb - qa).cross(qc - qa)))


    @ti.kernel
    def construct_Xg_inv(self):
        for i in range(self.triangles_num):
            ia, ib, ic = self.triangles[i]
            a = ti.Vector([self.node_pos_init[ia].x, self.node_pos_init[ia].y])
            b = ti.Vector([self.node_pos_init[ib].x, self.node_pos_init[ib].y])
            c = ti.Vector([self.node_pos_init[ic].x, self.node_pos_init[ic].y])
            B_i_inv = ti.Matrix.cols([b - a, c - a])
            self.Xg_inv[i] = B_i_inv.inverse()


    @ti.kernel
    def construct_rhs_stretch(self):
        for f_i in range(self.triangles_num):
            idx1, idx2, idx3 = self.triangles[f_i]
            a, b, c = self.node_pos[idx1], self.node_pos[idx2], self.node_pos[idx3]
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f_i], ti.f64)

            U, sig, V = ti.svd(F_i, ti.f64)
            self.stretch_energy[f_i] = 0.5 * self.triange_area[f_i] * ((sig[0,0]-1.)**2 + (sig[1,1]-1.)**2)


def main():
    shape = [0.1, 0.1]
    contact_sofa = [231, 252, 273, 427, 428, 429]
    contact_list = [66, 115]
    marker_sofa = [100, 254, 386]
    marker_list = [30, 67, 103]               # 角度的顶点是第一个
    # marker_sofa = [100, 380, 394]           # sofa中marker的序号
    # marker_list = [30, 100, 107]            # markder在模型中的序号
    fix = range(11)
    gain = 2.e-1

    node_np, _, ele_np = mesh_obj_tri(shape, 0.01/2)
    msh_file:str = "Mesh/shape_split.msh"                   # sofa使用更细分的网格模型
    write_msh2_tri(msh_file, node_np, ele_np)

    root = Sofa.Core.Node('root')
    _, move_handle = createScene(root, contact_sofa)
    Sofa.Simulation.init(root)

    soft_sofa = SofaObject(node_np, ele_np)                 # 计算deformation

    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')

    params = {"E": 1.e4, "nu": 0.4, "dt": 0.01, "density": 10.e2}
    soft = MyObject(shape, fix, contact_list, marker_list, **params)
    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    for step in range(80):
        print(f"Time Step: {step} ======================================")
        dots_pos_sofa_new = get_marker_pos(dofs, marker_sofa)
        print(f'Detected marker position: {dots_pos_sofa_new.flatten()}')

        sofa_pos_tmp = dofs.findData('position').value
        soft_sofa.node_pos.from_numpy(sofa_pos_tmp[:,:2])
        save_vtu('Mesh/shape_split.msh', sofa_pos_tmp, f'shape_{step:04d}.vtu')

        soft.substep(step)
        soft.compute_dcontact(dots_pos_sofa_new[:,:2])
        soft.update_dot_pos()
        print(f"Model marker position: {soft.dot_pos.to_numpy().flatten()}")

        dy_dcontact = soft.dy_contact.reshape(-1, 2)        # reshape到与接触点个数相同
        end_speed = -gain * dy_dcontact[soft.contact_particle_list]
        end_speed_compress = compress_vectors(end_speed, 0.02)
        soft.contact_vel.from_numpy(end_speed_compress)

        print(f"End speed: {end_speed.flatten()}")

        add_move(move_handle, dt, np.repeat(end_speed_compress * dt, 3, axis=0))
        Sofa.Simulation.animate(root, dt)



if __name__ == "__main__":
    main()