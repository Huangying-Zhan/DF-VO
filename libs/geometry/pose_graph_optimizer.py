import numpy as np
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        """Optimize the pose graph

        Args:
            max_iteration (int): maximum iteration
        """
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        """Add vertex

        Args:
            id (int): index
            pose (g2o.Isometry3d): T_wc
        """
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):
        """Add edge

        Args:
            vertices (list): two vertices / vertex index
            measurement (g2o.Isometry3d): relative pose T_ij
        """

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        """Get pose matrix at vertex-id

        Args:
            id (int): vertex index

        Returns:
            pose (array, [4x4]): pose of vertex
        """
        return self.vertex(id).estimate().matrix()
