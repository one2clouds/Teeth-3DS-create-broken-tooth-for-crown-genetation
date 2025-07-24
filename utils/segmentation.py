import torch
import numpy as np
import open3d as o3d

from model.dgcnn_semseg import DGCNN_semseg
from utils.reverse_decimation import reverse_decimation_mapping

def simplify_mesh(mesh , target_number=16000):
    return mesh.simplify_quadric_decimation(target_number)


def normalize(points, axis_independent=False):
    means = points.mean(axis=0)
    stds = points.std(axis=0)

    std = np.max(stds)

    for i in range(3):
        if axis_independent is True:
            points[:, i] = (points[:, i] - means[i]) / stds[i] #point 1
        else:
            points[:, i] = (points[:, i] - means[i]) / std

    return points

def calc_feature_vector(mesh, normalize_axis_independent=False, normalize_vertex_normals = False):
    # 頂点座標の取得
    vertices = np.asarray(mesh.vertices).astype(dtype='float32')
    #print('vertices.shape = ', vertices.shape)

    # 座標の標準化
    vertices = normalize(vertices, normalize_axis_independent)


    # 頂点法線の取得
    if not mesh.has_vertex_normals():
        # 法線情報がない場合は計算する
        mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals).astype(dtype='float32')


    if normalize_vertex_normals:
        # 法線ベクトルの標準化
        vertex_normals = normalize(vertex_normals, normalize_axis_independent)

    # メッシュ法線の取得
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals).astype(dtype='float32')

    # 頂点の色情報の取得
    count_colors = 0
    vertex_colors = np.asarray(np.round(np.asarray(mesh.vertex_colors) * 255.0)).astype(dtype='int16')
    for i in range(len(vertex_colors)):
        v_color = vertex_colors[i]
        if count_colors == 0:
            print(v_color)
        count_colors += 1
    #print('vertex_colors.shape = ', vertex_colors.shape)

    # 頂点座標と法線の表示
    if False:
        for i in range(len(vertices)):
            vertex = vertices[i]
            normal = vertex_normals[i]


    # 三角形単位でループを行い、頂点座標と法線を表示
    triangles = mesh.triangles
    ids = np.asarray(triangles)
    print('triangles.shape = ', ids.shape)
    ids = ids.reshape(-1)
    print(ids.shape)

    triangle_vertices = vertices[ids].reshape(len(triangles), 9).astype(dtype='float32')
    triangle_vertex_normals = vertex_normals[ids].reshape(len(triangles), 9).astype(dtype='float32')


    # メッシュの重心座標
    triangle_centers = np.zeros((len(triangles), 3)).astype(dtype='float32')


    for i in range(len(triangles)):
        triangle = triangles[i]
        vertex_indices = triangle.tolist()
        mesh_vertices = vertices[vertex_indices]
        # mesh_normals = vertex_normals[vertex_indices]
        # mesh_colors = vertex_colors[vertex_indices]

        # メッシュの3頂点の座標および法線
        px = np.zeros((3))
        for j in range(3):
            vertex = mesh_vertices[j]
            px += vertex
        px /= 3.0
        triangle_centers[i, :] = px


    X = np.hstack([
        triangle_centers,
        triangle_vertices,
        triangle_normals,
        triangle_vertex_normals
    ])

    return X

class Segmentation:
    def __init__(self, checkpoint: str):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.num_classes = 17
        self.input_dims = 24

        self.num_points = 15984

        self.model = DGCNN_semseg(num_classes=self.num_classes, input_dims=self.input_dims)
        self.model = self.model.to(self.device)

        self.model.load_state_dict(torch.load(checkpoint)["state_dict"])
        self.model.eval()

    def inference(self, mesh):
        simplified_mesh = simplify_mesh(mesh)

        X =  calc_feature_vector(simplified_mesh)

        data = torch.from_numpy(np.asarray(X)).to(self.device).unsqueeze(0)
        data = data.permute(0, 2, 1)

        seg_pred = self.model(data)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        pred_np = pred.detach().cpu().numpy().tolist()[0]

        orig_labels = reverse_decimation_mapping(mesh, simplified_mesh, pred_np)

        torch.cuda.empty_cache()

        return orig_labels


