import torch
import open3d as o3d
import numpy as np

from model.dgcnn_cls import DGCNN_cls

class JawClassification():
    def __init__(self):
        self.checkpoint_path = "/home/shirshak/PoinTr/zzz_data_preparation/src/model_checkpoints/jaw_classification.pth"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DGCNN_cls(output_channels=1, input_dims=3, k=20, emb_dims=1024, dropout=0.5)
        self.model.to(self.device)

        self.checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])

        self.model.eval()

    def preprocess_obj_to_point_cloud(self, mesh, num_points = 2048):
        # Preprocess the point cloud
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

        # Normalize PCD
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        points -= centroid  # Centering
        max_distance = np.max(np.linalg.norm(points, axis=1))
        points /= max_distance  # Scaling

        # Convert to tensor and transpose to shape (3, 2048)
        points = torch.tensor(points, dtype=torch.float32).transpose(0, 1)  # Shape to (3, 2048)

        return points
    
    def inference(self, mesh):
        point_cloud = self.preprocess_obj_to_point_cloud(mesh)

        with torch.no_grad():
            inputs = point_cloud.to(self.device)

            outputs = self.model(inputs.unsqueeze(0))
            probabilities = torch.sigmoid(outputs).cpu().data.numpy()
            prediction = probabilities > 0.5

        return prediction