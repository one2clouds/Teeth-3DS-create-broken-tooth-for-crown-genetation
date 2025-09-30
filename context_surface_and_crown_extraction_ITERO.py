import json 
import numpy as np 
import open3d as o3d
from extract_context_teeth import extract_teeth_group_and_their_gums, mesh_to_pcd
import os 
from glob import glob
import re

abut_context_teeth_dict_ = {
    "abutment_teeth":["Left_Neighbor", "Abutment Teeth" ,"Right_Neighbor", "Left_Antagonist", "Middle_Antagonist", "Right_Antagonist"],
    32: [[31,41], [32], [33,34], [21,11], [22], [23,24]], 
    35: [[34,33], [35], [36,37], [24,23], [25], [26,27]],
    36: [[35,34], [36], [37,38], [25,24], [26], [27,28]],
    37: [[36,35], [37], [38], [26,25], [27], [28]],

    44: [[45,46], [44], [43,42], [15,16], [14], [13,12]],
    46: [[47,48], [46], [45,44], [17, 18], [16], [15, 14]],
    47: [[48], [47], [46,45], [18], [17], [16,15]],
}

def extract_highest_priority(tooth_number, available_teeth, abut_dict):
    result = []
    for group in abut_dict[tooth_number]:
        selected = None
        for candidate in group:
            if candidate in available_teeth:
                selected = candidate
                break  # stop at the first match (highest priority)
        if selected is not None:
            result.append(selected)
    return result

def extract_segments_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    segments = []
    for cell in data['cells']:
        segments.append(cell['fdi'])
    segments = np.array(segments)
    try:
        abutment_teeth = data["abutment_teeth"]
    except KeyError:
        abutment_teeth = None
        print(f"Warning: No 'abutment_teeth' key in {json_file_path}")

    return segments, abutment_teeth



if __name__=="__main__":
    ply_names = sorted(glob("../CROWN_GEN_DATASET/CROWN-ITERO-ANNOTATED-TOOTH/ply/*.ply"))
    json_names = sorted(glob("../CROWN_GEN_DATASET/CROWN-ITERO-ANNOTATED-TOOTH/json/*.json"))
    
    print(len(ply_names))
    elements = []
    for ply_name, json_name in zip(ply_names, json_names):
        # print(ply_name)
        # print(json_name)
        segment, abutment_teeth = extract_segments_from_json(json_name)
        if abutment_teeth:
            # print(json_name)
            # print(abutment_teeth)
            # print("-"*80)
            # elements.append(abutment_teeth[0])


            print(abut_context_teeth_dict_[abutment_teeth[0]])

            label = "master"
            mesh = o3d.io.read_triangle_mesh(ply_name)

            available_teeth = np.unique(segment) # A SET OF AVAILABLE LABELS WHICH WE HAVE TO CHOOSE FROM 

            target_segment_ids = extract_highest_priority(abutment_teeth[0], available_teeth, abut_context_teeth_dict_)


            three_teeth_mesh, _, three_teeth_segments = extract_teeth_group_and_their_gums(mesh, segment, target_segment_ids)
            pcd_mesh, npy_mesh = mesh_to_pcd(three_teeth_mesh, num_points=4096)

            unique_customer_id = re.search(r"CROWN-ITERO-(.+?)_(?:Antagonist|Preparation)Scan", os.path.basename(ply_name)).group(1)

            os.makedirs(f"outputs-CROWN-ITERO-ANNOTATED/{unique_customer_id}", exist_ok=True)
            print(os.path.join(f"outputs-CROWN-ITERO-ANNOTATED/{unique_customer_id}", f"{unique_customer_id}_{label}.pcd"))
            o3d.io.write_point_cloud(os.path.join(f"outputs-CROWN-ITERO-ANNOTATED/{unique_customer_id}", f"{unique_customer_id}_{label}.pcd"), pcd_mesh)

            base_ply_name = os.path.basename(ply_name)
            base_json_name = os.path.basename(json_name)


            if "Antagonist" in ply_name:
                new_ply_name = os.path.join(os.path.dirname(ply_name), base_ply_name.replace("Antagonist", "Preparation"))
                new_json_name = os.path.join(os.path.dirname(json_name), base_json_name.replace("Antagonist", "Preparation"))


            elif "Preparation" in ply_name:
                new_ply_name = os.path.join(os.path.dirname(ply_name), base_ply_name.replace("Preparation", "Antagonist"))
                new_json_name = os.path.join(os.path.dirname(json_name), base_json_name.replace("Preparation", "Antagonist"))


            # Name -> Antagonist just means the opposite jaw of the preparation tooth 
            segment, _ = extract_segments_from_json(new_json_name)
            label = "Antagonist"
            mesh = o3d.io.read_triangle_mesh(new_ply_name)


            available_teeth = np.unique(segment) # A SET OF AVAILABLE LABELS WHICH WE HAVE TO CHOOSE FROM 

            target_segment_ids = extract_highest_priority(abutment_teeth[0], available_teeth, abut_context_teeth_dict_)

            three_teeth_mesh, _, three_teeth_segments = extract_teeth_group_and_their_gums(mesh, segment, target_segment_ids)
            pcd_mesh, npy_mesh = mesh_to_pcd(three_teeth_mesh, num_points=4096)

            # unique_customer_id = re.search(r"CROWN-ITERO-(.+?)_(?:Antagonist|Preparation)Scan", os.path.basename(new_ply_name)).group(1)

            os.makedirs(f"outputs-CROWN-ITERO-ANNOTATED/{unique_customer_id}", exist_ok=True)
            print(os.path.join(f"outputs-CROWN-ITERO-ANNOTATED/{unique_customer_id}", f"{unique_customer_id}_{label}.pcd"))    
            o3d.io.write_point_cloud(os.path.join(f"outputs-CROWN-ITERO-ANNOTATED/{unique_customer_id}", f"{unique_customer_id}_{label}.pcd"), pcd_mesh)



    # EXTRACTION OF ITERO CROWN
    ply_dir_names = sorted(glob("./outputs-CROWN-ITERO-ANNOTATED/*/"))
    ply_crown_names = sorted(glob("../CROWN_GEN_DATASET/Crown-for-abutment-ITERO-Only/*/*"))

    for ply_crown_name in ply_crown_names:
        mesh = o3d.io.read_triangle_mesh(ply_crown_name)
        crown_pcd, crown_npy = mesh_to_pcd(mesh, num_points=1536)
        for ply_dir_name in ply_dir_names: 
            if ply_crown_name.split("/")[-2] == ply_dir_name.split('/')[-2]:
                print(os.path.join(ply_dir_name, ply_crown_name.split('/')[-2] + "_shell.pcd"))
                o3d.io.write_point_cloud(os.path.join(ply_dir_name, ply_crown_name.split('/')[-2] + "_shell.pcd"), crown_pcd)
                continue