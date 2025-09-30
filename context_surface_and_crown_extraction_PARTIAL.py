import json 
import numpy as np 
import open3d as o3d
from extract_context_teeth import extract_teeth_group_and_their_gums, mesh_to_pcd
import os 
from glob import glob
import re


# Why instead of just one neighbor, we try to keep instances for two neighbors, it is because for example we want to extract left neighbor of tooth 36(molar), it is 35(premolar) but in edge case some people wouldn't have two premolar, and only one premolar(34 or 35)
# In such case we take backup teeth 35 which is its nearest left neighbor 
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
    ply_names = sorted(glob("../CROWN_GEN_DATASET/CROWN-PARTIAL-ANNOTATED-TOOTH/ply/*.ply"))
    json_names = sorted(glob("../CROWN_GEN_DATASET/CROWN-PARTIAL-ANNOTATED-TOOTH/json/*.json"))

    elements = []
    for ply_name, json_name in zip(ply_names, json_names):
        # print(ply_name)
        # print(json_name)
        segment_only_abutment, abutment_teeth = extract_segments_from_json(json_name)
        if abutment_teeth:
            # print(json_name)
            # print(abutment_teeth)
            # print("-"*80)
            # elements.append(abutment_teeth[0])

            # print(abut_context_teeth_dict_[abutment_teeth[0]])

            label = "master"
            mesh = o3d.io.read_triangle_mesh(ply_name)

            available_teeth = np.unique(segment_only_abutment) # A SET OF AVAILABLE LABELS WHICH WE HAVE TO CHOOSE FROM 
            target_segment_ids = extract_highest_priority(abutment_teeth[0], available_teeth, abut_context_teeth_dict_)

            # print(available_teeth)
            # print(target_segment_ids)

            abutment_mesh, _, _ = extract_teeth_group_and_their_gums(mesh, segment_only_abutment, target_segment_ids)
            pcd_abutment_mesh, npy_abutment_mesh = mesh_to_pcd(abutment_mesh, num_points=1000)

            unique_customer_id = re.search(r"CROWN-PARTIAL-([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{5}-\d{3}_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{5}-\d{3})", os.path.basename(json_name)).group(1)
            
            if abutment_teeth[0] in [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28]: # UPPER JAW 
                new_MASTER_ply_name = os.path.join(os.path.dirname(ply_name), "CROWN-PARTIAL-" + unique_customer_id + "-upperjaw.ply")
                new_MASTER_json_name = os.path.join(os.path.dirname(json_name), "CROWN-PARTIAL-" + unique_customer_id + "-upperjaw.json")

                new_ANTAGONIST_ply_name = os.path.join(os.path.dirname(ply_name), "CROWN-PARTIAL-" + unique_customer_id + "-lowerjaw.ply")
                new_ANTAGONIST_json_name = os.path.join(os.path.dirname(json_name), "CROWN-PARTIAL-" + unique_customer_id + "-lowerjaw.json")

            
            elif abutment_teeth[0] in [31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]: #LOWER JAW 

                new_MASTER_ply_name = os.path.join(os.path.dirname(ply_name), "CROWN-PARTIAL-" + unique_customer_id + "-lowerjaw.ply")
                new_MASTER_json_name = os.path.join(os.path.dirname(json_name), "CROWN-PARTIAL-" + unique_customer_id + "-lowerjaw.json")

                new_ANTAGONIST_ply_name = os.path.join(os.path.dirname(ply_name), "CROWN-PARTIAL-" + unique_customer_id + "-upperjaw.ply")
                new_ANTAGONIST_json_name = os.path.join(os.path.dirname(json_name), "CROWN-PARTIAL-" + unique_customer_id + "-upperjaw.json")

            # print(new_MASTER_ply_name)
            # print(new_ANTAGONIST_ply_name)
            
            segment, _ = extract_segments_from_json(new_MASTER_json_name)
            mesh = o3d.io.read_triangle_mesh(new_MASTER_ply_name)

            available_teeth = np.unique(segment) # A SET OF AVAILABLE LABELS WHICH WE HAVE TO CHOOSE FROM 
            target_segment_ids = extract_highest_priority(abutment_teeth[0], available_teeth, abut_context_teeth_dict_)

            MASTER_mesh, _, _ = extract_teeth_group_and_their_gums(mesh, segment, target_segment_ids)
            pcd_master_mesh, npy_master_mesh = mesh_to_pcd(MASTER_mesh, num_points=3096)


            #COMBINE ABUTMENT TOOTH PCD MESH AND MASTER MESH
            combined_pcd_MASTER_mesh = pcd_abutment_mesh + pcd_master_mesh

            os.makedirs(f"outputs-CROWN-PARTIAL-ANNOTATED/{unique_customer_id}", exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(f"outputs-CROWN-PARTIAL-ANNOTATED/{unique_customer_id}", f"{unique_customer_id}_{label}.pcd"), combined_pcd_MASTER_mesh)
            print(os.path.join(f"outputs-CROWN-PARTIAL-ANNOTATED/{unique_customer_id}", f"{unique_customer_id}_{label}.pcd"))


            # FOR OPPOSITE JAW  
            segment, _ = extract_segments_from_json(new_ANTAGONIST_json_name)
            label = "Antagonist"
            mesh = o3d.io.read_triangle_mesh(new_ANTAGONIST_ply_name)

            available_teeth = np.unique(segment) # A SET OF AVAILABLE LABELS WHICH WE HAVE TO CHOOSE FROM 
            target_segment_ids = extract_highest_priority(abutment_teeth[0], available_teeth, abut_context_teeth_dict_)

            ANTAGONIST_mesh, _, _ = extract_teeth_group_and_their_gums(mesh, segment, target_segment_ids)
            pcd_antagonist_mesh, npy_antagonist_mesh = mesh_to_pcd(ANTAGONIST_mesh, num_points=4096)

            o3d.io.write_point_cloud(os.path.join(f"outputs-CROWN-PARTIAL-ANNOTATED/{unique_customer_id}", f"{unique_customer_id}_{label}.pcd"), pcd_antagonist_mesh)
            print(os.path.join(f"outputs-CROWN-PARTIAL-ANNOTATED/{unique_customer_id}", f"{unique_customer_id}_{label}.pcd"))
        else:
            continue


    # EXTRACTION OF PARTIAL CROWN

    ply_dir_names = sorted(glob("./outputs-CROWN-PARTIAL-ANNOTATED/*/"))
    ply_crown_names = sorted(glob("../CROWN_GEN_DATASET/Crown-for-abutment-PARTIAL-Only/*/*"))

    for ply_crown_name in ply_crown_names:
        mesh = o3d.io.read_triangle_mesh(ply_crown_name)
        crown_pcd, crown_npy = mesh_to_pcd(mesh, num_points=1536)
        for ply_dir_name in ply_dir_names: 
            # print(ply_crown_name.split("/")[-2]) 
            # print(ply_dir_name.split('/')[-2])
            if ply_crown_name.split("/")[-2] in ply_dir_name.split('/')[-2]:
                print(os.path.join(ply_dir_name, ply_crown_name.split('/')[-2] + "_shell.pcd"))
                o3d.io.write_point_cloud(os.path.join(ply_dir_name, ply_dir_name.split('/')[-2] + "_shell.pcd"), crown_pcd)
                continue