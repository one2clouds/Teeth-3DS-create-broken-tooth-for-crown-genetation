from utils.jaw_classification import JawClassification
from utils.segmentation import Segmentation
from utils.bounding_box import get_bounding_boxes

def map_upper_labels(labels: list):
    ret = []
    for data in labels:
        if data == 0:
            ret.append(0)
        else:
            ret.append(data+10 if (data in range(1,9)) else data+12)
    return(ret)

def map_lower_labels(labels: list):
    ret = []
    for data in labels:
        if data == 0:
            ret.append(0)
        else:
            ret.append(data+30 if (data in range(1,9)) else data+32)
    return(ret)


def get_response(mesh):
    upper_checkpoint : str = "./model_checkpoints/upper_jaw.pth"
    lower_checkpoint : str = "./model_checkpoints/lower_jaw.pth"

    segment_upper = Segmentation(upper_checkpoint)
    segment_lower = Segmentation(lower_checkpoint)

    jaw_classification = JawClassification()
    jaw = "lower" if jaw_classification.inference(mesh) else "upper"

    labels = map_lower_labels(segment_lower.inference(mesh)) if (jaw == "lower") else map_upper_labels(segment_upper.inference(mesh))
    bounding_boxes = get_bounding_boxes(mesh, labels)
    color_map = {
            0 :[228,228,228],
            11:[255,153,153], 12:[255,230,153], 13:[204,255,153], 14:[153,255,179], 15:[153,255,255], 16:[153,179,255], 17:[204,153,255], 18:[255,153,230],
            21:[255,102,102], 22:[255,217,102], 23:[179,255,102], 24:[102,255,140], 25:[102,255,255], 26:[102,140,255], 27:[179,102,255], 28:[255,102,217],
            31: [255,153,153], 32: [255,230,153], 33: [204,255,153], 34: [153,255,179], 35: [153,255,255], 36: [153,179,255], 37: [204,153,255], 38: [255,153,230],
            41: [255,102,102], 42: [255,217,102], 43: [179,255,102], 44: [102,255,140], 45: [102,255,255], 46: [102,140,255], 47: [179,102,255], 48: [255,102,217]
        }
    response = {
        "jaw": jaw,
        "labels": labels,
        "bounding_boxes" : bounding_boxes,
        "color_map": color_map
    }

    return response