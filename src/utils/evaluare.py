import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


def compute_average_precision(rec, prec):
    # functie adaptata din 2010 Pascal VOC development kit
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    for i in range(len(m_pre) -  1, -1, 1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    m_rec = np.array(m_rec)
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return average_precision


def eval_detections(detections, scores, file_names, ground_truth_path):
    ground_truth_file = np.loadtxt(ground_truth_path, dtype='str')
    ground_truth_file_names = np.array(ground_truth_file[:, 0])
    ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int32)

    num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
    gt_exists_detection = np.zeros(num_gt_detections)
    # sorteazam detectiile dupa scorul lor
    sorted_indices = np.argsort(scores)[::-1]
    file_names = file_names[sorted_indices]
    scores = scores[sorted_indices]
    detections = detections[sorted_indices]

    num_detections = len(detections)
    true_positive = np.zeros(num_detections)
    false_positive = np.zeros(num_detections)
    duplicated_detections = np.zeros(num_detections)

    for detection_idx in range(num_detections):
        indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

        gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
        bbox = detections[detection_idx]
        max_overlap = -1
        index_max_overlap_bbox = -1
        for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
            overlap = intersection_over_union(bbox, gt_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
                index_max_overlap_bbox = indices_detections_on_image[gt_idx]

        # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
        if max_overlap >= 0.3:
            if gt_exists_detection[index_max_overlap_bbox] == 0:
                true_positive[detection_idx] = 1
                gt_exists_detection[index_max_overlap_bbox] = 1
            else:
                false_positive[detection_idx] = 1
                duplicated_detections[detection_idx] = 1
        else:
            false_positive[detection_idx] = 1

    cum_false_positive = np.cumsum(false_positive)
    cum_true_positive = np.cumsum(true_positive)

    rec = cum_true_positive / num_gt_detections
    prec = cum_true_positive / (cum_true_positive + cum_false_positive)
    average_precision = compute_average_precision(rec, prec)
    plt.plot(rec, prec, '-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('All faces: average precision %.3f' % average_precision)
    plt.savefig('precizie_medie_all_faces.png')
    plt.show()

def eval_detections_character(detections, scores, file_names,ground_truth_path,character):
    ground_truth_file = np.loadtxt(ground_truth_path, dtype='str')
    ground_truth_file_names = np.array(ground_truth_file[:, 0])
    ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int32)

    num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
    gt_exists_detection = np.zeros(num_gt_detections)
    # sorteazam detectiile dupa scorul lor
    sorted_indices = np.argsort(scores)[::-1]
    file_names = file_names[sorted_indices]
    scores = scores[sorted_indices]
    detections = detections[sorted_indices]

    num_detections = len(detections)
    true_positive = np.zeros(num_detections)
    false_positive = np.zeros(num_detections)
    duplicated_detections = np.zeros(num_detections)

    for detection_idx in range(num_detections):
        indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

        gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
        bbox = detections[detection_idx]
        max_overlap = -1
        index_max_overlap_bbox = -1
        for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
            overlap = intersection_over_union(bbox, gt_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
                index_max_overlap_bbox = indices_detections_on_image[gt_idx]

        # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
        if max_overlap >= 0.3:
            if gt_exists_detection[index_max_overlap_bbox] == 0:
                true_positive[detection_idx] = 1
                gt_exists_detection[index_max_overlap_bbox] = 1
            else:
                false_positive[detection_idx] = 1
                duplicated_detections[detection_idx] = 1
        else:
            false_positive[detection_idx] = 1

    cum_false_positive = np.cumsum(false_positive)
    cum_true_positive = np.cumsum(true_positive)

    rec = cum_true_positive / num_gt_detections
    prec = cum_true_positive / (cum_true_positive + cum_false_positive)
    average_precision = compute_average_precision(rec, prec)
    plt.plot(rec, prec, '-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(character + ' faces: average precision %.3f' % average_precision)
    plt.savefig('precizie_medie_' + character + '.png')
    plt.show()

def evaluate_results_task1(solution_path,ground_truth_path,verbose = 0):

	#incarca detectiile + scorurile + numele de imagini	
	detections = np.load(solution_path + "detections_all_faces.npy",allow_pickle=True,fix_imports=True,encoding='latin1')
	print(detections.shape)

	scores = np.load(solution_path + "scores_all_faces.npy",allow_pickle=True,fix_imports=True,encoding='latin1')
	print(scores.shape)
	
	file_names = np.load(solution_path + "file_names_all_faces.npy",allow_pickle=True,fix_imports=True,encoding='latin1')
	print(file_names.shape)

	eval_detections(detections, scores, file_names, ground_truth_path)

def evaluate_results_task2(solution_path,ground_truth_path,character, verbose = 0):

	#incarca detectiile + scorurile + numele de imagini	
	detections = np.load(solution_path + "detections_" + character + ".npy",allow_pickle=True,fix_imports=True,encoding='latin1')
	print(detections.shape)

	scores = np.load(solution_path + "scores_"+ character + ".npy",allow_pickle=True,fix_imports=True,encoding='latin1')
	print(scores.shape)
	
	file_names = np.load(solution_path + "file_names_"+ character + ".npy",allow_pickle=True,fix_imports=True,encoding='latin1')
	print(file_names.shape)

	eval_detections_character(detections, scores, file_names, ground_truth_path, character)
  
verbose = 0
class_name = "deedee"
my_path = f"../data/predictions/{class_name}_predictions.txt"
boxes = []
scores = []
file_names = []

with open(my_path, "r") as f:
    for line in f:
        image_name, x_min, y_min, x_max, y_max, confidence = line.strip().split()
        boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        scores.append(float(confidence))
        file_names.append(image_name)

boxes = np.array(boxes)
scores = np.array(scores)
file_names = np.array(file_names)

output_path = f"../data/solutions/{class_name}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

np.save(output_path + "detections_all_faces.npy", boxes)
np.save(output_path + "scores_all_faces.npy", scores)
np.save(output_path + "file_names_all_faces.npy", file_names)

#change this on your machine
solution_path_root = output_path
ground_truth_path_root = f"../data/solutions/ground_truth/"

gt_path = f"../data/train/{class_name}_annotations.txt"
with open(gt_path, "r") as f:
    gt = f.readlines()
    gt = gt[:len(file_names)]

print(f"Evaluating {len(file_names)} faces")

gt_file_name = f"{class_name}_gt_validare.txt"
with open(ground_truth_path_root + gt_file_name, "w") as f:
    for line in gt:
        image_name, x_min, y_min, x_max, y_max, _ = line.strip().split()
        f.write(f"{image_name} {x_min} {y_min} {x_max} {y_max}\n")

#task1
solution_path = solution_path_root
# ground_truth_path = ground_truth_path_root + "task1_gt_validare.txt"
ground_truth_path = ground_truth_path_root + gt_file_name
evaluate_results_task1(solution_path, ground_truth_path, verbose)


# #task2
# solution_path = solution_path_root + "task2/"


# ground_truth_path = ground_truth_path_root + "task2_dad_gt_validare20.txt"
# evaluate_results_task2(solution_path, ground_truth_path, "dad", verbose)

# ground_truth_path = ground_truth_path_root + "task2_mom_gt_validare20.txt"
# evaluate_results_task2(solution_path, ground_truth_path, "mom", verbose)

# ground_truth_path = ground_truth_path_root + "task2_dexter_gt_validare20.txt"
# evaluate_results_task2(solution_path, ground_truth_path, "dexter", verbose)

# ground_truth_path = ground_truth_path_root + "task2_deedee_gt_validare20.txt"
# evaluate_results_task2(solution_path, ground_truth_path, "deedee", verbose)
