import numpy as np
from glob import glob

input_dim = 640

def compute_iou(box, anchors):
    ious = []
    for anchor in anchors:
        w_min = np.min([box[0], anchor[0]])
        h_min = np.min([box[1], anchor[1]])
        intersection = w_min * h_min
        union = box[0] * box[1] + anchor[0] * anchor[1] - intersection
        iou = intersection / (union - intersection)
        ious.append(iou)
    return ious

def kmeans(boxes, k, dist=np.median):
    box_num = len(boxes)
    nearest_id = np.zeros(box_num)
    np.random.seed(42)
    clusters = boxes[np.random.choice([i for i in range(box_num)], k, replace=False)]
    while True:
        distance = []
        for i in range(box_num):
            ious = compute_iou(boxes[i], clusters)
            dis = [1-iou for iou in ious]
            distance.append(dis)
        distance = np.array(distance)
        new_nearest_id = np.argmin(distance, axis=1)
        if (new_nearest_id == nearest_id).all():
            break
        nearest_id = new_nearest_id
        for j in range(k):
            clusters[j] = dist(boxes[new_nearest_id == j], axis=0)
    return clusters

def load_dataset(path):
    path = path + '/*.txt'
    txt_list = glob(path)
    data_set = []
    for txt in txt_list:
        with open(txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            coordinate = line.split(' ')
            w, h = np.array(coordinate[3:5], dtype=np.float64)
            data_set.append([w, h])
    return np.array(data_set)

def main():
    txt_path = '/home/yhshih/yolov7/GC10/train/labels'
    data = load_dataset(txt_path)
    clusters = kmeans(data, 10)
    print('cluster center:*****************')
    print(clusters * input_dim)
    accuracy = np.mean([np.max(compute_iou(box, clusters)) for box in data])*100
    print('Accuracy(Average iou): %.4f %%' % accuracy)
    anchor_ratio = np.around(clusters[:, 0] / clusters[:, 1], decimals=2)
    anchor_ratio = list(anchor_ratio)
    print('Final anchor ratio: ', anchor_ratio)
    print('Sorted anchor ratio: ', sorted(anchor_ratio))

if __name__ == "__main__":
    main()
