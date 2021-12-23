import random
from math import sqrt
import numpy as np
from skimage.metrics import structural_similarity as ssim


# Calculate the Euclidean distance between two vectors
def euclidean_distance(image1, image2):
    distance = 0.0
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if isinstance(image1[i][j], complex):
                row1_r = image1[i][j].real
                row1_i = image1[i][j].imag
                row2_r = image2[i][j].real
                row2_i = image2[i][j].imag
                distance += (row1_r - row2_r) ** 2 + (row1_i - row2_i) ** 2
            else:
                distance += (image1 - image2) ** 2
    return sqrt(distance)


def mse(imageA, imageB):
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Locate the most similar neighbors
def get_neighbors(train, labels, test_image, num_neighbors, metric):
    distances = list()
    for i in range(train.shape[0]):
        train_image = train[i]
        if metric == 'euclidean':
            dist = euclidean_distance(test_image, train_image)
        elif metric == 'mse':
            dist = mse(test_image, train_image)
        elif metric == 'ssim':
            dist = 1-ssim(test_image, train_image)
        else:
            print("Error: metric not allowed.")
            return
        distances.append((labels[i], dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train, labels, test_row, metric, num_neighbors=5):
    neighbors = get_neighbors(train, labels, test_row, num_neighbors, metric)
    prediction = max(set(neighbors), key=neighbors.count)
    return prediction


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def k_nearest_neighbors(train, labels, test, metric, num_neighbors):
    predictions = list()
    for image in test:
        output = predict_classification(train, labels, image, 'euclidean', num_neighbors)
        predictions.append(output)
    return predictions


def knn_traintestsplit(dataset, labels, test_size):
    test_images = int(test_size * dataset.shape[0])
    test_indices = random.sample(range(dataset.shape[0]), test_images)
    train_indices = [x for x in range(dataset.shape[0])]
    for x in test_indices:
        train_indices.remove(x)
    X_test = np.array([dataset[x] for x in test_indices])
    Y_test = np.array([labels[x] for x in test_indices])
    X_train = np.array([dataset[x] for x in train_indices])
    Y_train = np.array([labels[x] for x in train_indices])
    return X_train, X_test, Y_train, Y_test


def evaluate_algorithm(X_train, X_test, Y_train, Y_test, metric='euclidean', num_neighbors=5):
    if len(Y_train) < num_neighbors:
        print("Not enough training images, use lower test_size.")
        print("Training images:", len(Y_train), "num_neighbors:", num_neighbors)
        return
    predictions = k_nearest_neighbors(X_train, Y_train, X_test, metric, num_neighbors)
    score = accuracy_metric(Y_test, predictions)
    return score


# Make a prediction with KNN
labels = []
for i in range(45):
    labels.append(random.randint(1, 4))

dataset = []
for i in range(45):
    image = []
    for m in range(10):
        temp_arr = np.random.random(10) + np.random.random(10) * 1j
        image.append(temp_arr)
    image = np.array(image)
    dataset.append(image)
dataset = np.array(dataset)
print('labels:', set(labels))

iterations = 100
metrics = ['ssim', 'euclidean', 'mse']
for metric in metrics:
    test_scores = []
    train_scores = []
    for test in range(iterations):
        train_img, test_img, train_id, test_id = knn_traintestsplit(dataset, labels, .8)
        score = evaluate_algorithm(train_img.copy(), test_img.copy(), train_id.copy(), test_id.copy(), metric)
        test_scores.append(score)
        score = evaluate_algorithm(train_img.copy(), train_img.copy(), train_id.copy(), train_id.copy(), metric)
        train_scores.append(score)
    print(metric, 'Accuracy: %s' % (sum(test_scores)/iterations))
    print(metric, 'Training Accuracy: %s' % (sum(train_scores)/iterations))
