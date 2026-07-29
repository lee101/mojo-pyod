import numpy as np

from mojopyod.models.knn import KNN

rng = np.random.default_rng(7)
X = rng.normal(size=(500, 8))
X[-10:] += 6.0

detector = KNN(contamination=0.02, n_neighbors=10, method="largest")
labels = detector.fit_predict(X)

print(labels.sum())
print(detector.decision_scores_[labels == 1])
