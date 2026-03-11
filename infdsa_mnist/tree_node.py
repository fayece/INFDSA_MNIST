class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, prediction=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction