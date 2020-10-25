import pandas as pd 
import umap.umap_ as umap
from sklearn.svm import SVC

class SVM:
    # svm loads, manipulates, and analyzes a given dataset
    def __init__(self, csv):
        df = pd.read_csv(csv)
        self.target = df.iloc[:, 0]
        self.data = df.iloc[:, 1:]
    
    # Define methods
    def scale(self):
        # We don't expect our data to be Gaussian, so let's use normalization
        min_val = self.data.min().min()
        max_val = self.data.max().max()
        self.data = (self.data - min_val) / (max_val - min_val)
        return self.data
    
    def umap(self):
        # Use UMAP for dimensionality reduction
        self.reducer = umap.UMAP().fit(self.data)
        self.data = pd.DataFrame(self.reducer.transform(self.data))
        return self.reducer
    
    def predict(self, kern='rbf'):
        # SVM splits the data using hyperplanes, aka some shape that divides the data. 
        # To make the data easier to split cleanly, SVM transforms the data
        # Using kernel functions. There are multiple types of kernels you can choose from, 
        # including linear, radial basis, and sigmoid.
        self.svmfit = SVC(kernel=kern).fit(self.data, self.target)
        self.pred = self.svmfit.predict(self.data)
        return self.pred

    def accuracy(self):
        return sum(self.pred == self.target)/len(self.pred)

# This part below is what actually runs when you run
# python svm.py
if __name__ == '__main__':
    mnist_svm = SVM("data/mnist_test.csv")
    mnist_svm.scale()
    mnist_svm.umap()
    mnist_svm.predict()
    print("For SVM scaled and reduced, the prediction accuracy was {}".format(mnist_svm.accuracy()))
