import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class svm:
    def __init__(self,visualization=True):
        self.visualization=visualization
        self.colors={1:"r",-1:"b"}
        if self.visualization:
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(1,1,1)

    def fit(self,data):
        self.data=data
        #{|| w ||: [w,b]}
        opt_dict={}
        transforms=[[1,1],
                    [-1,1],
                    [-1,-1],
                    [1,-1]]
        all_data=[ ]
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value=max(all_data)
        self.min_feature_value=min(all_data)
        all_data=None




    def predict(self,data):
        #sign(x.w+b)
        classification=np.sign(np.dot(np.array(features),self.w)+self.b)
        return classification
