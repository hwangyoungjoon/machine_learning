import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class svm:
    def __init__(self,visualization=True):
        self.visualization=visualization
        self.colors={1:'r',-1:'b'}
        if self.visualization:
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(1,1,1)
    def fit(self,data):
        self.data=data
        # {||w||:[w,b]}
        opt_dict={}
        transforms=[[1,1],
                    [-1,1],
                    [-1,-1],
                    [1,-1]]
        all_data=[]
        for yi in self.data:
            # print(yi)
            for featureset in self.data[yi]:
                # print(featureset)
                for feature in featureset:
                    # print(feature)
                    all_data.append(feature)
        #support vector yi(xi.w+b)=1
        # print(all_data)
        self.max_feature_value=max(all_data)
        self.min_feature_value=min(all_data)
        all_data=None
        # print(self.max_feature_value,self.min_feature_value)

        step_sizes=[self.max_feature_value*0.1,
                    self.max_feature_value*0.01,
                    self.max_feature_value*0.001]
        # print(step_sizes)
        b_range_multiple=5
        b_multiple=5
        latest_optimum=self.max_feature_value*10

        for step in step_sizes:
            w=np.array([latest_optimum, latest_optimum])
            print(w)
            # we can do this because convex
            optimized=False
            while not optimized:

                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                self.max_feature_value*b_range_multiple,
                                step*b_multiple):
                    for transformation in transforms:
                        w_t=w*transformation
                        # print(w_t)
                        found_option=True
                        #weakest link in the svm fundamentally
                        #smo attemps to fix this a bit
                        #yi(xi.w+b)>=1

                        for i in self.data:
                            yi=i
                            for xi in self.data[i]:
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    # print(yi*(np.dot(w_t,xi)+b))
                                    found_option=False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)]=[w_t,b]
                            # print(opt_dict)
                if w[0]<0:
                    optimized=True
                    # print("optimized a step")
                else:
                    # w=[5,5]
                    # w-step=[4,4]

                    w=w-step
            norms=sorted([n for n in opt_dict])
            opt_choice=opt_dict[norms[0]]

            self.w=opt_choice[0]
            self.b=opt_choice[1]
            latest_optimum=opt_choice[0][0]+step*2


    def predict(self,feature):
        # sign(x.w+b)
        classification=np.sign(np.dot(np.array(feature),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

            #hyperplane=x.w+b
            # v=x.w+b
            #positive_sv=1
            #negative_sv=-1
            #decision_bound=0
        def hyperplane(x,w,b,v):

            return (-w[0]*x-b+v)/w[1]

        datarange=(self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min=datarange[0]
        hyp_x_max=datarange[1]
        print(datarange)
        #(w.x+b)=1
        #positive sv
        psv1=hyperplane(hyp_x_min,self.w,self.b,1)
        psv2=hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])
        print(psv1,psv2)
        # (w.x+b)=-1
        #negative sv
        nsv1=hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2=hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])
        print(nsv1,nsv2)
        # (w.x+b)=0

        db1=hyperplane(hyp_x_min,self.w,self.b,0)
        db2=hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])
        print(db1,db2)
        plt.show()





data_dict={-1:np.array([[1,7],
                        [2,8],
                        [3,8],]),
            1:np.array([[5,1],
                        [6,-1],
                        [7,3],])}
support_vc=svm()
support_vc.fit(data=data_dict)
support_vc.visualize()
