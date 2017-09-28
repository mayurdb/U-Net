from skeleton import Segment
import numpy as np
import traceback
import math
from skimage.io import imread
import gzip
# from scipy.spatial.distance import russellrao
# from scipy.spatial.distance import hamming

class Evaluator(object):

    def __init__(self, data_dir):
        self.path = data_dir

    def load_images(self):
        self.X = []
        self.Y = []
        for i in range(5):
            self.X.append(imread(self.path+'/test/'+str(i)+'.jpg'))
            # self.X.append(imread(self.path+'/Train_Data/train-'+str(i)+'-mask.jpg'))
            self.Y.append(imread(self.path+'/test/'+str(i)+'-mask.jpg'))
            # self.Y.append(imread(self.path+'/Train_Data/train-0-mask.jpg'))

    def get_iou(self, im1, im2):
        y = im1.flatten()
        x = im2.flatten()
        x[np.where(x <= 127)] = 0
        x[np.where(x > 127)] = 1
        print(x)
        print(y)
        a = np.sum(np.bitwise_and(x,y))
        b = np.sum(np.bitwise_or(x,y))
        iou = a/float(b)
        # c11 = x.shape[0]-x.shape[0]*russellrao(x,y)
        # c01 = x.shape[0]*hamming(x,y)
        # iou = c11/float(c11+c01)
        print("IOU =",iou)
        return iou

    def evaluate(self, model):
        correct = 0

        for i in range(len(self.X)):
            print("Processing image", i, "of", len(self.X))
            target = self.Y[i]
            try:
                pred = np.asarray(model.get_mask(self.X[i]))
                if pred.shape==target.shape:
                    if self.get_iou(pred, target) >= 0.5:
                        correct += 1
            except Exception as e:
                print(traceback.print_exc())

        accuracy = correct/float(len(self.X))
        score = 0
        if accuracy <0.5:
            score = 15.0 * accuracy
        else:
            score = 7.5*(math.exp(1.386*(accuracy-0.5)))

        print("Accuracy =", accuracy*100)
        print("Marks =", score)

if __name__ == "__main__":

        evaluator = Evaluator("data")
        evaluator.load_images()
        obj = Segment.load_model()
        evaluator.evaluate(obj)
