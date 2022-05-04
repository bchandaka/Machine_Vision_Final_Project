import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pykitti
from vo_funcs import VisualOdometry
from tqdm import tqdm
import pykitti 

class KalmanFilter():
    def __init__(self, data: pykitti.raw):
        self.data = data
        _, self.t_start = pykitti.utils.pose_from_oxts_packet(data.oxts[0].packet, scale=1)

    def getGT(self, index):
        R, t = pykitti.utils.pose_from_oxts_packet(self.data.oxts[index].packet, scale=1)
        t -= self.t_start
        return t

    def predict(A, X_vec, B, u, P, Q):
        predicted_X = A @ X_vec + B @ u
        predicted_P = A @ P @ A.T + Q
        return predicted_X, predicted_P

    def calc_gain(P, H, R):
        numerator = P@H.T
        denominator = H@P@H.T + R
        K = numerator/denominator
        print(K)
        K[0,1] = 0
        K[1,0] = 0
        return K
    def update(X_vec, K, z, H, P):
        print(K)
        updated_X = X_vec + K @ (z - H@X_vec)
        n = len(K@H)
        updated_P = (np.eye(n)-K@H)@P
        return updated_X, updated_P
    def getMeasurement(index, H):
        pass
    def simulateCutout(startInd, endInd):
        pass

