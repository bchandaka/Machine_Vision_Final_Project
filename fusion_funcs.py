import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pykitti
from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy.signal import lfilter
import sys
from vo_funcs import VisualOdometry
from tqdm import tqdm
import os
DATA_DIR = './data' 

def load_data(date, drive):
    data = pykitti.raw(DATA_DIR, date, drive)
    return data

class SensorFusion():
    def __init__(self, start_state, data: pykitti.raw, cutout_region=list(range(80,150)), use_smoothing=False):
        # Input Data
        self.data = data
        _, self.t_start = pykitti.utils.pose_from_oxts_packet(data.oxts[0].packet, scale=1)
        self.vo = VisualOdometry(data)

        # Kalman States
        self.state = start_state
        self.pred_state = start_state

        # Simulation cutout
        self.cutout_region = cutout_region
        self.cutout_var = 10

        degree = 30
        # self.R_oxts_cam = np.array(R.from_euler('z', degree, degrees=True).as_matrix())
        self.R_oxts_cam = np.linalg.inv(data.calib.T_cam0_imu[:3,:3])
        # History info
        # self.pose_est = [self.state]
        # self.pose_gt = [self.getGPS(0)]
        # self.pose_gt_w_cutout = [self.getGPS(0)[:2]]
        # self.pose_vo = [self.getGPS(0)]
        self.pose_est = []
        self.pose_gt = []
        self.pose_gt_w_cutout = []
        self.pose_vo = []
        self.K_hist = []

        # Setting up smoothing
        self.use_smoothing = use_smoothing
        self.define_lfilter_vals()
        
    def getOXTSMeasurement(self, index, H):
        t = self.getGPS(index)

        x, y, z = t[:3]
        x_vel = self.data.oxts[index].packet.ve
        y_vel = self.data.oxts[index].packet.vn
        z_vel = self.data.oxts[index].packet.vu
        x_vel, y_vel, z_vel = self.R_oxts_cam @ np.array([x_vel, y_vel, z_vel])
        # print("OXTS vel", x_vel)
        pos_uncertainty = self.data.oxts[index].packet.pos_accuracy
        vel_uncertainty = self.data.oxts[index].packet.vel_accuracy

        if index in self.cutout_region:
            pos_uncertainty = self.cutout_var # meters
            x = x + np.random.normal(0, pos_uncertainty)
            y = y + np.random.normal(0, pos_uncertainty)
        
        R = np.diag([pos_uncertainty**2, pos_uncertainty**2, vel_uncertainty**2, vel_uncertainty**2])

        return H @ np.array([x, y, x_vel, y_vel]), R

    def getGPS(self, index):
        R, t = pykitti.utils.pose_from_oxts_packet(self.data.oxts[index].packet, scale=1)
        t -= self.t_start
        t = self.R_oxts_cam @ t
        return t

    def setCutout(self, startInd, endInd):
        self.cutout_region = list(range(startInd, endInd))

    def setCutoutVar(self, var):
        self.cutout_var = var

    def setDegOffset(self, degree):
        self.R_oxts_cam = np.array(R.from_euler('z', degree, degrees=True).as_matrix())

    # ---Smoothing---
    def define_lfilter_vals(self):
        self.lfilter_b = [signal.firwin(8, 0.004), signal.firwin(8, 0.004)]
        self.lfilter_z = [np.zeros(self.lfilter_b[0].size-1), np.zeros(self.lfilter_b[1].size-1)]

    def get_smoothed_est(self, x, i):
        new_x, self.lfilter_z[0] = signal.lfilter(self.lfilter_b[0], 1, [x[0]], zi=self.lfilter_z[0])
        new_y, self.lfilter_z[1] = signal.lfilter(self.lfilter_b[1], 1, [x[1]], zi=self.lfilter_z[1])
        # if i in self.cutout_region:
        if len(self.pose_est) > self.lfilter_b[0].size:
            x[0] = new_x
            x[1] = new_y
        return x

    # ---Kalman Filter---
    def predict(self, A, B, u, P, Q):
        predicted_X = A @ self.state + B @ u
        predicted_P = A @ P @ A.T + Q
        return predicted_X, predicted_P

    def calc_gain(self, P, H, R):
        numerator = P@H.T
        denominator = H@P@H.T + R
        K = numerator/denominator
        # print(K)
        K = np.diag(np.diag(K))
        # K[0,1] = 0
        # K[1,0] = 0
        return K

    def update(self, K, z, H, P):
        # print(K)
        updated_X = self.pred_state + K @ (z - H@self.pred_state)
        n = len(K@H)
        updated_P = (np.eye(n)-K@H)@P
        return updated_X, updated_P
        
    def run(self, A, B, P, Q, H, n_iter=-1):
        dt = 0.1
        for i, gt_pose in enumerate(tqdm(self.vo.gt_poses[:n_iter], unit="poses")):
            # Get VO control input velocity
            if i < 1:
                cur_pose = self.vo.gt_poses[0]
                delta_x = cur_pose[0, 3]
                delta_y = cur_pose[2, 3]
                self.pose_vo.append(cur_pose[:3,3])
            else:
                transf = self.vo.get_pose(i)
                old_pose = np.copy(cur_pose)
                cur_pose = np.matmul(cur_pose, transf)
                self.pose_vo.append(1.5*cur_pose[:3,3])
                # print(cur_pose)
                delta_x, delta_z, delta_y = 1.5*(cur_pose[:3, 3] - (old_pose[:3, 3]))
            x_vo_velocity = delta_x/dt
            y_vo_velocity = delta_y/dt
            # print("X_vel", x_vo_velocity)
            # print(x_vo_velocity, self.data.oxts[i].packet.vf, self.data.oxts[i].packet.vl)
            u = np.array([ 
                0,
                0,
                x_vo_velocity, #+ 0.2*self.data.oxts[i].packet.vn
                y_vo_velocity
            ])
            # print("Iteration", i)
            #
            self.pred_state, predicted_P = self.predict(A, B, u, P, Q)
            z, R = self.getOXTSMeasurement(i, H)
            self.pose_gt_w_cutout.append(z[:2])
            K = self.calc_gain(predicted_P, H, R)

            # print("z:", z, "predicted_X", self.pred_state)

            self.state, P = self.update(K, z, H, predicted_P)
            if self.use_smoothing:
                self.state = self.get_smoothed_est(self.state, i)
            # print("Update_x:", self.state, "GT:", self.getGPS(i))
            
            self.pose_est.append(self.state)
            gt_t = gt_pose[:3, 3]
            gt_t = self.R_oxts_cam @ gt_t
            self.pose_gt.append(gt_t)
            # self.pose_gt.append(self.getGPS(i))
        self.pose_est = np.array(self.pose_est)
        self.pose_gt = np.array(self.pose_gt)
        self.pose_vo = np.array(self.pose_vo)
        self.pose_gt_w_cutout = np.array(self.pose_gt_w_cutout)
    def plot_res(self, exp_name, filename, vis_cutout):
        cutout_num = "$\sigma_{cutout} = "+ repr(self.cutout_var) + "$"
        figure, (pos_axis,err_axis) = plt.subplots(1, 2, figsize=(12.8,4.8))
        pos_axis.plot(self.pose_est[:,0], self.pose_est[:,1], label="KF Estimates", color='blue')
        pos_axis.plot(self.pose_gt[:,0], self.pose_gt[:,1], label="Ground Truth", color='orange')
        pos_axis.plot(self.pose_vo[:,0], self.pose_vo[:,2], label="Visual Odometry", color='green')

        if vis_cutout:
            pos_axis.scatter(self.pose_gt_w_cutout[:,0], self.pose_gt_w_cutout[:,1], label="GT_w_cutout", color='red')
        pos_axis.set_xlabel("X Position (m)")
        pos_axis.set_ylabel("Y Position (m)")
        pos_axis.set_title("Vehicle Position")
        pos_axis.legend()
        kf_error = np.linalg.norm(self.pose_est[:,:2]-self.pose_gt[:,:2], axis=-1)
        gt_error = np.linalg.norm(self.pose_gt_w_cutout[:,:2]-self.pose_gt[:,:2], axis=-1)
        vo_error = np.linalg.norm(self.pose_vo[:,[0,2]]-self.pose_gt[:,:2], axis=-1)
        t = np.linspace(0,len(vo_error)*0.1,len(vo_error))
        err_axis.plot(t, gt_error, label="Cutout Error", color='red')
        err_axis.plot(t, vo_error, label="VO Error", color='green')
        err_axis.plot(t, kf_error, label="KF Error", color='blue')
        err_axis.set_ylabel("Euclidean Distance Error (m)")
        err_axis.set_xlabel("Time (s)")
        err_axis.set_title("Distance Error over Time")
        err_axis.legend()
        plt.suptitle( ', '.join([exp_name, cutout_num]))
        plt.savefig(os.path.join('results', filename), dpi=100)

        plt.show()
    def print_metrics(self):
        pose_gt_cutout = self.pose_gt[self.cutout_region]
        pose_est_cutout = self.pose_est[self.cutout_region]
        pose_cutout_cutout = self.pose_gt_w_cutout[self.cutout_region]
        pose_vo_cutout = self.pose_vo[self.cutout_region]

        avg_kf_error = np.mean(np.linalg.norm(pose_gt_cutout[:,:2]-pose_est_cutout[:,:2], axis=-1))
        avg_cutout_error = np.mean(np.linalg.norm(pose_gt_cutout[:,:2]-pose_cutout_cutout[:,:2], axis=-1))
        avg_vo_error = np.mean(np.linalg.norm(pose_gt_cutout[:,:2]-pose_vo_cutout[:,[0,2]], axis=-1))

        print(f"KF Error: {avg_kf_error}, VO Error: {avg_vo_error}, Cutout Error: {avg_cutout_error}")

    def run_default(self, n_iter=-1):
        dt = 0.1
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        B = np.array([ 
            [0, 0, dt, 0],
            [0, 0, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        A[:,2:] *=0.5
        B[:,2:] *=0.5

        P = np.diag([10,10,10,10])
        Q = np.diag([1,1,1,1])
        H = np.diag([1,1,1,1])
        self.run(A, B, P, Q, H, n_iter)