'''
    File name         : KalmanFilter.py
    Description       : KalmanFilter class used for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, A, H, Q, R, m0, v0):
       
        # Define sampling time
        self.dt = dt

        # Define the State Transition Matrix A
        self.A = A

        # Intial State
        self.x = m0

        # Define Measurement Mapping Matrix
        self.H = H

        #Initial Process Noise Covariance
        self.Q = Q


        #Initial Measurement Noise Covariance
        self.R = R

        #Initial Covariance Matrix
        self.P = v0


    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795

        # Update time state
        #x_k =Ax_(k-1)     Eq.(9)
        
        self.x = np.dot(self.A, self.x)
        
        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(self.A, np.dot(self.P, np.transpose(self.A))) + self.Q
        
        return self.x, self.P

    def update(self, z):
        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        
        S = np.dot(self.H, np.dot(self.P, np.transpose(self.H))) + self.R
        S = (S+np.transpose(S))/2
        
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(self.P, np.dot(np.transpose(self.H), np.linalg.inv(S)))  #Eq.(11)

        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))   #Eq.(12)

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x, self.P


    def updateMissing(self):

        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman Gain ignored
        
        K = np.zeros(np.transpose(self.H).shape)  #Eq.(11)

        # self.x = np.round(self.x)   #Eq.(12)

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x, self.P

        
        
