from KalmanFilter import KalmanFilter

import numpy as np
import plot
import math

import cv2
from get_background import get_background

import sys

import filter_smooth

# use to evaluate accuracy
def rmse(x, x2):
    tot=0
    for i in range(len(x)):
        tot+=(x[i]-x2[i])

    return math.sqrt(abs(tot)/len(x))


def main(readFromCSV=False):
    dt=0.019936
    x_std_meas=y_std_meas=1e-10
    std_acc= 1e-10
    
    A = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                   [0, 1, dt, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, dt, .5*dt**2],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]],
                  dtype=np.double)

    H = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0]],
                            dtype=np.double)

    Q =  np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)*std_acc**2

    R = np.diag([x_std_meas**2, y_std_meas**2])

    m0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.double)
    V0 = np.diag(np.ones(len(m0))*0.001)
    
        
    if readFromCSV:
        measured = np.genfromtxt('data/postions_session003_start0.00_end15548.27.csv',delimiter=',')

        total = 10000
        #total = measured.shape[0] -1

        measured = measured[1:total+1]

        filtered, smoothed=filter_smooth.main([])


        # plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total)
        plot.plotPositions(False, filtered["xnn"][0,0,:], filtered["xnn1"][0,0,:], measured[:,1], smoothed["xnN"][0,0,:], False, filtered["xnn"][3,0,:], filtered["xnn1"][3,0,:], measured[:,2], smoothed["xnN"][3,0,:], total)

        # plotVelocities(true_vel, measured, filtered, smoothed, dt, N)
        fd_v1, fd_v2 = plot.plotVelocities(False, measured[:,[1,2]].T, filtered["xnn"], smoothed["xnN"], dt, total)

        # plotAccelerations(fd_v1, fd_v2, true_acc, filtered, smoothed, dt, N)
        plot.plotAccelerations(fd_v1, fd_v2,False, filtered["xnn"], smoothed["xnN"], dt, total)
        
    else:
        cap = cv2.VideoCapture("mouse.avi")
        # get the video frame height and width
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        measured=np.empty((total,2))
        predicted=np.empty((total,2))
        filtered=np.empty((total,2))
        
        matrixIndex=0

        # get the background model
        background = get_background("mouse.avi")
        # convert the background model to grayscale format
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        frame_count = 0
        consecutive_frame = 8


        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                                 
                frame_count += 1
                orig_frame = frame.copy()
                # IMPORTANT STEP: convert the frame to grayscale first
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame_count % consecutive_frame == 0 or frame_count == 1:
                    frame_diff_list = []
                # find the difference between current frame and base frame
                frame_diff = cv2.absdiff(gray, background)
                # thresholding to convert the frame to binary
                ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
                # dilate the frame a bit to get some more white area...
                # ... makes the detection of contours a bit easier
                dilate_frame = cv2.dilate(thres, None, iterations=2)
                # append the final result into the `frame_diff_list`
                frame_diff_list.append(dilate_frame)
                # if we have reached `consecutive_frame` number of frames
                
                if len(frame_diff_list) == consecutive_frame:
                    # add all the frames in the `frame_diff_list`
                    sum_frames = sum(frame_diff_list)
                    # find the contours around the white segmented areas
                    contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # draw the contours, not strictly necessary
                    for i, cnt in enumerate(contours):
                        cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
                    if len(contours)>0:

                        # continue through the loop if contour area is less than 500...
                        # ... helps in removing noise detection
                        cAreas=[]
                        
                        for i in range(len(contours)):
                            cAreas.append(cv2.contourArea(contours[i]))
                        
                        i=cAreas.index(max(cAreas))
                        
                        if cv2.contourArea(contours[i]) < 250:
                            sp, _ = KF.predict()

                            x1 = sp[0]
                            y1 = sp[3]
                            cv2.rectangle(orig_frame, (int(x1), int(y1)),(int( x1 + w), int(y1 + h)), (255, 0, 0), 2)
                            cv2.putText(orig_frame, "Predicted Position", (int(x1 + w), int(y1)), 0, 0.5, (255, 0, 0), 2)
                            
                            h=36
                            w=31
                            su, _ = KF.updateMissing()
                            x = su[0]
                            y = su[3]
                            cv2.rectangle(orig_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                            cv2.putText(orig_frame, "Filtered Position", (int(x + 30), int(y +25)), 0, 0.5, (0, 0, 255), 2)
                            cv2.imshow('Detected Objects', orig_frame)
                            continue
                        
                        # get the xmin, ymin, width, and height coordinates from the contours
                        (x, y, w, h) = cv2.boundingRect(contours[i])

                        
                        cv2.putText(orig_frame, "Measured Position", (int(x + 15), int(y - 15)), 0, 0.5, (0, 255, 0), 2)

                        # draw the bounding boxes
                        cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        try:
                            sp, _ = KF.predict()
                        except UnboundLocalError:
                            m0=np.matrix([[x], [0], [0], [y], [0], [0]])

                            v0 = np.diag(np.ones(m0.shape[0])*0.001)

                            #Create KalmanFilter object KF
                            #KalmanFilter(dt, A, H, Q, R, m0, v0)
                            KF = KalmanFilter(dt, A, H, Q, R, m0, v0)
                            sp, _ = KF.predict()
                            
                        x1 = sp[0]
                        y1 = sp[3]
                        cv2.rectangle(orig_frame, (int(x1), int(y1)),(int( x1 + w), int(y1 + h)), (255, 0, 0), 2)
                        cv2.putText(orig_frame, "Predicted Position", (int(x1 + w), int(y1)), 0, 0.5, (255, 0, 0), 2)

                        z = np.array([[x, y]]).T
                        su, _ = KF.update(z=z)
                        x = su[0]
                        y = su[3]
                        cv2.rectangle(orig_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

                        cv2.putText(orig_frame, "Filtered Position", (int(x + 30), int(y +25)), 0, 0.5, (0, 0, 255), 2)

                        
                    else:
                        sp, _ = KF.predict()
                        x = sp[0]
                        y = sp[3]
                        cv2.rectangle(orig_frame, (int(x), int(y)),(int( x + w), int(y + h)), (255, 0, 0), 2)

                        cv2.putText(orig_frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)

                        su, _ = KF.updateMissing()
                        x = su[0]
                        y = su[3]
                        cv2.rectangle(orig_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

                        cv2.imshow('Detected Objects', orig_frame)

                        cv2.putText(orig_frame, "Filtered Position", (int(x + 30), int(y +25)), 0, 0.5, (0, 0, 255), 2)
                        
                    cv2.imshow('Detected Objects', orig_frame)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

main(True)
