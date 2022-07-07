from KalmanFilter import KalmanFilter

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math

import cv2
from get_background import get_background

import sys

import filter_smooth


def rmse(x, x2):
    tot=0
    for i in range(len(x)):
        tot+=(x[i]-x2[i])

    return math.sqrt(abs(tot)/len(x))

def plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total):

    trace_mes = go.Scatter(x=measuredx, y=measuredy,
                           mode="markers",
                           name="measured",
                           showlegend=True,
                           )
    trace_true = go.Scatter(x=truex, y=truey,
                            mode="markers",
                            name="true",
                            showlegend=True,
                            )
    trace_filtered = go.Scatter(x=filteredx,
                                y=filteredy,
                                mode="markers",
                                name="filtered",
                                showlegend=True,
                                )
    trace_smoothed = go.Scatter(x=smoothedx,
                                y=smoothedy,
                                mode="markers",
                                name="smoothed",
                                showlegend=True,
                                )
    fig = go.Figure()
    fig.add_trace(trace_mes)
    fig.add_trace(trace_true)
    fig.add_trace(trace_filtered)
    fig.add_trace(trace_smoothed)
    fig.update_layout(xaxis_title="x (pixels)", yaxis_title="y (pixels)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    #import pdb; pdb.set_trace()

def plotVelocities(measured, filtered, smoothed, dt, N):
    fd1 = []
    fd2 = []

    for i in range(1, N):
        fd1.append( (measured[i, 0] - measured[i-1, 0]) / dt)
        fd2.append( (measured[i, 1] - measured[i-1,1]) / dt)

    x = np.arange(0, N*dt, dt)
    
    trace_fdx = go.Scatter(x=x, y=fd1,
                           mode="markers",
                           name="Finite Diff x",
                           marker=dict(
                                color='red',
                                line=dict(
                                    color='red'
                                )
                            ),
                           showlegend=True,
                           )
    trace_fdy = go.Scatter(x=x, y=fd2,
                           mode="markers",
                           marker_symbol="circle-open",
                           name="Finite Diff y",
                           marker=dict(
                               color='red',
                                line=dict(
                                    color='red'
                                )
                            ),
                           showlegend=True,
                           )
    trace_smx = go.Scatter(x=x, y=smoothed[1, 0, :N],
                            mode="markers",
                            name="Smoothed x",
                           marker=dict(
                                color='green',
                                line=dict(
                                    color='green'
                                )
                            ),
                            showlegend=True,
                            )
    trace_smy = go.Scatter(x=x, y=smoothed[4, 0, :N],
                           mode="markers",
                           marker_symbol="circle-open",
                           name="Smoothed y",
                           marker=dict(
                               color='green',
                                line=dict(
                                    color='green'
                                )
                            ),
                           showlegend=True,
                           )
    trace_filtx = go.Scatter(x=x, y=filtered[1, 0, :N],
                            mode="markers",
                            name="Filtered x",
                             marker=dict(
                                color='purple',
                                line=dict(
                                    color='purple'
                                )
                            ),
                            showlegend=True,
                            )
    trace_filty = go.Scatter(x=x, y=filtered[4, 0, :N],
                            mode="markers",
                           marker_symbol="circle-open",
                            name="Filtered y",
                             marker=dict(
                                 color='purple',
                                line=dict(
                                    color='purple'
                                )
                            ),
                            showlegend=True,
                            )
    
    fig = go.Figure()
    fig.add_trace(trace_fdx)
    fig.add_trace(trace_fdy)
    fig.add_trace(trace_filtx)
    fig.add_trace(trace_filty)
    fig.add_trace(trace_smy)
    fig.add_trace(trace_smx)
    fig.update_layout(xaxis_title="Time", yaxis_title="Velocity",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    #import pdb; pdb.set_trace()
    return fd1, fd2


def plotAccelerations(fd_v1, fd_v2, filtered, smoothed, dt, N):
    fd1 = []
    fd2 = []

    for i in range(1, N-1):
        fd1.append( (fd_v1[i] - fd_v1[i-1]) / dt)
        fd2.append( (fd_v2[i] - fd_v2[i-1]) / dt)

    x = np.arange(0, N*dt, dt)
    
    trace_fdx = go.Scatter(x=x, y=fd1,
                           mode="markers",
                           name="Finite Diff x.",
                           marker=dict(
                                color='red',
                                line=dict(
                                    color='red'
                                )
                            ),
                           showlegend=True,
                           )
    trace_fdy = go.Scatter(x=x, y=fd2,
                           mode="markers",
                           marker_symbol="circle-open",
                           name="Finite Diff y",
                           marker=dict(
                                color='red',
                                line=dict(
                                    color='red'
                                )
                            ),
                           showlegend=True,
                           )
    trace_smx = go.Scatter(x=x, y=smoothed[2, 0, :N],
                            mode="markers",
                            name="Smoothed x",
                           marker=dict(
                                color='green',
                                line=dict(
                                    color='green'
                                )
                            ),
                            showlegend=True,
                            )
    trace_smy = go.Scatter(x=x, y=smoothed[5, 0, :N],
                            mode="markers",
                           marker_symbol="circle-open",
                            name="Smoothed y",
                           marker=dict(
                                color='green',
                                line=dict(
                                    color='green'
                                )
                            ),
                            showlegend=True,
                            )
    trace_filtx = go.Scatter(x=x, y=filtered[2, 0, :N],
                            mode="markers",
                            name="Filtered x",
                             marker=dict(
                                color='purple',
                                line=dict(
                                    color='purple'
                                )
                            ),
                            showlegend=True,
                            )
    trace_filty = go.Scatter(x=x, y=filtered[5, 0, :N],
                            mode="markers",
                           marker_symbol="circle-open",
                            name="Filtered y",
                             marker=dict(
                                color='purple',
                                line=dict(
                                    color='purple'
                                )
                            ),
                            showlegend=True,
                            )
    
    fig = go.Figure()
    fig.add_trace(trace_fdx)
    fig.add_trace(trace_fdy)
    fig.add_trace(trace_filtx)
    fig.add_trace(trace_filty)
    fig.add_trace(trace_smx)
    fig.add_trace(trace_smy)
    fig.update_layout(xaxis_title="Time", yaxis_title="Acceleration",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()

def compareOutputs(measx, f1x, f2x, measy, f1y, f2y):

    fig, ax = plt.subplots()

    line1,=ax.plot(measx, measy, label="measured")
    line2,=ax.plot(f1x, f1y, label="mine")
    line3,=ax.plot(f2x, f2y, label="Joaquin's")


    lines=[line1, line2, line3]

    leg=ax.legend()
    graphs = {}

    lineLegends=leg.get_lines()
    
    for i in range(len(lines)):
        lineLegends[i].set_picker(True)
        lineLegends[i].set_pickradius(5)
        graphs[lineLegends[i]]=lines[i]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()

        graphs[legend].set_visible(not isVisible)
        legend.set_visible(not isVisible)

        fig.canvas.draw()

    plt.connect('pick_event', on_pick)


    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


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
    
        
    if readFromCSV:
        measured = np.genfromtxt('data/postions_session003_start0.00_end15548.27.csv',delimiter=',')
        
        total = 10000

        measured = measured[1:total+1]

        contaminated = np.empty((total,2))
        contaminated[:] = np.nan

        noise = 1e-10
        
        for i in range(total):
            contaminated[i, 0] = measured[i][1] + noise
            contaminated[i, 1] = measured[i][2] + noise

        filtered, smoothed=filter_smooth.main([])


        #plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total)
        plotPositions(measured[:,1], filtered["xnn"][0,0,:], filtered["xnn1"][0,0,:], contaminated[:,0], smoothed["xnN"][0,0,:], measured[:,2], filtered["xnn"][3,0,:], filtered["xnn1"][3,0,:], contaminated[:,1], smoothed["xnN"][3,0,:], total)

        # plotVelocities(measured, filtered, smoothed, dt, N)
        fd_v1, fd_v2 = plotVelocities(contaminated, filtered["xnn"], smoothed["xnN"], dt, total)

        # plotAccelerations(fd_v1, fd_v2, filtered, smoothed, dt, N)
        plotAccelerations(fd_v1, fd_v2, filtered["xnn"], smoothed["xnN"], dt, total)
        
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
