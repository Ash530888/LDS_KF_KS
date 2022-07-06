import plotly.graph_objects as go
import numpy as np

import inference
import learning
import torch

import pickle

def rmse(x, x2):
    tot=0
    for i in range(len(x)-1):
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


def plotVelocities(true_vel, measured, filtered, smoothed, dt, N):
    fd1 = []
    fd2 = []

    for i in range(1, N):
        fd1.append( (measured[0, i] - measured[0, i-1]) / dt)
        fd2.append( (measured[1, i] - measured[1, i-1]) / dt)

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
    trace_truex = go.Scatter(x=x, y=true_vel[0, :],
                            mode="markers",
                            name="True x",
                             marker=dict(
                                color='blue',
                                line=dict(
                                    color='blue'
                                )
                            ),
                            showlegend=True,
                            )
    trace_truey = go.Scatter(x=x, y=true_vel[1, :],
                            mode="markers",
                           marker_symbol="circle-open",
                            name="True y",
                             marker=dict(
                                color='blue',
                                line=dict(
                                    color='blue'
                                )
                            ),
                            showlegend=True,
                            )
    
    fig = go.Figure()
    fig.add_trace(trace_fdx)
    fig.add_trace(trace_fdy)
    fig.add_trace(trace_filtx)
    fig.add_trace(trace_filty)
    fig.add_trace(trace_truex)
    fig.add_trace(trace_truey)
    fig.add_trace(trace_smx)
    fig.add_trace(trace_smy)
    fig.update_layout(xaxis_title="Time", yaxis_title="Velocity",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    return fd1, fd2


def plotAccelerations(fd_v1, fd_v2, true_acc, filtered, smoothed, dt, N):
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
    trace_truex = go.Scatter(x=x, y=true_acc[0, :],
                            mode="markers",
                            name="True x",
                             marker=dict(
                                color='blue',
                                line=dict(
                                    color='blue'
                                )
                            ),
                            showlegend=True,
                            )
    trace_truey = go.Scatter(x=x, y=true_acc[1, :],
                            mode="markers",
                           marker_symbol="circle-open",
                            name="True y",
                             marker=dict(
                                color='blue',
                                line=dict(
                                    color='blue'
                                )
                            ),
                            showlegend=True,
                            )
    
    fig = go.Figure()
    fig.add_trace(trace_fdx)
    fig.add_trace(trace_fdy)
    fig.add_trace(trace_filtx)
    fig.add_trace(trace_filty)
    fig.add_trace(trace_truex)
    fig.add_trace(trace_truey)
    fig.add_trace(trace_smx)
    fig.add_trace(trace_smy)
    fig.update_layout(xaxis_title="Time", yaxis_title="Acceleration",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()

def learnParams(std_acc, x_std_meas, y_std_meas, dt, N):
    B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, .5*dt**2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]],
                 dtype=np.double)

    Z = np.matrix([[1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]],
                  dtype=np.double)

    Q =  np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)*(std_acc**2)

    R = np.diag([x_std_meas**2, y_std_meas**2])

    m0 = np.array([0, 0, 0, 0, 0, 0 ])

    V0 = np.diag(np.ones(len(m0))*(0.001**2))

    M = B.shape[0]
    P = Z.shape[0]

##    Sigma = np.array([[1e-5, 1e-5, 1e-5, 0, 0, 0],
##                   [1e-5, 1e-5,   1e-5,      0, 0, 0],
##                   [1e-5, 1e-5,   1e-5,       0, 0, 0],
##                   [0, 0, 0, 1e-5, 1e-5, 1e-5],
##                   [0, 0, 0, 1e-5, 1e-5,   1e-5],
##                   [0, 0, 0, 1e-5, 1e-5,    1e-5]],
##                  dtype=np.double)
##
##    Gamma = np.diag([100, 100])

    w = np.transpose(np.random.multivariate_normal(np.zeros(M), Q, N))
    v = np.transpose(np.random.multivariate_normal(np.zeros(P), R, N))
    
    measured = np.empty((P, N))
    true_vel = np.empty((2, N))
    true_acc = np.empty((2, N))
    measured[:] = np.nan
    x = np.empty((M, N))
    
    x0 = np.random.multivariate_normal(m0, V0)
    measured[:, 0] = np.add(np.dot(Z, x0).squeeze(), v[:,0])

    
    for i in range(N):
        if i==0:
            x[:, i] = np.add(np.dot(B, x0), w[:,i])
        else:
            x[:, i] = np.add(np.dot(B, x[:, i-1]), w[:,i])

        true_vel[0, i] = x[1, i]
        true_vel[1, i] = x[4, i]

        true_acc[0, i] = x[2, i]
        true_acc[1, i] = x[5, i]
        
        measured[:, i] = np.add(np.dot(Z, x[:, i]).squeeze(), v[:, i]) 


    #INITIAL
    filtered = inference.filterLDS(measured, B, Q, m0, V0, Z, R)
    smoothed = inference.smoothLDS(B, filtered["xnn"], filtered["Vnn"], filtered["xnn1"], filtered["Vnn1"], m0, V0)

    # plotVelocities(true_vel, measured, filtered, smoothed, dt, N)
    fd_v1, fd_v2 = plotVelocities(true_vel, measured, filtered["xnn"], smoothed["xnN"], dt, N)

    # plotAccelerations(fd_v1, fd_v2, true_acc, filtered, smoothed, dt, N)
    plotAccelerations(fd_v1, fd_v2, true_acc, filtered["xnn"], smoothed["xnN"], dt, N)
    
    plotPositions(x[0, :], filtered["xnn"][0, 0, :], filtered["xnn1"][0, 0, :],
                  measured[0, :], smoothed["xnN"][0, 0, :], x[3, :],
                  filtered["xnn"][3, 0, :], filtered["xnn1"][3, 0, :],
                  measured[1, :], smoothed["xnN"][3, 0, :], N)

    #LEARNING
    sqrt_diag_R_0 = torch.DoubleTensor([x_std_meas**2, y_std_meas**2])
    m0_0 = torch.DoubleTensor([x0[0], x0[1], x0[2], x0[3], x0[4], x0[5]])
    sqrt_diag_V0_0 = torch.DoubleTensor([0.001
                                         for i in range(len(m0_0))])

    y = torch.from_numpy(measured.astype(np.double))
    B = torch.from_numpy(B.astype(np.double))
    Qt = torch.from_numpy(Q.astype(np.double))
    Z = torch.from_numpy(Z.astype(np.double))
    optim_res = learning.torch_optimize_SS_tracking_DWPA_diagV0(
        y=y, B=B, sigma_ax0=std_acc, sigma_ay0=std_acc, Qt=Qt,
        Z=Z, sqrt_diag_R_0=sqrt_diag_R_0, m0_0=m0_0,
        sqrt_diag_V0_0=sqrt_diag_V0_0, max_iter=N)

    print("optim res.")
    print(optim_res)
    print(optim_res['x'])

    B = B.numpy()
    Z = Z.numpy()

    sigma_a = float(optim_res['x'][0][0])
    sqrt_diag_R = np.diag([optim_res['x'][2][0], optim_res['x'][2][1]])
    print(sqrt_diag_R)
    m0 = optim_res['x'][3].numpy()
    print(m0)
    sqrt_diag_V0 = np.diag(optim_res['x'][4].numpy())
    print(sqrt_diag_V0)

    Q =  np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)*(sigma_a**2)

    R = sqrt_diag_R

    V0 = sqrt_diag_V0

##    M = B.shape[0]
##    P = Z.shape[0]
##
##    measured = np.empty((P, N))
##    true_vel = np.empty((2, N))
##    true_acc = np.empty((2, N))
##    measured[:] = np.nan
##    x = np.empty((M, N))
##    
##    x0 = np.random.multivariate_normal(m0, V0)
##    measured[:, 0] = np.add(np.dot(Z, x0).squeeze(), v[:,0])
##
##    w = np.transpose(np.random.multivariate_normal(np.zeros(M), Q, N))
##    v = np.transpose(np.random.multivariate_normal(np.zeros(P), R, N))
##    
##    for i in range(N):
##        if i==0:
##            x[:, i] = np.add(np.dot(B, x0), w[:,i])
##        else:
##            x[:, i] = np.add(np.dot(B, x[:, i-1]), w[:,i])
##
##        true_vel[0, i] = x[1, i]
##        true_vel[1, i] = x[4, i]
##
##        true_acc[0, i] = x[2, i]
##        true_acc[1, i] = x[5, i]
##        
##        measured[:, i] = np.add(np.dot(Z, x[:, i]).squeeze(), v[:, i])

    filtered = inference.filterLDS(measured, B, Q, m0, V0, Z, R)
    smoothed = inference.smoothLDS(B, filtered["xnn"], filtered["Vnn"], filtered["xnn1"], filtered["Vnn1"], m0, V0)

    # plotVelocities(true_vel, measured, filtered, smoothed, dt, N)
    fd_v1, fd_v2 = plotVelocities(true_vel, measured, filtered["xnn"], smoothed["xnN"], dt, N)

    # plotAccelerations(fd_v1, fd_v2, true_acc, filtered, smoothed, dt, N)
    plotAccelerations(fd_v1, fd_v2, true_acc, filtered["xnn"], smoothed["xnN"], dt, N)
    
    plotPositions(x[0, :], filtered["xnn"][0, 0, :], filtered["xnn1"][0, 0, :],
                  measured[0, :], smoothed["xnN"][0, 0, :], x[3, :],
                  filtered["xnn"][3, 0, :], filtered["xnn1"][3, 0, :],
                  measured[1, :], smoothed["xnN"][3, 0, :], N)

    input("continue... ")


def main():
    N=10000
    dt = 1e-5
    
    # LESS NOISY SIM
##    x_std_meas = y_std_meas = 1e-10
##    std_acc = 1e-10

    # MORE NOISY SIM
    x_std_meas = y_std_meas = 1e2
    std_acc = 1e4

    B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, .5*dt**2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]],
                 dtype=np.double)

    Z = np.matrix([[1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]],
                  dtype=np.double)

    Q =  np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)*(std_acc**2)

    R = np.diag([x_std_meas**2, y_std_meas**2])

    m0 = np.array([0, 0, 0, 0, 0, 0 ])

    V0 = np.diag(np.ones(len(m0))*(0.001**2))

    M = B.shape[0]
    P = Z.shape[0]
    
    w = np.transpose(np.random.multivariate_normal(np.zeros(M), Q, N))
    v = np.transpose(np.random.multivariate_normal(np.zeros(P), R, N))

    # measured=np.empty((N,2))
    measured = np.empty((P, N))
    true_vel = np.empty((2, N))
    true_acc = np.empty((2, N))
    measured[:] = np.nan
    x = np.empty((M, N))
    # predicted=np.empty((N,2))
    # filtered=np.empty((N,2))
    # y = np.zeros(shape=(P, N))

    # noise=np.random.normal(0, 1000, N)

    # x = np.add(m0, w[0,0])
    x0 = np.random.multivariate_normal(m0, V0)
    measured[:, 0] = np.add(np.dot(Z, x0).squeeze(), v[:,0])

    # measured[0,:] = y[:,0] + noise[0]

    # for i in range(1, N):
    for i in range(N):
        if i==0:
            x[:, i] = np.add(np.dot(B, x0), w[:,i])
        else:
            x[:, i] = np.add(np.dot(B, x[:, i-1]), w[:,i])

        true_vel[0, i] = x[1, i]
        true_vel[1, i] = x[4, i]

        true_acc[0, i] = x[2, i]
        true_acc[1, i] = x[5, i]
        
        measured[:, i] = np.add(np.dot(Z, x[:, i]).squeeze(), v[:, i])

        # measured[i,:] = y[:,i] + noise[i]

    # learnParams(std_acc, x_std_meas, y_std_meas, dt, N)
    learnParams(1e2 , 1e2, 1e2, dt, N)

    
    filtered = inference.filterLDS(measured, B, Q, m0, V0, Z, R)
    smoothed = inference.smoothLDS(B, filtered["xnn"], filtered["Vnn"], filtered["xnn1"], filtered["Vnn1"], m0, V0)

    # plotVelocities(true_vel, measured, filtered, smoothed, dt, N)
    fd_v1, fd_v2 = plotVelocities(true_vel, measured, filtered["xnn"], smoothed["xnN"], dt, N)

    # plotAccelerations(fd_v1, fd_v2, true_acc, filtered, smoothed, dt, N)
    plotAccelerations(fd_v1, fd_v2, true_acc, filtered["xnn"], smoothed["xnN"], dt, N)
    
    # plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total)
    # plotPositions(y[0,:], filtered["xnn"][0,0,: N], filtered["xnn1"][0,0,: N], measured[:,0], smoothed["xnN"][0,0,: N], y[1, :], filtered["xnn"][3,0,: N], filtered["xnn1"][3,0,: N], measured[:,1], smoothed["xnN"][3,0,: N], N)
    plotPositions(x[0, :], filtered["xnn"][0, 0, :], filtered["xnn1"][0, 0, :],
                  measured[0, :], smoothed["xnN"][0, 0, :], x[3, :],
                  filtered["xnn"][3, 0, :], filtered["xnn1"][3, 0, :],
                  measured[1, :], smoothed["xnN"][3, 0, :], N)

    


if __name__ == "__main__":
    main()
