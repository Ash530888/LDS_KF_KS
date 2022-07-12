import numpy as np

import inference
import learning
import plot
import torch

import pickle

def rmse(x, x2):
    tot=0
    for i in range(len(x)-1):
        tot+=(x[i]-x2[i])

    return math.sqrt(abs(tot)/len(x))



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

    B = B.numpy()
    Z = Z.numpy()

    sigma_a = float(optim_res['x'][0][0])
    sqrt_diag_R = np.diag([optim_res['x'][2][0], optim_res['x'][2][1]])
    m0 = optim_res['x'][3].numpy()
    sqrt_diag_V0 = np.diag(optim_res['x'][4].numpy())

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


def main():
    N=10000
    dt = 1e-3
    
    # LESS NOISY SIM
##    x_std_meas = y_std_meas = 1e-10
##    std_acc = 1e1

    # MORE NOISY SIM
    x_std_meas = y_std_meas = 1e-3
    std_acc = 1.0

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
    #learnParams(1e2 , 1e2, 1e2, dt, N)

    
    filtered = inference.filterLDS(measured, B, Q, m0, V0, Z, R)
    smoothed = inference.smoothLDS(B, filtered["xnn"], filtered["Vnn"], filtered["xnn1"], filtered["Vnn1"], m0, V0)

    # plotVelocities(true_vel, measured, filtered, smoothed, dt, N)
    fd_v1, fd_v2 = plot.plotVelocities(true_vel, measured, filtered["xnn"], smoothed["xnN"], dt, N)

    # plotAccelerations(fd_v1, fd_v2, true_acc, filtered, smoothed, dt, N)
    plot.plotAccelerations(fd_v1, fd_v2, true_acc, filtered["xnn"], smoothed["xnN"], dt, N)
    
    # plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total)
    
    plot.plotPositions(x[0, :], filtered["xnn"][0, 0, :], filtered["xnn1"][0, 0, :],
                  measured[0, :], smoothed["xnN"][0, 0, :], x[3, :],
                  filtered["xnn"][3, 0, :], filtered["xnn1"][3, 0, :],
                  measured[1, :], smoothed["xnN"][3, 0, :], N)

    


if __name__ == "__main__":
    main()
