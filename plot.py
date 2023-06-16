import plotly.graph_objects as go
import numpy as np

# If true data isn't available pass False instead

def plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total):
    trace_mes = go.Scatter(x=measuredx, y=measuredy,
                           mode="markers",
                           name="measured",
                           showlegend=True,
                           )
    if truex.all()!=False and truey.all()!=False:
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
    if truex.all()!=False and truey.all()!=False:
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

    if true_vel.all()!=False:
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
    if true_vel.all()!=False:
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
    if true_acc.all()!=False:
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
    if true_acc.all()!=False:
        fig.add_trace(trace_truex)
        fig.add_trace(trace_truey)
    fig.add_trace(trace_smx)
    fig.add_trace(trace_smy)
    fig.update_layout(xaxis_title="Time", yaxis_title="Acceleration",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()

