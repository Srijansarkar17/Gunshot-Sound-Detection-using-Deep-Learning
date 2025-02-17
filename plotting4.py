import numpy as np
import plotly.graph_objs as go
from scipy.optimize import least_squares

# Constants
SPEED_OF_SOUND = 343  # m/s

# Microphone positions (hexagonal arrangement)
R = 50  # radius of hexagon in meters
MIC_POSITIONS = np.array([
    [R * np.cos(theta), R * np.sin(theta), 0]
    for theta in np.linspace(0, 2 * np.pi, 7)[:-1]
])

def calculate_tdoa(toa):
    """Calculate TDOA for all microphone pairs."""
    n_mics = len(toa)
    tdoa = np.zeros((n_mics, n_mics))
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            tdoa[i, j] = toa[j] - toa[i]
            tdoa[j, i] = -tdoa[i, j]  # Ensure symmetry
    return tdoa

def hyperbolic_residuals(point, mic_positions, tdoa):
    """Residual function for least squares optimization."""
    x, y, z = point
    residuals = []
    for i in range(len(mic_positions)):
        for j in range(i + 1, len(mic_positions)):
            d1 = np.sqrt((x - mic_positions[i][0])**2 + 
                         (y - mic_positions[i][1])**2 + 
                         (z - mic_positions[i][2])**2)
            d2 = np.sqrt((x - mic_positions[j][0])**2 + 
                         (y - mic_positions[j][1])**2 + 
                         (z - mic_positions[j][2])**2)
            residuals.append((d1 - d2) - SPEED_OF_SOUND * tdoa[i, j])
    return residuals

def locate_gunshot(toa):
    """Locate the gunshot source given ToA values using least squares."""
    tdoa = calculate_tdoa(toa)

    # Initial guess: center of the microphone array, slightly above the plane
    initial_guess = np.mean(MIC_POSITIONS, axis=0) + [0, 0, 5]

    # Perform least squares optimization
    result = least_squares(hyperbolic_residuals, initial_guess, args=(MIC_POSITIONS, tdoa))

    return result.x, tdoa

def plot_results_octant_3d_plotly(source_location, mic_positions):
    """Plot the results in 3D octant map with microphones, source, and directions using Plotly."""
    
    # Create traces for microphones and source
    mic_trace = go.Scatter3d(
        x=mic_positions[:, 0],
        y=mic_positions[:, 1],
        z=mic_positions[:, 2],
        mode='markers+text',
        text=[f'Mic {i+1}' for i in range(len(mic_positions))],
        marker=dict(size=5, color='red'),
        name='Microphones'
    )
    
    source_trace = go.Scatter3d(
        x=[source_location[0]],
        y=[source_location[1]],
        z=[source_location[2]],
        mode='markers+text',
        text=['Estimated Source'],
        marker=dict(size=10, color='green', symbol='diamond'),
        name='Estimated Source'
    )
    
    # Base trace (hexagon)
    theta = np.linspace(0, 2 * np.pi, 7)
    x_base = R * np.cos(theta)
    y_base = R * np.sin(theta)
    z_base = np.zeros_like(theta)
    
    base_trace = go.Scatter3d(
        x=x_base,
        y=y_base,
        z=z_base,
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Microphone Base'
    )
    
    # Source vector trace
    vector_trace = go.Scatter3d(
        x=[0, source_location[0]],
        y=[0, source_location[1]],
        z=[0, source_location[2]],
        mode='lines',
        line=dict(color='purple', dash='dot'),
        name='Source Vector'
    )
    
    # Create surfaces for the octants
    ax_limit = max(R * 1.5, abs(source_location[0]), abs(source_location[1]), abs(source_location[2])) * 1.2
    xx, yy = np.meshgrid([-ax_limit, ax_limit], [-ax_limit, ax_limit])
    
    # Surfaces representing octants
    surfaces = [
        go.Surface(x=xx, y=yy, z=np.zeros_like(xx), opacity=0.1, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']]),
        go.Surface(x=np.zeros_like(xx), y=xx, z=yy, opacity=0.1, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']]),
        go.Surface(x=xx, y=np.zeros_like(xx), z=yy, opacity=0.1, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']])
    ]
    
    # Combine all traces
    data = [mic_trace, source_trace, base_trace, vector_trace] + surfaces
    
    # Define the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X (meters)', range=[-ax_limit, ax_limit]),
            yaxis=dict(title='Y (meters)', range=[-ax_limit, ax_limit]),
            zaxis=dict(title='Z (meters)', range=[-ax_limit, ax_limit]),
            aspectratio=dict(x=1, y=1, z=1),
            annotations=[
                dict(
                    showarrow=False,
                    text=dir_label,
                    x=dir_pos[0],
                    y=dir_pos[1],
                    z=dir_pos[2],
                    xanchor='center',
                    yanchor='middle'
                ) for dir_label, dir_pos in {
                    'N': (0, ax_limit, 0),
                    'S': (0, -ax_limit, 0),
                    'E': (ax_limit, 0, 0),
                    'W': (-ax_limit, 0, 0),
                    'NE': (ax_limit * 0.707, ax_limit * 0.707, 0),
                    'NW': (-ax_limit * 0.707, ax_limit * 0.707, 0),
                    'SE': (ax_limit * 0.707, -ax_limit * 0.707, 0),
                    'SW': (-ax_limit * 0.707, -ax_limit * 0.707, 0)
                }.items()
            ]
        ),
        title='3D Gunshot Source Localization (Plotly Octant Map)'
    )
    
    # Create figure
    fig = go.Figure(data=data, layout=layout)
    
    # Show the plot
    fig.show()
    fig.write_html("3d_plot.html")

# Example ToA values
toa = np.array([0.5, 0.6, 0.8, 0.3, 0.2, 0.1])

# Locate the gunshot
estimated_location, tdoa = locate_gunshot(toa)

print(f"Estimated source location: ({estimated_location[0]:.2f}, {estimated_location[1]:.2f}, {estimated_location[2]:.2f}) meters")

# Plot the results in 3D octant map using Plotly
plot_results_octant_3d_plotly(estimated_location, MIC_POSITIONS)
