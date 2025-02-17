import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import random

# Constants
R = 50  # radius of hexagon in meters
MIC_POSITIONS = np.array([
    [R * np.cos(theta), R * np.sin(theta), 0]
    for theta in np.linspace(0, 2 * np.pi, 7)[:-1]
])

def create_base_plot():
    # Create the base plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
    
    # Microphone positions
    fig.add_trace(go.Scatter3d(
        x=MIC_POSITIONS[:, 0],
        y=MIC_POSITIONS[:, 1],
        z=MIC_POSITIONS[:, 2],
        mode='markers+text',
        text=[f'Mic {i+1}' for i in range(len(MIC_POSITIONS))],
        marker=dict(size=5, color='red'),
        name='Microphones'
    ))
    
    # Base hexagon
    theta = np.linspace(0, 2 * np.pi, 7)
    x_base = R * np.cos(theta)
    y_base = R * np.sin(theta)
    z_base = np.zeros_like(theta)
    fig.add_trace(go.Scatter3d(
        x=x_base,
        y=y_base,
        z=z_base,
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Microphone Base'
    ))
    
    # Octant surfaces
    ax_limit = R * 1.5
    xx, yy = np.meshgrid([-ax_limit, ax_limit], [-ax_limit, ax_limit])
    for surface in [
        go.Surface(x=xx, y=yy, z=np.zeros_like(xx), opacity=0.1, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']]),
        go.Surface(x=np.zeros_like(xx), y=xx, z=yy, opacity=0.1, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']]),
        go.Surface(x=xx, y=np.zeros_like(xx), z=yy, opacity=0.1, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']])
    ]:
        fig.add_trace(surface)
    
    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (meters)', range=[-ax_limit, ax_limit]),
            yaxis=dict(title='Y (meters)', range=[-ax_limit, ax_limit]),
            zaxis=dict(title='Z (meters)', range=[-ax_limit, ax_limit]),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        title='Gunshot Detection System Visualization'
    )
    
    return fig

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph', figure=create_base_plot()),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('live-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n):
    # This function simulates getting data from the Verilog module
    # In a real implementation, you would interface with your hardware here
    
    # Simulate gunshot detection
    gunshot_detected = random.choice([True, False])
    
    if gunshot_detected:
        # Simulate coordinates from Verilog module
        x = random.uniform(-R, R)
        y = random.uniform(-R, R)
        z = random.uniform(0, R)
    else:
        x, y, z = 0, 0, 0
    
    fig = go.Figure(create_base_plot())
    
    if gunshot_detected:
        # Add gunshot location
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=10, color='green', symbol='diamond'),
            name='Detected Gunshot'
        ))
        
        # Add direction vector
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines',
            line=dict(color='purple', width=5),
            name='Direction'
        ))
        
        # Make the direction line blink
        fig.data[-1].line.dash = 'dot' if n % 2 == 0 else 'solid'
    
    fig.update_layout(title='Gunshot Detection System Visualization')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)