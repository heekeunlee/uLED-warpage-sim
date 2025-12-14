import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Streamlit Config ---
st.set_page_config(layout="wide", page_title="uLED Warpage Simulation")

st.title("uLED Panel Warpage Simulation")
st.markdown("### Interactive 3D Visualization")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
z_scale = st.sidebar.slider("Vertical Visual Scale (Exaggeration)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
show_wireframe = st.sidebar.checkbox("Show Wireframe", value=True)

# --- Simulation Logic (Reused) ---
def calculate_warpage():
    materials = [
        {'name': 'PET',         't': 100e-6, 'E': 3.0e9,  'CTE': 60e-6, 'v': 0.4},
        {'name': 'PI',          't': 20e-6,  'E': 5.0e9,  'CTE': 20e-6, 'v': 0.34},
        {'name': 'SiOx_bot',    't': 0.5e-6, 'E': 70.0e9, 'CTE': 0.5e-6, 'v': 0.17},
        {'name': 'OC',          't': 3.0e-6, 'E': 3.0e9,  'CTE': 60e-6, 'v': 0.4},
        {'name': 'SiOx_top',    't': 0.2e-6, 'E': 70.0e9, 'CTE': 0.5e-6, 'v': 0.17},
    ]

    T_oven = 150.0 
    T_room = 25.0   
    delta_T = T_room - T_oven

    current_z = 0
    for m in materials:
        m['z_center'] = current_z + m['t'] / 2.0
        current_z += m['t']

    numerator_zn = sum(m['E'] * m['t'] * m['z_center'] for m in materials)
    denominator_zn = sum(m['E'] * m['t'] for m in materials)
    z_n = numerator_zn / denominator_zn

    EI_eff = 0
    for m in materials:
        I_local = (1.0/12.0) * (m['t']**3)
        d = m['z_center'] - z_n
        EI_eff += m['E'] * (I_local + m['t'] * d**2)

    M_T = 0
    for m in materials:
        M_T += m['E'] * m['CTE'] * delta_T * (m['z_center'] - z_n) * m['t']

    curvature = M_T / EI_eff 
    return curvature

curvature = calculate_warpage()
radius = 1.0 / curvature if curvature != 0 else float('inf')
shape_desc = "Bowl (Concave Up)" if curvature > 0 else "Dome (Concave Down)"

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Curvature**: {curvature:.2f} $m^{{-1}}$")
st.sidebar.markdown(f"**Radius**: {radius*100:.2f} cm")
st.sidebar.markdown(f"**Shape**: {shape_desc}")

# --- 3D Plotting ---
width = 0.035 # m
length = 0.045 # m

x = np.linspace(-width/2, width/2, 50)
y = np.linspace(-length/2, length/2, 50)
X, Y = np.meshgrid(x, y)

# Z deflection in meters
Z_meters = curvature * (X**2 + Y**2) / 2.0
# Z for display (convert to microns) - NO SCALING here, scaling is visual
Z_microns = Z_meters * 1e6 

# Create Plotly Surface
# We plot the ACTUAL micron values
fig = go.Figure(data=[go.Surface(z=Z_microns, x=X*100, y=Y*100, colorscale='Viridis', opacity=0.9)])

# Aspect Ratio Logic for "Exaggeration"
# We control the 'z' component of the aspect ratio to stretch the box visually.
# x_ratio = 1.0 (Base)
# y_ratio = length/width (To keep X/Y relative proportion correct)
# z_ratio = Controlled by slider.
#   - If z_scale is small (e.g. 0.1), box is flat.
#   - If z_scale is large (e.g. 2.0), box is tall.

aspect_ratio_x = 1.0
aspect_ratio_y = length / width
aspect_ratio_z = z_scale  # Direct mapping from slider (0.1 ~ 3.0 recommended)

fig.update_layout(
    title=f'Warpage visualization (Visual Exaggeration: {z_scale:.1f})',
    scene={
        'xaxis': {'title': 'Width (cm)'},
        'yaxis': {'title': 'Length (cm)'},
        'zaxis': {'title': 'Deflection (um)'},
        'aspectmode': 'manual',
        'aspectratio': dict(x=aspect_ratio_x, y=aspect_ratio_y, z=aspect_ratio_z)
    },
    autosize=True,
    width=800,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### How to interpret
- **Z-axis**: Shows the actual deflection in microns (um).
- **Slider**: Adjusts the **visual height** of the Z-axis to make the curvature easier to see.
- **Color**: Represents the height (yellow = high, blue = low).
""")
