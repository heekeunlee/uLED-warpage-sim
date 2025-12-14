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

    # --- Bottom Layer Logic (Added) ---
    if bottom_layer_type != 'None':
        # Add the selected bottom layer at index 0 (Bottom)
        if bottom_layer_type == 'Copper (Cu)':
             materials.insert(0, {'name': 'Copper (Cu)', 't': bottom_thickness_um*1e-6, 'E': 110e9, 'CTE': 17e-6, 'v': 0.34})
        elif bottom_layer_type == 'Aluminum (Al)':
             materials.insert(0, {'name': 'Aluminum (Al)', 't': bottom_thickness_um*1e-6, 'E': 69e9, 'CTE': 23e-6, 'v': 0.33})
        elif bottom_layer_type == 'Stainless Steel (SUS)':
             materials.insert(0, {'name': 'Stainless Steel', 't': bottom_thickness_um*1e-6, 'E': 200e9, 'CTE': 16e-6, 'v': 0.3})
        elif bottom_layer_type == 'Glass':
             materials.insert(0, {'name': 'Glass', 't': bottom_thickness_um*1e-6, 'E': 70e9, 'CTE': 0.5e-6, 'v': 0.2})
             
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
    return curvature, materials, delta_T

# Sidebar Inputs for Bottom Layer
st.sidebar.subheader("Dome Shape Solution")
bottom_layer_type = st.sidebar.selectbox("Add Bottom Layer", ['None', 'Copper (Cu)', 'Aluminum (Al)', 'Stainless Steel (SUS)', 'Glass'])
if bottom_layer_type != 'None':
    bottom_thickness_um = st.sidebar.slider("Layer Thickness (um)", 10, 200, 50, step=10)
else:
    bottom_thickness_um = 0

curvature, materials_data, delta_T = calculate_warpage()
radius = 1.0 / curvature if curvature != 0 else float('inf')
shape_desc = "Bowl (Concave Up)" if curvature > 0 else "Dome (Concave Down)"

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Curvature**: {curvature:.2f} $m^{{-1}}$")
st.sidebar.markdown(f"**Radius**: {radius*100:.2f} cm")
st.sidebar.markdown(f"**Shape**: {shape_desc}")

# --- Material Properties Table ---
st.markdown("### Simulation Parameters")
import pandas as pd
df_materials = pd.DataFrame(materials_data)
# Clean up for display
df_display = df_materials.copy()
df_display['Thickness (um)'] = df_display['t'] * 1e6
df_display['Modulus (GPa)'] = df_display['E'] / 1e9
df_display['CTE (ppm/K)'] = df_display['CTE'] * 1e6
df_display = df_display[['name', 'Thickness (um)', 'Modulus (GPa)', 'CTE (ppm/K)', 'v']]
df_display.columns = ['Layer', 'Thickness (um)', 'Modulus (GPa)', 'CTE (ppm/K)', 'Poisson Ratio']

st.table(df_display)
st.caption(f"Process Condition: Cooling form 150°C to 25°C (ΔT = {delta_T} K)")

st.markdown("---")


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

st.markdown("---")

# --- Thickness Analysis Graph ---
st.header("Thickness Optimization Graph")
st.markdown("Simulation of bottom foil thickness vs. warpage (Dome Shape).")

if st.checkbox("Show Thickness Analysis Graphs", value=True):
    # Simulation Logic for Range
    def simulate_range(mat_name, mat_props, t_range_um):
        curvatures = []
        radii = []
        
        base_stack = [
           {'name': 'PET',         't': 100e-6, 'E': 3.0e9,  'CTE': 60e-6, 'v': 0.4},
           {'name': 'PI',          't': 20e-6,  'E': 5.0e9,  'CTE': 20e-6, 'v': 0.34},
           {'name': 'SiOx_bot',    't': 0.5e-6, 'E': 70.0e9, 'CTE': 0.5e-6, 'v': 0.17},
           {'name': 'OC',          't': 3.0e-6, 'E': 3.0e9,  'CTE': 60e-6, 'v': 0.4},
           {'name': 'SiOx_top',    't': 0.2e-6, 'E': 70.0e9, 'CTE': 0.5e-6, 'v': 0.17},
        ]
        
        # Process Conditions (Same as main)
        T_oven = 150.0 
        T_room = 25.0   
        delta_T = T_room - T_oven

        for t_um in t_range_um:
            # Create stack
            stack = [{'name': mat_name, 't': t_um * 1e-6, **mat_props}] + [m.copy() for m in base_stack]
            
            # Calc Neutral Axis
            current_z = 0
            for m in stack:
                m['z_center'] = current_z + m['t'] / 2.0
                current_z += m['t']
            
            num_zn = sum(m['E'] * m['t'] * m['z_center'] for m in stack)
            den_zn = sum(m['E'] * m['t'] for m in stack)
            z_n = num_zn / den_zn
            
            # Calc EI
            EI_eff = 0
            for m in stack:
                I_local = (1.0/12.0) * (m['t']**3)
                d = m['z_center'] - z_n
                EI_eff += m['E'] * (I_local + m['t'] * d**2)
            
            # Calc M_T
            M_T = 0
            for m in stack:
                M_T += m['E'] * m['CTE'] * delta_T * (m['z_center'] - z_n) * m['t']
            
            k = M_T / EI_eff
            r = 1.0 / k if k != 0 else 0
            
            curvatures.append(k)
            radii.append(abs(r) * 100) # cm
            
        return curvatures, radii

    t_values = np.linspace(10, 100, 20) # 10um to 100um
    
    # Copper
    cu_props = {'E': 110e9, 'CTE': 17e-6, 'v': 0.34}
    k_cu, r_cu = simulate_range('Copper', cu_props, t_values)
    
    # Aluminum
    al_props = {'E': 69e9, 'CTE': 23e-6, 'v': 0.33}
    k_al, r_al = simulate_range('Aluminum', al_props, t_values)
    
    # Plotting
    import pandas as pd
    
    # Curvature Plot
    df_k = pd.DataFrame({
        'Thickness (um)': t_values,
        'Copper (Cu)': k_cu,
        'Aluminum (Al)': k_al
    })
    df_k = df_k.set_index('Thickness (um)')
    st.markdown("#### Curvature ($m^{-1}$) vs Thickness")
    st.write("Negative curvature indicates Dome shape (Convex). More negative = More bent.")
    st.line_chart(df_k)
    
    # Radius Plot
    df_r = pd.DataFrame({
        'Thickness (um)': t_values,
        'Copper (Cu)': r_cu,
        'Aluminum (Al)': r_al
    })
    df_r = df_r.set_index('Thickness (um)')
    st.markdown("#### Radius of Curvature (cm) vs Thickness")
    st.write("Smaller radius = More bent.")
    st.line_chart(df_r)
