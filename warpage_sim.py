import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Material Properties ---
# Structure (Bottom to Top): PET -> PI -> SiOx -> OC -> SiOx
# Units needed for consistency:
# Thickness: meters (m)
# Modulus (E): Pascals (Pa)
# CTE (alpha): 1/Kelvin (1/K)

materials = [
    {'name': 'PET',         't': 100e-6, 'E': 3.0e9,  'CTE': 60e-6, 'v': 0.4},
    {'name': 'PI',          't': 20e-6,  'E': 5.0e9,  'CTE': 20e-6, 'v': 0.34},
    {'name': 'SiOx_bot',    't': 0.5e-6, 'E': 70.0e9, 'CTE': 0.5e-6, 'v': 0.17},
    {'name': 'OC',          't': 3.0e-6, 'E': 3.0e9,  'CTE': 60e-6, 'v': 0.4},
    {'name': 'SiOx_top',    't': 0.2e-6, 'E': 70.0e9, 'CTE': 0.5e-6, 'v': 0.17},
]

# Process Conditions
T_oven = 150.0  # Celsius
T_room = 25.0   # Celsius
delta_T = T_room - T_oven  # Cooling is negative delta T

# --- Calculation (Stoney / Timoshenko for Multilayer) ---
# We calculate the neutral axis and then the curvature.
# Reference: Townshend et al. or equivalent composite beam theory.
# Simplified approach: Transformed section method.

# 1. Calculate weighted average properties
# Transformation factor n_i = E_i / E_ref (Let's stick to ΣEi*ti and moments)

total_thickness = sum(m['t'] for m in materials)
width = 0.035 # m (3.5 cm)
length = 0.045 # m (4.5 cm)

# Indicial notation for sum:
# Numerator: Sum(E_i * t_i * alpha_i * delta_T * z_i) ? No, that's for stress.
# Let's use the force/moment balance matrix method for accuracy, or the equivalent CTE/Flexural Rigidity method.

# Method: Calculate Neutral Axis (z_n)
# z measured from bottom
current_z = 0
for m in materials:
    m['z_center'] = current_z + m['t'] / 2.0
    current_z += m['t']

numerator_zn = sum(m['E'] * m['t'] * m['z_center'] for m in materials)
denominator_zn = sum(m['E'] * m['t'] for m in materials)
z_n = numerator_zn / denominator_zn

# Calculate Flexural Rigidity (D or EI_eff)
EI_eff = 0
for m in materials:
    # Parallel axis theorem: I = I_local + A * d^2
    # I_local = (1/12) * w * t^3 (per unit width, w=1)
    I_local = (1.0/12.0) * (m['t']**3)
    d = m['z_center'] - z_n
    EI_eff += m['E'] * (I_local + m['t'] * d**2)

# Calculate Thermal Moment (M_T)
# M_T = Sum(E_i * alpha_i * delta_T * (z_i - z_n) * t_i) (per unit width)
M_T = 0
for m in materials:
    M_T += m['E'] * m['CTE'] * delta_T * (m['z_center'] - z_n) * m['t']

# Curvature (kappa) = M_T / EI_eff
# Note: Usually M = EI * kappa.
# The thermal strain induces an internal moment that causes bending.
# If we assume free expansion is restrained and then released to bend:
# Moment balance: M_internal + M_thermal = 0 ?
# Standard formula: curvature = (Sum Ei ti bi (zi-zn)) ... actually, the M_T formulation above essentially capture the "thermal moment" driving the bend.
# More precise: curvature = M_T / EI_eff gives the curvature induced to RELIEVE the thermal moment.
# Let's verify sign convention.
# If top expands MORE than bottom (or shrinks LESS), it bows up (concave down).
# Cooling: Delta T is negative.
# High CTE materials shrink MORE.
# If Top is high CTE, it shrinks more -> Concave Up (Smile).
# If Bottom is high CTE, it shrinks more -> Concave Down (Frown).

curvature = M_T / EI_eff 
radius = 1.0 / curvature if curvature != 0 else float('inf')

print(f"Neutral Axis z_n: {z_n*1e6:.2f} um")
print(f"Effective Stiffness EI: {EI_eff:.2e} Pa*m3")
print(f"Thermal Moment M_T: {M_T:.2f} N")
print(f"Calculated Curvature: {curvature:.4f} m^-1")
print(f"Radius of Curvature: {radius:.4f} m")

# --- Interpret Shape ---
# Positive Curvature in this derivation:
# M_T defined as E * a * dT * z.
# If z > zn (top) and dT < 0 (cool) and result is positive...
# Let's rethink simple logic check.
# Bottom: PET (thick, high CTE 60).
# Top: SiOx (stiff, low CTE) + OC (high CTE but thin).
# Major volume is PET (100um, 60ppm).
# PI (20um, 20ppm) is stiffer than PET but lower CTE.
# PET shrinks A LOT. PI shrinks less.
# Bottom shrinks MORE than top effective stack?
# If Bottom shrinks more, it pulls the edges in -> Concave Up (Bowl/Smile).
# Let's see the sign.

shape_desc = "Bowl (Concave Up)" if curvature > 0 else "Dome (Concave Down)"
# Wait, let's verify M_T sign.
# If bottom (z < z_n) has High CTE * neg dT -> Large Negative term.
# If top (z > z_n) has Low CTE * neg dT -> Small Negative term.
# (z - z_n) is neg for bottom, pos for top.
# Bottom term: E * CTE * neg_dT * neg_dist = POSITIVE Moment contribution.
# Top term: E * CTE * neg_dT * pos_dist = NEGATIVE Moment contribution.
# Since PET (bottom) is thick and high CTE, the POSITIVE term likely dominates.
# M_T > 0 implies Positive Curvature.
# Standard beam convention: Positive curvature = Concave Up d2y/dx2 > 0.
# So "Bowl" shape.

print(f"Predicted Shape: {shape_desc}")

# --- Plotting ---
x = np.linspace(-width/2, width/2, 50)
y = np.linspace(-length/2, length/2, 50)
X, Y = np.meshgrid(x, y)

# Deflection Z approx - (x^2 + y^2) / (2R) for spherical cap assumption?
# Or cylindrical? Assuming isotropic, spherical.
# Z = kappa * (x^2 + y^2) / 2
Z = curvature * (X**2 + Y**2) / 2.0

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X*100, Y*100, Z*1e6, cmap='viridis', edgecolor='none', alpha=0.8)

ax.set_title(f'Simulated Warpage: {shape_desc}\nCurvature: {curvature:.2f} $m^{-1}$', fontsize=14)
ax.set_xlabel('Width (cm)')
ax.set_ylabel('Length (cm)')
ax.set_zlabel('Deflection (um)')
# Create a dummy scalar map for the colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=Z.min()*1e6, vmax=Z.max()*1e6))
sm.set_array([])
fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5, label='Deflection (um)')

# Invert Z axis to show "bowl" properly if needed?
# If concave up (bowl), Z is positive at edges.
# Plot should look like a bowl.

plt.savefig('warpage_result.png')
print("Plot saved to warpage_result.png")
