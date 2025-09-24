#!/usr/bin/env python3
"""
Streamlit app: Interactive Friedmann scale-factor solver + plot.
t=0 is always the Big Bang.
Handles turnaround and recollapse robustly.
Shows present-day marker (a=1).
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Cosmology defaults
# ---------------------------
SEC_PER_GYR = 3.15576e16
MPC_IN_KM = 3.085677581e19
def H0_to_Gyr_inv(h0_km_s_Mpc):
    return h0_km_s_Mpc / MPC_IN_KM * SEC_PER_GYR  # Gyr^-1

DEFAULT_H0 = 67.4
DEFAULT_OMEGA_m = 0.313
DEFAULT_OMEGA_L = 0.687
DEFAULT_OMEGA_r = 9e-5  # small fixed radiation component

# Integration limits & tolerances
T_MAX = 50.0    # Gyr after Big Bang
ATOL = 1e-10
RTOL = 1e-8
MAX_STEP = 0.1  # Gyr
A_MIN = 1e-8    # minimum scale factor before stopping contraction

# ---------------------------
# Core cosmological functions
# ---------------------------
def E_of_a(a, omega_r, omega_m, omega_k, omega_L):
    return (omega_r * a**(-4) +
            omega_m * a**(-3) +
            omega_k * a**(-2) +
            omega_L)

def H_mag(a, H0_gyr, omega_r, omega_m, omega_k, omega_L):
    E = E_of_a(a, omega_r, omega_m, omega_k, omega_L)
    return H0_gyr * np.sqrt(max(E, 0.0))

def rhs_expansion(t, y, H0_gyr, omega_r, omega_m, omega_k, omega_L):
    a = y[0]
    return [a * H_mag(a, H0_gyr, omega_r, omega_m, omega_k, omega_L)]

def rhs_contraction(t, y, H0_gyr, omega_r, omega_m, omega_k, omega_L):
    a = y[0]
    return [-a * H_mag(a, H0_gyr, omega_r, omega_m, omega_k, omega_L)]

# ---------------------------
# Solve a(t)
# ---------------------------
def solve_scale_factor(H0_km_s_Mpc, omega_m, omega_L,
                       omega_r=DEFAULT_OMEGA_r,
                       t_max=T_MAX, num_points=1000):
    """
    Integrate from Big Bang (a~0) at t=0 forward in time up to t_max Gyr.
    Handles turnaround and contraction.
    """
    omega_k = 1.0 - (omega_r + omega_m + omega_L)
    H0_gyr = H0_to_Gyr_inv(H0_km_s_Mpc)

    a0 = 1e-8  # start at near-zero scale factor
    t0 = 0.0

    # event for turnaround
    def event_turnaround(t, y):
        a = y[0]
        return E_of_a(a, omega_r, omega_m, omega_k, omega_L)
    event_turnaround.terminal = True
    event_turnaround.direction = -1

    # first integrate expansion until turnaround or t_max
    sol_exp = solve_ivp(
        lambda t, y: rhs_expansion(t, y, H0_gyr, omega_r, omega_m, omega_k, omega_L),
        (t0, t_max),
        [a0],
        events=event_turnaround,
        atol=ATOL, rtol=RTOL, max_step=MAX_STEP, method='RK45'
    )

    t_list = list(sol_exp.t)
    a_list = list(sol_exp.y[0])

    # If turnaround event triggered, integrate contraction
    if len(sol_exp.t_events[0]) > 0:
        t_turn = sol_exp.t_events[0][0]
        a_turn = sol_exp.y_events[0][0][0]
        a_turn_nudged = max(a_turn * 0.999999, A_MIN)
        sol_contr = solve_ivp(
            lambda t, y: rhs_contraction(t, y, H0_gyr, omega_r, omega_m, omega_k, omega_L),
            (t_turn, t_max),
            [a_turn_nudged],
            atol=ATOL, rtol=RTOL, max_step=MAX_STEP, method='RK45'
        )
        t_list += list(sol_contr.t[1:])
        a_list += list(sol_contr.y[0][1:])

    t_arr = np.array(t_list)
    a_arr = np.array(a_list)

    # stop if we hit A_MIN
    mask = a_arr >= A_MIN
    t_arr = t_arr[mask]
    a_arr = a_arr[mask]

    # resample
    t_grid = np.linspace(t_arr[0], min(t_arr[-1], t_max), num_points)
    a_grid = np.interp(t_grid, t_arr, a_arr)

    return t_grid, a_grid, omega_k

# ---------------------------
# Find present-day time t when a=1
# ---------------------------
def find_present_time(t_grid, a_grid):
    idx = np.where(a_grid >= 1.0)[0]
    if len(idx) == 0:
        return None
    first_idx = idx[0]
    if first_idx == 0:
        return t_grid[0]
    a1, a2 = a_grid[first_idx-1], a_grid[first_idx]
    t1, t2 = t_grid[first_idx-1], t_grid[first_idx]
    return t1 + (1.0 - a1)*(t2 - t1)/(a2 - a1)

# ---------------------------
# Streamlit app UI
# ---------------------------
st.set_page_config(page_title="Friedmann Scale Factor", layout="wide")

st.title("Interactive Friedmann Scale Factor a(t)")
st.write("Adjust cosmological parameters and see the scale factor evolve over time.")

col1, col2, col3 = st.columns(3)

with col1:
    omega_m = st.slider(r"Ωₘ (Matter)", 0.0, 1.0, DEFAULT_OMEGA_m, 0.001)
with col2:
    omega_L = st.slider(r"ΩΛ (Cosmological Constant)", -1.0, 1.0, DEFAULT_OMEGA_L, 0.001)
with col3:
    H0_val = st.slider("H₀ [km/s/Mpc]", 30.0, 120.0, DEFAULT_H0, 0.1)

# Solve model
t, a, omega_k = solve_scale_factor(H0_val, omega_m, omega_L)
t_present = find_present_time(t, a)

# Plot using matplotlib
fig, ax = plt.subplots(figsize=(9,6))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.plot(t, a, color='orange', lw=2)

# present-day marker
if t_present is not None:
    ax.axvline(t_present, color='white', ls='--', lw=1)
    ax.text(t_present, 1.02, 'Present day',
            rotation=90, color='white', va='bottom', ha='center')

ax.set_xlabel("Cosmic time since Big Bang (Gyr)", color='white')
ax.set_ylabel("Scale factor a(t)", color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

ax.set_xlim(0, T_MAX)
ax.set_ylim(0, max(a)*1.05)

ax.text(0.98, 0.98, f"$\\Omega_k$ = {omega_k:+.4f}",
        transform=ax.transAxes, ha='right', va='top', color='white', fontsize=10,
        bbox=dict(facecolor='black', edgecolor='white', pad=6, alpha=0.6))

ax.set_title("Friedmann scale factor a(t) (t=0 at Big Bang)", color='white')

st.pyplot(fig)



