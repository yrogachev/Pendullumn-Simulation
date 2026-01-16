# two body orbit sandbox
# multiple reference frames
# energy and momentum tracking

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons









# constants
g_const = 1.0 
dt_base = 0.05
speed_factor = 1.0

# state (inertial frame)
m1 = 100.0
m2 = 10.0

p1 = np.array([0.0, 0.0])
v1 = np.array([0.0, 0.0])

p2 = np.array([10.0, 0.0])
v2 = np.array([0.0, 3.0])

# rotation
spin1 = 0.0
spin2 = 0.0
spin_v1 = 0.5
spin_v2 = 2.0

# frame: 0=inertial, 1=m1, 2=m2, 3=cm
frame_mode = 'inertial'

paused = False
trail_x = []
trail_y = []
max_trail = 400

# history for energy
hist_t = []
hist_ke = []
hist_pe = []
hist_tot = []









# physics
def get_forces(pos1, pos2, mass1, mass2):
    diff = pos2 - pos1
    r_sq = np.sum(diff**2)
    r = np.sqrt(r_sq)
    
    if r < 0.2: r = 0.2 # collision limit
    
    f_mag = g_const*mass1*mass2 / (r*r)
    unit = diff / r
    
    return f_mag * unit


def get_cm():
    return (m1*p1 + m2*p2) / (m1+m2)


def set_tangential(idx, val):
    # sets tangential speed relative to the other mass
    global v1, v2
    diff = p2 - p1
    r = np.sqrt(np.sum(diff**2))
    if r == 0: return
    
    # tangent vector (perp to diff)
    tangent = np.array([-diff[1]/r, diff[0]/r])
    
    if idx == 1:
        # projected v1 onto tangent? no, just set it
        # student logic: just overwrite v1 with tangent * val
        v1 = tangent * val
    else:
        v2 = tangent * val






# interactions
def update_m1(val): global m1; m1 = val
def update_m2(val): global m2; m2 = val
def update_s1(val): global spin_v1; spin_v1 = val
def update_s2(val): global spin_v2; spin_v2 = val
def update_sim_speed(val): global speed_factor; speed_factor = val

def update_v1_t(val): set_tangential(1, val)
def update_v2_t(val): set_tangential(2, val)

def change_frame(label):
    global frame_mode, trail_x, trail_y
    frame_mode = label
    # clear trails on frame switch
    trail_x, trail_y = [], []

def toggle_pause(event):
    global paused
    paused = not paused

def reset_lab(event):
    global p1, p2, v1, v2, trail_x, trail_y, hist_t, hist_ke, hist_pe, hist_tot
    p1, v1 = np.array([0.0, 0.0]), np.array([0.0, 0.0])
    p2, v2 = np.array([10.0, 0.0]), np.array([0.0, 3.0])
    trail_x, trail_y = [], []
    hist_t, hist_ke, hist_pe, hist_tot = [], [], [], []






# main loop
def update(frame):
    global p1, p2, v1, v2, spin1, spin2, trail_x, trail_y
    global hist_t, hist_ke, hist_pe, hist_tot

    dt = dt_base * speed_factor

    if not paused:
        # inertial physics
        f = get_forces(p1, p2, m1, m2)
        v1 = v1 + (f/m1)*dt
        v2 = v2 - (f/m2)*dt
        p1 = p1 + v1*dt
        p2 = p2 + v2*dt
        
        spin1 += spin_v1*dt
        spin2 += spin_v2*dt
        
        # energy
        v1s = np.sum(v1**2)
        v2s = np.sum(v2**2)
        ke = 0.5*m1*v1s + 0.5*m2*v2s
        r = np.sqrt(np.sum((p2-p1)**2))
        pe = -g_const*m1*m2/r
        
        hist_t.append(frame*dt_base)
        hist_ke.append(ke)
        hist_pe.append(pe)
        hist_tot.append(ke+pe)
        if len(hist_t)>150:
            for l in [hist_t, hist_ke, hist_pe, hist_tot]: l.pop(0)

    # transform for display
    cm = get_cm()
    
    if frame_mode == 'inertial':
        disp1, disp2, disp_cm = p1, p2, cm
    elif frame_mode == 'mass 1':
        disp1, disp2, disp_cm = p1-p1, p2-p1, cm-p1
    elif frame_mode == 'mass 2':
        disp1, disp2, disp_cm = p1-p2, p2-p2, cm-p2
    else: # cm
        disp1, disp2, disp_cm = p1-cm, p2-cm, cm-cm

    # graphics
    sun_obj.set_data([disp1[0]], [disp1[1]])
    planet_obj.set_data([disp2[0]], [disp2[1]])
    cm_cross.set_data([disp_cm[0]], [disp_cm[1]])
    
    # spin lines
    s1x, s1y = disp1[0]+0.6*np.cos(spin1), disp1[1]+0.6*np.sin(spin1)
    spin_l1.set_data([disp1[0], s1x], [disp1[1], s1y])
    s2x, s2y = disp2[0]+0.4*np.cos(spin2), disp2[1]+0.4*np.sin(spin2)
    spin_l2.set_data([disp2[0], s2x], [disp2[1], s2y])
    
    if not paused:
        trail_x.append(disp2[0])
        trail_y.append(disp2[1])
        if len(trail_x)>max_trail:
            trail_x.pop(0); trail_y.pop(0)
    
    trail_l.set_data(trail_x, trail_y)

    # update energy plot
    if len(hist_t)>0:
        line_k.set_data(hist_t, hist_ke)
        line_p.set_data(hist_t, hist_pe)
        line_tot.set_data(hist_t, hist_tot)
        ax_en.set_xlim(min(hist_t), max(hist_t)+0.1)
        all_e = hist_ke + hist_pe + hist_tot
        ax_en.set_ylim(min(all_e)-5, max(all_e)+5)

    # table
    r_val = np.sqrt(np.sum((p2-p1)**2))
    v1_mag = np.sqrt(np.sum(v1**2))
    v2_mag = np.sqrt(np.sum(v2**2))
    
    current_energy = hist_tot[-1] if len(hist_tot) > 0 else 0.0
    
    table_str = (f"masses: {m1:.1f}, {m2:.1f} kg\n"
                 f"distance: {r_val:.2f} m\n"
                 f"v1: {v1_mag:.2f} m/s\n"
                 f"v2: {v2_mag:.2f} m/s\n"
                 f"spin1: {spin_v1:.1f} rad/s\n"
                 f"spin2: {spin_v2:.1f} rad/s\n"
                 f"energy: {current_energy:.1f} J")
    data_text.set_text(table_str)

    return sun_obj, planet_obj, cm_cross, spin_l1, spin_l2, trail_l, line_k, line_p, line_tot





# setup
fig = plt.figure(figsize=(14, 8))
plt.subplots_adjust(bottom=0.3, right=0.75)

ax_main = fig.add_axes([0.05, 0.35, 0.65, 0.6])
ax_main.set_aspect('equal')
ax_main.set_xlim(-20, 20)
ax_main.set_ylim(-20, 20)
ax_main.grid(True, alpha=0.1)
ax_main.set_title("orbit laboratory", fontsize=14)

sun_obj, = ax_main.plot([], [], 'yo', ms=12)
planet_obj, = ax_main.plot([], [], 'bo', ms=6)
cm_cross, = ax_main.plot([], [], 'rx', ms=10, mew=2)
spin_l1, = ax_main.plot([], [], 'k-', lw=1)
spin_l2, = ax_main.plot([], [], 'k-', lw=1)
trail_l, = ax_main.plot([], [], 'b:', lw=1, alpha=0.4)

# sidebar energy
ax_en = fig.add_axes([0.78, 0.65, 0.2, 0.25])
ax_en.set_title("energy history")
line_k, = ax_en.plot([], [], 'r-', label='ke')
line_p, = ax_en.plot([], [], 'b-', label='pe')
line_tot, = ax_en.plot([], [], 'k--', label='tot')
ax_en.legend(fontsize=7)

# table
ax_tab = fig.add_axes([0.78, 0.35, 0.2, 0.25])
ax_tab.axis('off')
data_text = ax_tab.text(0, 0.9, "", va='top', fontsize=9, family='monospace',
                        bbox=dict(facecolor='white', alpha=0.5))

# controls
# left block
ax_m1 = plt.axes([0.1, 0.2, 0.15, 0.02])
s_m1 = Slider(ax_m1, 'm1 (kg)', 10, 1000, valinit=100)
s_m1.on_changed(update_m1)

ax_m2 = plt.axes([0.1, 0.16, 0.15, 0.02])
s_m2 = Slider(ax_m2, 'm2 (kg)', 1, 100, valinit=10)
s_m2.on_changed(update_m2)

ax_v1 = plt.axes([0.1, 0.12, 0.15, 0.02])
s_v1 = Slider(ax_v1, 'v1 tang (m/s)', -10, 10, valinit=0)
s_v1.on_changed(update_v1_t)

ax_v2 = plt.axes([0.1, 0.08, 0.15, 0.02])
s_v2 = Slider(ax_v2, 'v2 tang (m/s)', -10, 10, valinit=3)
s_v2.on_changed(update_v2_t)

# middle block
ax_ss = plt.axes([0.35, 0.2, 0.15, 0.02])
s_ss = Slider(ax_ss, 'sim speed', 0.1, 5.0, valinit=1.0)
s_ss.on_changed(update_sim_speed)

ax_sp1 = plt.axes([0.35, 0.16, 0.15, 0.02])
s_sp1 = Slider(ax_sp1, 'spin1 (rad/s)', 0, 10, valinit=0.5)
s_sp1.on_changed(update_s1)

ax_sp2 = plt.axes([0.35, 0.12, 0.15, 0.02])
s_sp2 = Slider(ax_sp2, 'spin2 (rad/s)', 0, 10, valinit=2.0)
s_sp2.on_changed(update_s2)

# right block
ax_pause = plt.axes([0.55, 0.15, 0.08, 0.04])
b_pause = Button(ax_pause, 'pause')
b_pause.on_clicked(toggle_pause)

ax_reset = plt.axes([0.55, 0.10, 0.08, 0.04])
b_reset = Button(ax_reset, 'reset')
b_reset.on_clicked(reset_lab)

# frame radio
ax_rad = plt.axes([0.65, 0.05, 0.1, 0.15])
ax_rad.set_facecolor('#f0f0f0')
radio = RadioButtons(ax_rad, ('inertial', 'mass 1', 'mass 2', 'cm'))
radio.on_clicked(change_frame)


ani = FuncAnimation(fig, update, interval=25)
plt.show()