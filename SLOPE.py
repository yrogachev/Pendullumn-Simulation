# linear slope simulation
# compare rolling vs sliding on a straight ramp
# center of mass is offset correctly

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle, Rectangle, Wedge









# constants
g = 9.81
dt = 0.02

# ramp geometry
slope_angle_deg = 30.0
ramp_length = 15.0
x_start, y_start = 2.0, 8.0

# physics state
# each racer: {type, s, v, color, marker}
racers = []
mu_k = 0.0
racing = False
time_elapsed = 0.0
show_forces = False

# types
curr_obj = 'block'










# types
curr_obj = 'block'
active_slopes = {'Linear'} # set of active slope types

# pre-calculated paths
# dict: type -> {x, y, s, theta}
paths_data = {}

# graphics
ramp_lines = {} # type -> line object










# build the ramp line
def build_path_data():
    global paths_data
    
    # End point based on current angle/length
    rad = np.deg2rad(slope_angle_deg)
    x_end = x_start + ramp_length*np.cos(rad)
    y_end = y_start - ramp_length*np.sin(rad)
    
    n_pts = 1000
    
    for s_type in ['Linear', 'Parabola', 'Brachistochrone', 'Circle']:
        # calc path arrays
        px, py = [], []
        
        if s_type == 'Linear':
            px = np.linspace(x_start, x_end, n_pts)
            py = np.linspace(y_start, y_end, n_pts)
            
        elif s_type == 'Parabola':
            k = (y_start - y_end) / (x_start - x_end)**2
            px = np.linspace(x_start, x_end, n_pts)
            py = y_end + k * (px - x_end)**2
            
        elif s_type == 'Brachistochrone':
            dx = x_end - x_start
            dy = y_start - y_end 
            t = np.linspace(0, np.pi, n_pts)
            cx = t - np.sin(t)
            cy = 1 - np.cos(t)
            px = x_start + cx * (dx / cx[-1])
            py = y_start - cy * (dy / cy[-1])
            
        elif s_type == 'Circle':
            R_x = x_end - x_start
            R_y = y_start - y_end
            ang = np.linspace(np.pi, 1.5*np.pi, n_pts)
            px = x_end + R_x * np.cos(ang)
            py = y_start + R_y * np.sin(ang)
            
        # calc s and theta
        ps = np.zeros(n_pts)
        pt = np.zeros(n_pts)
        for i in range(1, n_pts):
            dx = px[i] - px[i-1]
            dy = py[i] - py[i-1]
            dist = np.sqrt(dx**2 + dy**2)
            ps[i] = ps[i-1] + dist
            pt[i] = np.arctan2(-dy, dx)
        pt[0] = pt[1]
        
        paths_data[s_type] = {'x': px, 'y': py, 's': ps, 'theta': pt}
        
    update_ramp_lines()


def update_ramp_lines():
    if 'ax' not in globals(): return
    
    # clear old lines if needed or just toggle visibility
    # easier to just set data if active, or empty if not
    
    # colors
    colors = {'Linear':'black', 'Parabola':'blue', 'Brachistochrone':'red', 'Circle':'green'}
    
    for s_type in ['Linear', 'Parabola', 'Brachistochrone', 'Circle']:
        if s_type not in ramp_lines:
            # create line
            l, = ax.plot([], [], '-', lw=2, color=colors[s_type], label=s_type)
            ramp_lines[s_type] = l
            
        if s_type in active_slopes:
            d = paths_data[s_type]
            ramp_lines[s_type].set_data(d['x'], d['y'])
        else:
            ramp_lines[s_type].set_data([], [])
            
    ax.legend(loc='upper right', fontsize='small')


def get_ramp_coords():
    # legacy
    return [], []


# physics math
def get_beta(t):
    if t == 'block': return 0.0
    if t == 'sphere': return 0.4
    if t == 'cylinder': return 0.5
    if t == 'ring': return 1.0
    return 0.0


def get_pos(s_val, path_type):
    if path_type not in paths_data: return x_start, y_start
    d = paths_data[path_type]
    
    # interpolate s
    if len(d['s']) == 0: return x_start, y_start
    
    idx = np.searchsorted(d['s'], s_val)
    if idx >= len(d['x']): idx = -1
    
    return d['x'][idx], d['y'][idx]

def get_angle_at_s(s_val, path_type):
    if path_type not in paths_data: return 0.0
    d = paths_data[path_type]
    
    if len(d['s']) == 0: return 0.0
    idx = np.searchsorted(d['s'], s_val)
    if idx >= len(d['theta']): idx = -1
    return d['theta'][idx]










# interactions
def add_racer(event):
    if racing: return
    
    # add one racer for EACH active slope type
    for s_type in active_slopes:
        r = {
            'type': curr_obj,
            'slope_type': s_type,
            's': 0.0,
            'v': 0.0,
            'theta_rot': 0.0, 
            'color': {'block':'blue', 'sphere':'red', 'cylinder':'green', 'ring':'purple'}[curr_obj]
        }
        racers.append(r)
        
        # create plot object
        x, y = get_pos(0.0, s_type)
        rad_offset = 0.5
        
        # initial angle
        theta = get_angle_at_s(0.0, s_type)
        
        # normal vector
        nx, ny = np.sin(theta), np.cos(theta)
        
        cx = x + nx*rad_offset
        cy = y + ny*rad_offset
        
        if r['type'] == 'block':
            w, h = 1.0, 1.0
            dx = -0.5 * w * np.cos(theta)
            dy = -0.5 * w * (-np.sin(theta))
            
            rect = Rectangle((x+dx, y+dy), w, h, angle=-np.rad2deg(theta), color=r['color'], alpha=0.7)
            ax.add_patch(rect)
            r['patch'] = rect
            r['marker_line'] = None 
            
        else:
            circ = Circle((cx, cy), radius=rad_offset, color=r['color'], alpha=0.7)
            ax.add_patch(circ)
            r['patch'] = circ
            
            ln, = ax.plot([cx, cx], [cy, cy+rad_offset], 'w-', lw=2)
            r['marker_line'] = ln


def start_race(event):
    global racing, time_elapsed
    if not racers: return
    racing = True
    time_elapsed = 0.0


def reset_lab(event):
    global racing, racers, time_elapsed
    racing = False
    time_elapsed = 0.0
    for r in racers:
        r['patch'].remove()
        if r['marker_line']:
            r['marker_line'].remove()
            
    racers[:] = []
    # ax.legend().remove() # patches dont auto legend nicely here
    time_text.set_text("time: 0.00 s")


def update_angle(val):
    global slope_angle_deg
    slope_angle_deg = val
    build_path_data()
    
    # reset racers if angle changes? 
    # student would just let them float in air, but let's reset
    if not racing:
        for r in racers:
            x, y = get_pos(r['s'])
            # update visuals brute force
            pass
            
    if not racing and len(racers)>0:
        reset_lab(None)


def update_mu(val):
    global mu_k
    mu_k = val


def change_obj(label):
    global curr_obj
    curr_obj = label

def change_slope_type(label):
    global active_slopes
    # toggle
    if label in active_slopes:
        active_slopes.remove(label)
    else:
        active_slopes.add(label)
        
    update_ramp_lines()
    # reset not needed if we just toggle visibility? 
    # but physics might need reset if we change geometry logic
    # simplest: do nothing to physics, just redraw lines.
    # racers are already tied to specific slopes.


def toggle_forces(event):
    global show_forces
    show_forces = not show_forces










# vectors
force_plots = [] # list of lists

def update(frame):
    global time_elapsed, force_plots
    
    # clear old vectors
    for sublist in force_plots:
        for p in sublist: p.remove()
    force_plots[:] = []
    
    off = 0.5 # radius
    
    if racing:
        for r in racers:
            s_type = r['slope_type']
            # get path data
            # check if path data exists
            if s_type not in paths_data: continue
            
            p_data = paths_data[s_type]
            max_s = p_data['s'][-1]
            beta = get_beta(r['type'])
            
            if r['s'] < max_s:
                # physics
                theta = get_angle_at_s(r['s'], s_type)
                
                a_grav = g*np.sin(theta)
                
                if r['type'] == 'block':
                    # sliding friction
                    acc = a_grav - mu_k*g*np.cos(theta)
                else:
                    # rolling (assume no slip)
                    acc = a_grav / (1.0+beta)
                
                # if acc < 0 it stays still
                if r['v'] == 0 and acc < 0: acc = 0
                
                r['v'] += acc*dt
                r['s'] += r['v']*dt
                
                d_s = r['v']*dt
                if r['type'] != 'block':
                    r['theta_rot'] += d_s / off
                
                if r['s'] > max_s: r['s'] = max_s
                
            # update visuals
            base_x, base_y = get_pos(r['s'], s_type)
            theta = get_angle_at_s(r['s'], s_type)
            
            # center pos
            cx = base_x + off*np.sin(theta)
            cy = base_y + off*np.cos(theta)
            
            if r['type'] == 'block':
                # update Rectangle xy (bottom left corner)
                w = 1.0
                # corner offset along slope
                dx = -0.5 * w * np.cos(theta)
                dy = -0.5 * w * (-np.sin(theta))
                
                r['patch'].set_xy((base_x+dx, base_y+dy))
                r['patch'].angle = -np.rad2deg(theta)
                
                rx, ry = cx, cy # center for vectors
            else:
                # update Circle center
                r['patch'].center = (cx, cy)
                
                # update marker line for rotation
                
                rot = r['theta_rot']
                # tip pos
                tx = cx + off*np.sin(rot)
                ty = cy + off*np.cos(rot)
                
                r['marker_line'].set_data([cx, tx], [cy, ty])
                
                rx, ry = cx, cy

            
            if show_forces:
                # mg (green)
                sc = 0.3
                l1, = ax.plot([rx, rx], [ry, ry-g*sc], 'g-', lw=1)
                # normal (blue)
                nx, ny = np.sin(theta), np.cos(theta)
                n_mag = g*np.cos(theta)
                l2, = ax.plot([rx, rx+nx*n_mag*sc], [ry, ry+ny*n_mag*sc], 'b-', lw=1)
                # friction (red)
                fx, fy = -np.cos(theta), np.sin(theta)
                if r['type'] == 'block': f_mag = mu_k*n_mag
                else: f_mag = (beta/(1+beta))*g*np.sin(theta)
                l3, = ax.plot([rx, rx+fx*f_mag*sc], [ry, ry+fy*f_mag*sc], 'r-', lw=1)
                force_plots.append([l1, l2, l3])

        time_elapsed += dt
        time_text.set_text(f"time: {time_elapsed:.2f} s")

    all_artists = list(ramp_lines.values()) + [time_text]
    for r in racers:
        all_artists.append(r['patch'])
        if r['marker_line']:
            all_artists.append(r['marker_line'])
    
    return all_artists # patches update themselves via internal draw list










# plot setup
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.3, left=0.1)

ax.set_xlim(0, 25)
ax.set_ylim(-2, 12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.set_title("slope physics lab", fontsize=14)

build_path_data() 

time_text = ax.text(1, 11, "time: 0.00 s", fontsize=12)


# widgets
plt.subplots_adjust(bottom=0.25, left=0.1, right=0.8)

ax_angle = plt.axes([0.15, 0.12, 0.3, 0.03])
s_angle = Slider(ax_angle, 'angle', 0, 60, valinit=30)
s_angle.on_changed(update_angle)

ax_mu = plt.axes([0.15, 0.07, 0.3, 0.03])
s_mu = Slider(ax_mu, 'mu_k', 0, 1, valinit=0)
s_mu.on_changed(update_mu)

ax_reset = plt.axes([0.1, 0.01, 0.08, 0.04])
b_reset = Button(ax_reset, 'reset')
b_reset.on_clicked(reset_lab)

ax_add = plt.axes([0.2, 0.01, 0.08, 0.04])
b_add = Button(ax_add, 'add racer')
b_add.on_clicked(add_racer)

ax_start = plt.axes([0.3, 0.01, 0.08, 0.04])
b_start = Button(ax_start, 'race!')
b_start.on_clicked(start_race)

ax_force = plt.axes([0.4, 0.01, 0.08, 0.04])
b_force = Button(ax_force, 'forces')
b_force.on_clicked(toggle_forces)

# Sidebar on right
# Objects
ax_radio = plt.axes([0.82, 0.5, 0.15, 0.25])
ax_radio.set_facecolor('#f0f0f0')
ax_radio.set_title("Object Type", fontsize=10)
radio = RadioButtons(ax_radio, ('block', 'sphere', 'cylinder', 'ring'))
radio.on_clicked(change_obj)

# Slopes
from matplotlib.widgets import CheckButtons
ax_radio_s = plt.axes([0.82, 0.15, 0.15, 0.25])
ax_radio_s.set_facecolor('#f0f0f0')
ax_radio_s.set_title("Active Slopes", fontsize=10)
radio_s = CheckButtons(ax_radio_s, ('Linear', 'Parabola', 'Brachistochrone', 'Circle'), [True, False, False, False])
radio_s.on_clicked(change_slope_type)


ani = FuncAnimation(fig, update, interval=20)
plt.show()