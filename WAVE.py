# 1d wave simulation
# finite difference wave equation
# sandbox for reflection and interference

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons









# constants
n_points = 200
dx = 0.1
c = 2.0 
dt = 0.02 # keep this stable
steps_per_frame = 5

# string state
y = np.zeros(n_points)
y_prev = np.zeros(n_points)
y_next = np.zeros(n_points)

# settings
left_bound = 'hard' # none, soft, hard
right_bound = 'hard'
mid_barrier = False
barrier_hardness = 1.0 # 0 to 1
wave_mode = 'pulse' # pulse, osc
freq = 2.0
amp = 1.0

paused = False
time = 0.0










# physics update
def update_physics():
    global y, y_prev, y_next, time
    
    # r squared factor for wave eq
    # c*dt/dx must be <= 1 for stability
    r_sq = (c*dt/dx)**2


    # internal points
    for i in range(1, n_points-1):
        
        # standard wave eq
        y_next[i] = 2.0*y[i]-y_prev[i]+r_sq*(y[i+1]-2.0*y[i]+y[i-1])

        # middle barrier pinning
        if mid_barrier and i == n_points//2:
            # pins the point toward zero based on hardness
            y_next[i] = y_next[i] * (1.0 - barrier_hardness)


    # handle boundaries
    # left
    if wave_mode == 'osc':
        y_next[0] = amp*np.sin(2.0*np.pi*freq*time)
    else:
        if left_bound == 'hard':
            y_next[0] = 0.0
        elif left_bound == 'soft':
            y_next[0] = y_next[1]
        else: # none / open (simple absorbing)
            y_next[0] = y[1]+((c*dt-dx)/(c*dt+dx))*(y_next[1]-y[0])

    # right
    if right_bound == 'hard':
        y_next[-1] = 0.0
    elif right_bound == 'soft':
        y_next[-1] = y_next[-2]
    else: # none / open
        y_next[-1] = y[-2]+((c*dt-dx)/(c*dt+dx))*(y_next[-2]-y[-1])


    # shift arrays
    y_prev = np.copy(y)
    y = np.copy(y_next)
    time += dt










# interactions
def send_pulse(event):
    global y, y_prev
    # add a gaussian pulse in the middle-ish or near left
    x = np.linspace(0, n_points*dx, n_points)
    center = 2.0
    pulse = amp * np.exp(-(x-center)**2 / (2*0.5**2))
    y += pulse
    # shift y_prev slightly to give initial direction (right)
    # student trick: just offset center
    pulse_prev = amp * np.exp(-(x-(center-c*dt))**2 / (2*0.5**2))
    y_prev += pulse_prev

def reset_string(event):
    global y, y_prev, y_next, time
    y.fill(0)
    y_prev.fill(0)
    y_next.fill(0)
    time = 0.0

def update_f(val): global freq; freq = val
def update_a(val): global amp; amp = val
def update_bh(val): global barrier_hardness; barrier_hardness = val
def update_s(val):
    global steps_per_frame
    steps_per_frame = int(val)

def change_left(label): global left_bound; left_bound = label
def change_right(label): global right_bound; right_bound = label
def change_mode(label): global wave_mode; wave_mode = label

def toggle_barrier(event):
    global mid_barrier
    mid_barrier = not mid_barrier
    if mid_barrier:
        btn_bar.label.set_text("remove bar")
    else:
        btn_bar.label.set_text("add bar")

def toggle_pause(event):
    global paused
    paused = not paused










# animation
def animate(frame):
    if not paused:
        # loop for speed
        for _ in range(steps_per_frame):
            update_physics()
            
    line.set_ydata(y)
    
    # barrier marker
    if mid_barrier:
        bar_line.set_visible(True)
        bx = (n_points//2)*dx
        bar_line.set_data([bx, bx], [-2, 2])
    else:
        bar_line.set_visible(False)
        
    return line, bar_line










# setup plot
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.3, right=0.8)

x_vals = np.linspace(0, n_points*dx, n_points)
line, = ax.plot(x_vals, y, 'b-', lw=2)
bar_line, = ax.plot([], [], 'r--', lw=2, label='barrier')

ax.set_ylim(-2.5, 2.5)
ax.set_xlim(0, n_points*dx)
ax.grid(True, alpha=0.3)
ax.set_title("1d wave sandbox", fontsize=14)


# widgets
# sliders
ax_f = plt.axes([0.15, 0.18, 0.25, 0.03])
s_f = Slider(ax_f, 'freq (hz)', 0.1, 10.0, valinit=2.0)
s_f.on_changed(update_f)

ax_a = plt.axes([0.15, 0.13, 0.25, 0.03])
s_a = Slider(ax_a, 'amp (m)', 0.1, 2.0, valinit=1.0)
s_a.on_changed(update_a)

ax_s = plt.axes([0.15, 0.08, 0.25, 0.03])
s_s = Slider(ax_s, 'sim speed', 1, 20, valinit=5, valstep=1)
s_s.on_changed(update_s)

ax_bh = plt.axes([0.15, 0.03, 0.25, 0.03])
s_bh = Slider(ax_bh, 'bar hardness', 0.0, 1.0, valinit=1.0)
s_bh.on_changed(update_bh)

# buttons
ax_pulse = plt.axes([0.45, 0.15, 0.1, 0.05])
btn_pulse = Button(ax_pulse, "pulse")
btn_pulse.on_clicked(send_pulse)

ax_reset = plt.axes([0.45, 0.08, 0.1, 0.05])
btn_reset = Button(ax_reset, "reset")
btn_reset.on_clicked(reset_string)

ax_bar = plt.axes([0.57, 0.15, 0.1, 0.05])
btn_bar = Button(ax_bar, "add bar")
btn_bar.on_clicked(toggle_barrier)

ax_pause = plt.axes([0.57, 0.08, 0.1, 0.05])
btn_pause = Button(ax_pause, "pause")
btn_pause.on_clicked(toggle_pause)

# sidebar radio buttons
# left bound
ax_rl = plt.axes([0.82, 0.65, 0.12, 0.2])
ax_rl.set_facecolor('#f0f0f0')
ax_rl.set_title("left bound", fontsize=10)
radio_l = RadioButtons(ax_rl, ('hard', 'soft', 'none'))
radio_l.on_clicked(change_left)

# right bound
ax_rr = plt.axes([0.82, 0.4, 0.12, 0.2])
ax_rr.set_facecolor('#f0f0f0')
ax_rr.set_title("right bound", fontsize=10)
radio_r = RadioButtons(ax_rr, ('hard', 'soft', 'none'))
radio_r.on_clicked(change_right)

# mode
ax_rm = plt.axes([0.82, 0.15, 0.12, 0.2])
ax_rm.set_facecolor('#f0f0f0')
ax_rm.set_title("source", fontsize=10)
radio_m = RadioButtons(ax_rm, ('pulse', 'osc'))
radio_m.on_clicked(change_mode)


ani = FuncAnimation(fig, animate, interval=30)
plt.show()
