# pendulum simulation
# simple pendulum with damping
# user can drag the bob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Arc







# constants
g = 9.81
l = 2.0
damping = 0.05
dt = 0.01

mass = 1.0
cross_section = 0.01
air_density = 1.225
drag_coefficient = 0.47

# starting values
theta = 0.0
omega = 0.0
anchor = np.array([0.0, 0.0])

paused = False
dragging = False
speed_factor = 1.0

sim_time = 0.0
last_theta = theta
last_zero_crossing_time = None
period_estimate = np.nan







# gets bob position
def get_bob_pos(theta_val):
    x = anchor[0]+l*np.sin(theta_val)
    y = anchor[1]- l * np.cos(theta_val)


    return np.array([x, y])

# draws the string with some sag
def compute_string_points(anchor, bob_pos, segments=100, sag=0.1):
    t = np.linspace(0, 1, segments)
    x = anchor[0] + (bob_pos[0] - anchor[0]) * t
    y = anchor[1] + (bob_pos[1] - anchor[1]) * t - sag * np.sin(np.pi * t)


    return x, y

# calculates period
def compute_damped_period():
    omega0 = np.sqrt(g / l)


    if damping/2 < omega0:
        omega_d = np.sqrt(omega0**2-(damping/2)**2)
        t_d = 2*np.pi/omega_d

    else:
        t_d = np.nan


    return t_d






# mouse clicking
def on_press(event):
    global dragging


    if event.inaxes == ax:
        dragging = True

def on_motion(event):
    global theta, omega, dragging


    if dragging and event.xdata is not None and event.ydata is not None:
        dx = event.xdata - anchor[0]
        dy = event.ydata - anchor[1]


        if dx == 0 and dy == 0:
            return
        
        theta = np.arctan2(dx, -dy)
        omega = 0.0

def on_release(event):
    global dragging
    dragging = False





# slider and button functions
def update_speed(val):
    global speed_factor
    speed_factor = speed_slider.val

def toggle_pause(event):
    global paused
    paused = not paused


    if paused:
        pause_button.label.set_text("resume")

    else:
        pause_button.label.set_text("pause")


    plt.draw()

def manual_step(dt_step):
    global theta, omega
    alpha = -(g/l)*np.sin(theta)-damping*omega
    omega += alpha*dt_step*speed_factor
    theta += omega * dt_step*speed_factor

def on_key(event):

    if event.key == ' ':
        toggle_pause(event)


    elif paused:

        if event.key == 'left':
            manual_step(-dt)
            update(0)
            plt.draw()

        elif event.key == 'right':
            manual_step(dt)
            update(0)
            plt.draw()





# update loop
def update(frame):
    global theta, omega, bob_pos, dt, speed_factor, paused, dragging
    global sim_time, last_theta, last_zero_crossing_time, period_estimate


    if not paused and not dragging:
        sim_time += dt * speed_factor


    if not paused and not dragging:
        alpha = -(g/l)*np.sin(theta)-damping*omega
        omega += alpha*dt*speed_factor
        theta += omega * dt*speed_factor


    if not paused and not dragging:

        if last_theta * theta < 0:
            if last_zero_crossing_time is None:
                last_zero_crossing_time = sim_time

            else:
                half_period = sim_time-last_zero_crossing_time
                period_estimate = 2*half_period
                last_zero_crossing_time = sim_time
        
        last_theta = theta


    bob_pos = get_bob_pos(theta)
    x_string, y_string = compute_string_points(anchor, bob_pos, sag=0.1)
    
    string_line.set_data(x_string, y_string)
    bob_marker.set_data([bob_pos[0]], [bob_pos[1]])
    
    horizontal_line.set_data([anchor[0], bob_pos[0]], [bob_pos[1], bob_pos[1]])
    vertical_line.set_data([anchor[0], anchor[0]], [anchor[1], bob_pos[1]])
    
    horiz_dist = abs(bob_pos[0] - anchor[0])
    vert_dist = abs(bob_pos[1] - anchor[1])
    
    horiz_text.set_position(((anchor[0] + bob_pos[0]) / 2, bob_pos[1] + 0.05))
    horiz_text.set_text(f"{horiz_dist:.2f} m")
    
    vertical_text.set_position((anchor[0] - 0.15, (anchor[1] + bob_pos[1]) / 2))
    vertical_text.set_text(f"{vert_dist:.2f} m")
    
    theta_deg = theta*180/np.pi
    angle_measured = abs(theta_deg)
    

    if theta >= 0:
        arc_patch.theta1 = 270
        arc_patch.theta2 = 270 + angle_measured
        mid_angle_deg = 270+angle_measured/2

    else:
        arc_patch.theta1 = 270 - angle_measured
        arc_patch.theta2 = 270
        mid_angle_deg = 270-angle_measured/2
        

    mid_angle = np.deg2rad(mid_angle_deg)
    arc_label_radius = arc_radius+0.1
    arc_label_x = anchor[0] + arc_label_radius * np.cos(mid_angle)
    arc_label_y = anchor[1] + arc_label_radius * np.sin(mid_angle)
    
    arc_text.set_position((arc_label_x, arc_label_y))
    arc_text.set_text(f"{angle_measured:.1f} deg")
    
    freq = 1 / period_estimate if period_estimate and period_estimate > 0 else 0
    meas_str = ("measurements:\n"
                f"horizontal: {horiz_dist:.2f} m\n"
                f"vertical: {vert_dist:.2f} m\n"
                f"angle: {angle_measured:.1f} deg\n"
                f"period: {period_estimate:.2f} s\n"
                f"freq: {freq:.2f} hz")
    meas_text.set_text(meas_str)
    
    return (
            string_line, bob_marker, horizontal_line, vertical_line,
            horiz_text, vertical_text, arc_patch, arc_text, meas_text)





# plotting setup
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.30, left=0.25, right=0.75)
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.8, 1.0)
ax.grid(linestyle=':', color='gray', alpha=0.7)
ax.set_title("pendulum simulation", fontsize=14)

anchor_marker, = ax.plot(anchor[0], anchor[1], 'ko', markersize=8)
string_line, = ax.plot([], [], lw=2, color='saddlebrown', linestyle='--')
bob_marker, = ax.plot([], [], 'o', color='firebrick', markersize=15)
horizontal_line, = ax.plot([], [], lw=1.5, color='dodgerblue', linestyle='--')
vertical_line, = ax.plot([], [], lw=1.5, color='seagreen', linestyle='--')

horiz_text = ax.text(0, 0, "", color='dodgerblue', fontsize=10, ha='center', va='bottom')
vertical_text = ax.text(0, 0, "", color='seagreen', fontsize=10, ha='right', va='center')


ax_speed = plt.axes([0.20, 0.10, 0.60, 0.03])
speed_slider = Slider(ax_speed, 'speed', 0.1, 3.0, valinit=1.0, valstep=0.1, facecolor='skyblue')
speed_slider.on_changed(update_speed)


ax_pause = plt.axes([0.45, 0.02, 0.10, 0.04])
pause_button = Button(ax_pause, 'pause', color='lightcoral', hovercolor='salmon')
pause_button.on_clicked(toggle_pause)


arc_radius = 0.4 * l
arc_patch = Arc(anchor, width=2*arc_radius, height=2*arc_radius,
                angle=0, theta1=270, theta2=270, color='magenta', lw=2)
ax.add_patch(arc_patch)
arc_text = ax.text(anchor[0] + arc_radius, anchor[1] - 0.1, "",
                   color='magenta', fontsize=10, ha='center', va='bottom')


ax_cond = fig.add_axes([0.78, 0.55, 0.20, 0.35])
ax_cond.axis('off')
ax_cond.set_facecolor('#f0f0f0')
cond_str = ("conditions:\n"
            f"mass: {mass:.2f} kg\n"
            f"length: {l:.2f} m\n"
            f"g: {g:.2f} m/s2\n"
            f"cross section: {cross_section:.3f} m2\n"
            f"air density: {air_density:.3f} kg/m3\n"
            f"drag coeff: {drag_coefficient:.2f}")
cond_text = ax_cond.text(0.05, 0.95, cond_str, va='top', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


ax_meas = fig.add_axes([0.78, 0.10, 0.20, 0.35])
ax_meas.axis('off')
ax_meas.set_facecolor('#f0f0f0')
meas_text = ax_meas.text(0.05, 0.95, "", va='top', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event', on_key)

bob_pos = get_bob_pos(theta)
sim_time = 0.0
last_theta = theta
last_zero_crossing_time = None
period_estimate = np.nan

ani = FuncAnimation(fig, update, frames=600, interval=20, blit=False)
plt.show()
