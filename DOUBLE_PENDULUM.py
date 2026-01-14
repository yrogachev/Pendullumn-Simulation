# double pendulum simulation
# chaotic motion
# drag bobs to set angles

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button







# constants
g = 9.81
l1 = 1.5
l2 = 1.5
m1 = 1.0
m2 = 1.0
damping = 0.0
dt = 0.01

# starting state
t1 = np.pi / 2
t2 = np.pi / 2
w1 = 0.0
w2 = 0.0

anchor = np.array([0.0, 0.0])

paused = False
dragging_bob1 = False
dragging_bob2 = False
speed_factor = 1.0

# trail for the second bob
trail_x = []
trail_y = []
max_trail = 150







# positions
def get_pos(t1_val, t2_val):
    x1 = anchor[0] + l1*np.sin(t1_val)
    y1 = anchor[1] - l1*np.cos(t1_val)
    
    x2 = x1 + l2*np.sin(t2_val)
    y2 = y1 - l2*np.cos(t2_val)


    return np.array([x1, y1]), np.array([x2, y2])


# equations of motion
def compute_accels(t1, t2, w1, w2):
    
    # big messy numerator for theta1
    num1 = -g*(2*m1+m2)*np.sin(t1) - m2*g*np.sin(t1-2*t2)
    num2 = -2*np.sin(t1-t2)*m2
    num3 = w2**2*l2 + w1**2*l1*np.cos(t1-t2)
    
    den = l1*(2*m1+m2-m2*np.cos(2*t1-2*t2))
    
    a1 = (num1 + num2*num3) / den


    # numerator for theta2
    num4 = 2*np.sin(t1-t2)
    num5 = w1**2*l1*(m1+m2) + g*(m1+m2)*np.cos(t1) + w2**2*l2*m2*np.cos(t1-t2)
    
    den2 = l2*(2*m1+m2-m2*np.cos(2*t1-2*t2))
    
    a2 = (num4 * num5) / den2


    return a1, a2


# calculate energy to check conservation
def get_energy(t1, t2, w1, w2):
    pe = -(m1+m2)*g*l1*np.cos(t1) - m2*g*l2*np.cos(t2)
    ke = 0.5*m1*(l1*w1)**2 + 0.5*m2*((l1*w1)**2 + (l2*w2)**2 + 2*l1*l2*w1*w2*np.cos(t1-t2))


    return pe + ke






# mouse interaction
def on_press(event):
    global dragging_bob1, dragging_bob2, t1, t2, w1, w2
    
    if event.inaxes != ax:
        return
        
    pos1, pos2 = get_pos(t1, t2)
    
    # check distance to bobs
    d1 = np.sqrt((event.xdata-pos1[0])**2 + (event.ydata-pos1[1])**2)
    d2 = np.sqrt((event.xdata-pos2[0])**2 + (event.ydata-pos2[1])**2)
    
    if d1 < 0.3:
        dragging_bob1 = True
        w1 = 0.0
        w2 = 0.0
        
    elif d2 < 0.3:
        dragging_bob2 = True
        w1 = 0.0
        w2 = 0.0


def on_motion(event):
    global t1, t2
    
    if event.inaxes != ax:
        return


    if dragging_bob1:
        dx = event.xdata - anchor[0]
        dy = event.ydata - anchor[1]
        t1 = np.arctan2(dx, -dy)
        
    elif dragging_bob2:
        # relative to bob 1
        pos1, _ = get_pos(t1, t2)
        dx = event.xdata - pos1[0]
        dy = event.ydata - pos1[1]
        t2 = np.arctan2(dx, -dy)

def on_release(event):
    global dragging_bob1, dragging_bob2
    dragging_bob1 = False
    dragging_bob2 = False






# controls
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
    
def clear_trail(event):
    global trail_x, trail_y
    trail_x = []
    trail_y = []






# main loop
def update(frame):
    global t1, t2, w1, w2, trail_x, trail_y


    if not paused and not dragging_bob1 and not dragging_bob2:
        
        # simple integration step
        a1, a2 = compute_accels(t1, t2, w1, w2)
        
        w1 += a1*dt*speed_factor
        w2 += a2*dt*speed_factor
        
        t1 += w1*dt*speed_factor
        t2 += w2*dt*speed_factor
        
        # friction
        w1 *= 0.999
        w2 *= 0.999


    pos1, pos2 = get_pos(t1, t2)
    
    # update drawings
    rod1.set_data([anchor[0], pos1[0]], [anchor[1], pos1[1]])
    rod2.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
    
    bob1.set_data([pos1[0]], [pos1[1]])
    bob2.set_data([pos2[0]], [pos2[1]])
    

    if not paused:
        trail_x.append(pos2[0])
        trail_y.append(pos2[1])
        if len(trail_x) > max_trail:
            trail_x.pop(0)
            trail_y.pop(0)
            
    trail_line.set_data(trail_x, trail_y)
    
    # measure stuff
    e_total = get_energy(t1, t2, w1, w2)
    t1_deg = (t1*180/np.pi) % 360
    t2_deg = (t2*180/np.pi) % 360
    
    if t1_deg > 180:
        t1_deg -= 360
    if t2_deg > 180:
        t2_deg -= 360

    meas_str = ("measurements:\n"
                f"theta1: {t1_deg:.1f} deg\n"
                f"theta2: {t2_deg:.1f} deg\n"
                f"omega1: {w1:.2f} rad/s\n"
                f"omega2: {w2:.2f} rad/s\n"
                f"energy: {e_total:.2f} J")
    
    meas_text.set_text(meas_str)


    return rod1, rod2, bob1, bob2, trail_line, meas_text






# plotting setup
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25, left=0.1, right=0.7)

ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 2)
ax.grid(alpha=0.3)
ax.set_title("double pendulum", fontsize=14)

# graphics objects
rod1, = ax.plot([], [], lw=2, color='black')
rod2, = ax.plot([], [], lw=2, color='black')
bob1, = ax.plot([], [], 'o', markersize=12, color='blue')
bob2, = ax.plot([], [], 'o', markersize=12, color='red')
trail_line, = ax.plot([], [], '-', lw=1, color='gray', alpha=0.5)

# panel for data
ax_meas = fig.add_axes([0.75, 0.4, 0.2, 0.4])
ax_meas.axis('off')
ax_meas.set_facecolor('#f0f0f0')

meas_text = ax_meas.text(0.05, 0.95, "", va='top', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


# widgets
ax_speed = plt.axes([0.20, 0.10, 0.50, 0.03])
speed_slider = Slider(ax_speed, 'speed', 0.1, 5.0, valinit=1.0, facecolor='lightblue')
speed_slider.on_changed(update_speed)

ax_pause = plt.axes([0.75, 0.08, 0.15, 0.05])
pause_button = Button(ax_pause, 'pause', color='0.85', hovercolor='0.95')
pause_button.on_clicked(toggle_pause)

ax_clear = plt.axes([0.75, 0.02, 0.15, 0.05])
clear_button = Button(ax_clear, 'clear trail', color='0.85', hovercolor='0.95')
clear_button.on_clicked(clear_trail)


fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)


# init
pos1, pos2 = get_pos(t1, t2)
rod1.set_data([anchor[0], pos1[0]], [anchor[1], pos1[1]])
rod2.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
bob1.set_data([pos1[0]], [pos1[1]])
bob2.set_data([pos2[0]], [pos2[1]])

ani = FuncAnimation(fig, update, frames=200, interval=20, blit=False)
plt.show()
