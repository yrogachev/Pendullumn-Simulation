# projectile motion sim
# cannon with drag options
# vectors included

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons









# physics constants
g = 9.81
dt = 0.05
mass = 1.0


# drag coeffs
b_linear = 0.5
c_quad = 0.1

# state variables
x = 0.0
y = 0.0
vx = 0.0
vy = 0.0
t = 0.0

is_firing = False
trail_x = []
trail_y = []

# energy tracking
energy_t = []
energy_k = []
energy_p = []
energy_tot = []

show_energy = False

# user settings
init_height = 0.0
init_speed = 20.0
init_angle = 45.0
drag_mode = 'none' # none, lin, quad










# physics equations
def get_accel(vx_val, vy_val):
    ax = 0.0
    ay = -g
    
    v_sq = vx_val**2+vy_val**2
    v_mag = np.sqrt(v_sq)


    if drag_mode == 'linear':
        # f = -b*v
        ax = -(b_linear/mass)*vx_val
        ay = -g-(b_linear/mass)*vy_val
        
    elif drag_mode == 'quadratic':
        # f = -c*v^2
        if v_mag > 0:
            drag_f = c_quad*v_sq
            
            # components
            ax = -(drag_f/mass)*(vx_val/v_mag)
            ay = -g-(drag_f/mass)*(vy_val/v_mag)


    return ax, ay










# firing logic
def fire_cannon(event):
    global is_firing, x, y, vx, vy, t, trail_x, trail_y
    
    # reset pos
    x = 0.0
    y = init_height
    t = 0.0
    
    # calc init velocity
    rad = init_angle*np.pi/180.0
    vx = init_speed*np.cos(rad)
    vy = init_speed*np.sin(rad)
    
    trail_x = [x]
    trail_y = [y]
    
    # reset energy
    energy_t[:] = []
    energy_k[:] = []
    energy_p[:] = []
    energy_tot[:] = []
    
    is_firing = True
    
    # reset button text
    fire_button.label.set_text("reset")


def stop_reset(event):
    global is_firing, x, y, trail_x, trail_y
    
    if is_firing:
        # if currently flying, just stop/reset
        is_firing = False
        fire_button.label.set_text("fire")
    else:
        # fire logic is handled by the same button? 
        # actually let's make it dual purpose
        fire_cannon(event)

# wrapper to decide what button does
def on_button_click(event):
    global is_firing
    if is_firing:
        # stop it
        is_firing = False
        fire_button.label.set_text("fire")
    else:
        fire_cannon(event)










# sliders
def update_h(val):
    global init_height, y
    init_height = h_slider.val
    if not is_firing:
        y = init_height
        cannon_body.set_y(y)
        
def update_v(val):
    global init_speed
    init_speed = v_slider.val
    
def update_a(val):
    global init_angle
    init_angle = a_slider.val

def change_drag(label):
    global drag_mode
    if label == 'no drag':
        drag_mode = 'none'
    elif label == 'linear':
        drag_mode = 'linear'
    elif label == 'quadratic':
        drag_mode = 'quadratic'


def toggle_energy(event):
    global show_energy
    show_energy = not show_energy
    
    if show_energy:
        ax_energy.set_visible(True)
        energy_button.label.set_text("hide E")
    else:
        ax_energy.set_visible(False)
        energy_button.label.set_text("show E")










# main loop
def update(frame):
    global x, y, vx, vy, t, is_firing, trail_x, trail_y


    if is_firing:
        ax_val, ay_val = get_accel(vx, vy)
        
        # update physics
        vx += ax_val*dt
        vy += ay_val*dt
        
        x += vx*dt
        y += vy*dt
        t += dt
        
        # ground check
        if y < 0:
            y = 0
            is_firing = False
            fire_button.label.set_text("fire")
            
        trail_x.append(x)
        trail_y.append(y)
        
        # energy calc
        v_sq = vx**2+vy**2
        ke = 0.5*mass*v_sq
        pe = mass*g*y
        tot = ke+pe
        
        energy_t.append(t)
        energy_k.append(ke)
        energy_p.append(pe)
        energy_tot.append(tot)


    # visual updates
    projectile.set_data([x], [y])
    trail_line.set_data(trail_x, trail_y)
    
    # update energy plot if on
    if show_energy and len(energy_t) > 0:
        line_k.set_data(energy_t, energy_k)
        line_p.set_data(energy_t, energy_p)
        line_tot.set_data(energy_t, energy_tot)
        
        ax_energy.set_xlim(0, max(t, 1))
        max_e = max(max(energy_tot) if len(energy_tot)>0 else 1, 100)
        ax_energy.set_ylim(0, max_e * 1.1)

    
    # draw cannon pos (if not firing, it sits at start)
    if not is_firing and len(trail_x) < 2:
        cannon_body.set_y(init_height)
        cannon_body.set_x(0)


    # dashed lines for measurement
    h_line.set_data([0, x], [y, y])
    v_line.set_data([x, x], [0, y])
    
    h_text.set_position((x/2, y + 0.5))
    h_text.set_text(f"x: {x:.1f} m")
    
    v_text.set_position((x + 0.5, y/2))
    v_text.set_text(f"y: {y:.1f} m")
    
    
    # VECTORS
    # momentum (green)
    mom_scale = 0.05
    vec_mom.set_offsets([x, y])
    vec_mom.set_UVC(vx*mass, vy*mass)

    # gravity (blue) - constant
    grav_scale = 0.5
    vec_grav.set_offsets([x, y])
    vec_grav.set_UVC(0, -mass*g)
    
    # drag (red) - opposes velocity
    drag_ax, drag_ay = get_accel(vx, vy)
    # subtract gravity to get just drag force
    drag_fx = (drag_ax * mass)
    drag_fy = (drag_ay - (-g)) * mass 
    
    # force is purely drag part
    # f_net = f_g + f_drag => f_drag = f_net - f_g
    # ma = mg + f_drag => f_drag = m(a - g) ? 
    # wait a_y includes -g. 
    # acceleration from drag is (ay + g)
    
    d_fx = mass * drag_ax
    d_fy = mass * (drag_ay - (-g)) if is_firing else 0
    
    vec_drag.set_offsets([x, y])
    vec_drag.set_UVC(d_fx, d_fy)


    return projectile, trail_line, h_line, v_line, h_text, v_text, vec_mom, vec_grav, vec_drag, line_k, line_p, line_tot










# plot setup
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35, left=0.25, right=0.70)

ax.set_xlim(-5, 100)
ax.set_ylim(-5, 60)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.set_title("cannon simulation", fontsize=14)

# ground
ax.axhline(0, color='black', lw=2)

# cannon box
cannon_body = plt.Rectangle((-2, 0), 4, 2, color='black')
ax.add_patch(cannon_body)

# projectile things
projectile, = ax.plot([], [], 'ko', markersize=8)
trail_line, = ax.plot([], [], 'k--', lw=1, alpha=0.5)

# measure lines
h_line, = ax.plot([], [], 'b:', lw=1.5)
v_line, = ax.plot([], [], 'r:', lw=1.5)
h_text = ax.text(0, 0, "", color='blue', fontsize=10)
v_text = ax.text(0, 0, "", color='red', fontsize=10)

# vectors (quiver)
# q = ax.quiver(X, Y, U, V, ...)
vec_mom = ax.quiver(0, 0, 0, 0, color='green', scale=500, width=0.005, label='momentum')
vec_grav = ax.quiver(0, 0, 0, 0, color='blue', scale=50, width=0.005, label='gravity')
vec_drag = ax.quiver(0, 0, 0, 0, color='red', scale=50, width=0.005, label='drag')

ax.legend(loc='upper right')


# energy plot (hidden by default)
ax_energy = fig.add_axes([0.72, 0.4, 0.25, 0.3])
ax_energy.set_facecolor('#f9f9f9')
ax_energy.set_title("Energy", fontsize=10)
line_k, = ax_energy.plot([], [], 'r-', label='KE', lw=1)
line_p, = ax_energy.plot([], [], 'b-', label='PE', lw=1)
line_tot, = ax_energy.plot([], [], 'k--', label='Total', lw=1)
ax_energy.legend(fontsize=8)
ax_energy.set_visible(False)


# widgets
ax_h = plt.axes([0.25, 0.20, 0.50, 0.03])
h_slider = Slider(ax_h, 'height', 0.0, 40.0, valinit=0.0)
h_slider.on_changed(update_h)

ax_v = plt.axes([0.25, 0.15, 0.50, 0.03])
v_slider = Slider(ax_v, 'speed', 1.0, 50.0, valinit=20.0)
v_slider.on_changed(update_v)

ax_a = plt.axes([0.25, 0.10, 0.50, 0.03])
a_slider = Slider(ax_a, 'angle', 0.0, 90.0, valinit=45.0)
a_slider.on_changed(update_a)

ax_fire = plt.axes([0.8, 0.05, 0.1, 0.05])
fire_button = Button(ax_fire, 'fire', color='orange', hovercolor='yellow')
fire_button.on_clicked(on_button_click)


# energy toggle button
ax_en_btn = plt.axes([0.8, 0.12, 0.1, 0.05])
energy_button = Button(ax_en_btn, 'show E', color='lightblue', hovercolor='cyan')
energy_button.on_clicked(toggle_energy)


# radio buttons for friction
ax_radio = plt.axes([0.02, 0.4, 0.15, 0.15])
ax_radio.set_facecolor('#f0f0f0')
radio = RadioButtons(ax_radio, ('no drag', 'linear', 'quadratic'))
radio.on_clicked(change_drag)


ani = FuncAnimation(fig, update, frames=200, interval=30, blit=False)
plt.show()


