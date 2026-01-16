# coupled harmonic oscillator
# add masses and see what happens
# spring chain

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button











# physics globals
# lists for N masses
masses = [1.0]
k_constants = [10.0]
x_pos = [0.0]  # displacement from equilibrium
velocities = [0.0]
damping = 0.5
rest_length = 2.0

t = 0.0
dt = 0.05
paused = False
dragging_idx = -1

# for plotting history
history_t = []
history_x = [[]] 
history_ke = [[]] # list of lists per mass
history_pe = [[]]
# history_e totals can be derived or stored separate
history_total_ke = []
history_total_pe = []
history_total_e = []

# graphics objects
spring_lines = []
mass_points = []
lines_hist = []

max_hist = 200

# selected mass to edit
selected_idx = 0
plot_mode = 'system' # system or custom
selected_curves = set() # strings like 'm0_k', 'm0_p'










# physics solver
def compute_accels():
    accels = []
    n = len(masses)
    
    for i in range(n):
        # force from left spring
        # if i=0, attached to wall (x=0)
        # displacement is x_pos[i]
        
        f_left = -k_constants[i]*x_pos[i]
        
        if i > 0:
            # force from left neighbor pulling back
            # actually wait. 
            # force on i from i-1: k*(x[i-1] - x[i])
            f_left = -k_constants[i]*(x_pos[i]-x_pos[i-1])
            
        else:
            # attached to wall at 0
            f_left = -k_constants[i]*x_pos[i]
            
            
        f_right = 0.0
        if i < n-1:
            # force from right neighbor (i+1)
            # k_next * (x[i+1] - x[i])
            f_right = k_constants[i+1]*(x_pos[i+1]-x_pos[i])
            
            
        # damping
        f_drag = -damping*velocities[i]
        
        f_net = f_left+f_right+f_drag
        a = f_net/masses[i]
        accels.append(a)
        
    return accels


def check_collisions():
    # simple bounce
    min_dist = 0.2
    
    # 1. check wall (mass 0)
    # wall is at 0, eq pos is rest_length
    # actual pos = rest_length + x_pos[0]
    pos0 = rest_length + x_pos[0]
    if pos0 < min_dist:
        x_pos[0] = min_dist - rest_length
        velocities[0] *= -0.8 # inelastic wall
        
    # 2. check neighbors
    for i in range(len(masses)-1):
        # i and i+1
        pos_i = (i+1)*rest_length + x_pos[i]
        pos_next = (i+2)*rest_length + x_pos[i+1]
        
        dist = pos_next - pos_i
        if dist < min_dist:
            # collision!
            # exchange momentum ish
            # messy student physics: just swap v and separate
            
            # separation
            overlap = min_dist - dist
            x_pos[i] -= overlap/2
            x_pos[i+1] += overlap/2
            
            # elastic collision 1D
            m1 = masses[i]
            m2 = masses[i+1]
            u1 = velocities[i]
            u2 = velocities[i+1]
            
            v1 = (u1*(m1-m2) + 2*m2*u2) / (m1+m2)
            v2 = (u2*(m2-m1) + 2*m1*u1) / (m1+m2)
            
            velocities[i] = v1
            velocities[i+1] = v2










# adding/removing
def add_mass(event):
    global masses, k_constants, x_pos, velocities, history_x, history_t
    global history_ke, history_pe, history_total_ke, history_total_pe, history_total_e
    
    # add default
    masses.append(1.0)
    k_constants.append(10.0)
    x_pos.append(0.0)
    velocities.append(0.0)
    
    # reset history
    history_t = []
    history_x = [[] for _ in range(len(masses))]
    history_ke = [[] for _ in range(len(masses))]
    history_pe = [[] for _ in range(len(masses))]
    history_total_ke = []
    history_total_pe = []
    history_total_e = []
    
    update_plot_objects()
    update_checkbuttons()


def update_plot_objects():
    # rebuild the plot lines for history
    ax_plot.cla()
    ax_plot.set_xlim(0, 10) # dummy
    ax_plot.set_ylim(-5, 5)
    ax_plot.grid(True)
    
    global lines_hist, spring_lines, mass_points
    lines_hist = []
    for i in range(len(masses)):
        ln, = ax_plot.plot([], [], label=f'm{i}')
        lines_hist.append(ln)
    ax_plot.legend(loc='upper right', fontsize='x-small')
    
    # rebuild system graphics
    ax_main.cla()
    ax_main.set_xlim(0, (len(masses)+2)*rest_length)
    ax_main.set_ylim(-2, 2)
    ax_main.axhline(0, color='gray', linestyle=':')
    ax_main.set_yticks([])
    ax_main.plot([0,0], [-1,1], 'k-', lw=5) # wall
    
    spring_lines = []
    mass_points = []
    
    for i in range(len(masses)):
        # spring line
        sl, = ax_main.plot([], [], 'k-', lw=1)
        spring_lines.append(sl)
        # mass point
        mp, = ax_main.plot([], [], 'o', ms=10)
        mass_points.append(mp)


def update_k(val):
    k_constants[selected_idx] = k_slider.val
    
def update_m(val):
    masses[selected_idx] = m_slider.val
    
def select_prev(event):
    global selected_idx
    selected_idx = max(0, selected_idx - 1)
    refresh_sliders()

def select_next(event):
    global selected_idx
    selected_idx = min(len(masses)-1, selected_idx + 1)
    refresh_sliders()

def refresh_sliders():
    # update text
    sel_text.set_text(f"editing: mass {selected_idx}")
    
    # update slider vals without triggering callback loop?
    # mpl sliders are annoying, just set val
    k_slider.eventson = False
    m_slider.eventson = False
    
    k_slider.set_val(k_constants[selected_idx])
    m_slider.set_val(masses[selected_idx])
    
    k_slider.eventson = True
    m_slider.eventson = True


from matplotlib.widgets import RadioButtons, CheckButtons

check_widget = None

ax_table = None

def update_checkbuttons():
    global check_widget, ax_table
    
    # remove old axes if exists
    if ax_table is not None:
        try:
            ax_table.remove()
        except:
            pass
    ax_table = None
            
    if plot_mode == 'custom':
        # create new axes
        ax_table = fig.add_axes([0.86, 0.7, 0.13, 0.25])
        ax_table.set_facecolor('#f0f0f0') # bg
        # ax_table.axis('off') # CheckButtons needs axis? actually it draws on it.
        # CheckButtons usually needs frame?
        # let's keep axis on but invisible spines?
        ax_table.axis('off')
        
        # make labels
        labels = []
        actives = []
        for i in range(len(masses)):
            lbl_k = f"m{i} K"
            lbl_p = f"m{i} U"
            labels.append(lbl_k)
            labels.append(lbl_p)
            actives.append(lbl_k in selected_curves)
            actives.append(lbl_p in selected_curves)
            
        check_widget = CheckButtons(ax_table, labels, actives)
        
        # Adjust label size
        for l in check_widget.labels:
            l.set_fontsize(8)
            
        check_widget.on_clicked(toggle_curve)

def toggle_curve(label):
    if label in selected_curves:
        selected_curves.remove(label)
    else:
        selected_curves.add(label)

def change_plot_mode(label):
    global plot_mode
    plot_mode = label
    
    if plot_mode == 'system':
        ax_energy.set_title("Energy (System)")
    else:
        ax_energy.set_title("Energy (Custom)")
        
    update_checkbuttons()


# interaction
def on_press(event):
    global dragging_idx
    if event.inaxes != ax_main:
        return
        
    # find closest mass
    # visuals: x is (i+1)*rest + disp
    click_x = event.xdata
    
    min_dist = 100
    closest = -1
    
    for i in range(len(masses)):
        visual_x = (i+1)*rest_length + x_pos[i]
        dist = abs(click_x - visual_x)
        if dist < 0.5:
            if dist < min_dist:
                min_dist = dist
                closest = i
                
    if closest != -1:
        dragging_idx = closest


def on_motion(event):
    if dragging_idx != -1 and event.inaxes == ax_main:
        # set pos
        # x_screen = (i+1)*L + x_disp
        # x_disp = x_screen - (i+1)*L
        
        wanted_screen_x = event.xdata
        eq_pos = (dragging_idx+1)*rest_length
        x_pos[dragging_idx] = wanted_screen_x - eq_pos
        velocities[dragging_idx] = 0.0


def on_release(event):
    global dragging_idx
    dragging_idx = -1
    
def toggle_pause(event):
    global paused
    paused = not paused










# update loop
def update(frame):
    global t, history_t, history_x, plot_mode
    global history_ke, history_pe, history_total_ke, history_total_pe, history_total_e
    
    if not paused and dragging_idx == -1:
        # physics step
        accs = compute_accels()
        
        for i in range(len(masses)):
            velocities[i] += accs[i]*dt
            x_pos[i] += velocities[i]*dt
            
        check_collisions()
            
        t += dt
        history_t.append(t)
        
        # update history lists
        # ensure list of lists exists
        if len(history_x) != len(masses):
             history_x = [[] for _ in range(len(masses))]
             history_ke = [[] for _ in range(len(masses))]
             history_pe = [[] for _ in range(len(masses))]
             history_total_ke = []
             history_total_pe = []
             history_total_e = []
             history_t = []
        
        # calc energy
        sys_ke = 0.0
        sys_pe = 0.0
        
        for i in range(len(masses)):
            # KE
            v = velocities[i]
            ke_val = 0.5 * masses[i] * v*v
            history_ke[i].append(ke_val)
            sys_ke += ke_val
            
            # PE (left spring)
            if i == 0:
                ext = x_pos[i]
            else:
                ext = x_pos[i] - x_pos[i-1]
            pe_val = 0.5 * k_constants[i] * ext*ext
            history_pe[i].append(pe_val)
            sys_pe += pe_val
            
            history_x[i].append(x_pos[i])
            
        history_total_ke.append(sys_ke)
        history_total_pe.append(sys_pe)
        history_total_e.append(sys_ke + sys_pe)
        
        # trim
        if len(history_t) > max_hist:
            history_t.pop(0)
            history_total_ke.pop(0)
            history_total_pe.pop(0)
            history_total_e.pop(0)
            for i in range(len(masses)):
                history_x[i].pop(0)
                history_ke[i].pop(0)
                history_pe[i].pop(0)


    # DRAWING
    # 1. Main system
    # update objects
    if len(spring_lines) != len(masses):
        # safety rebuild
        update_plot_objects()
        
    for i in range(len(masses)):
        # Equilibrium position
        eq_x = (i+1)*rest_length
        # Actual position
        pos_x = eq_x + x_pos[i]
        
        # Draw spring from prev to curr
        prev_pos_x = 0.0 if i==0 else ((i)*rest_length + x_pos[i-1])
        
        # zigzag spring
        sx = np.linspace(prev_pos_x, pos_x, 10)
        sy = np.array([0, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0])
        
        spring_lines[i].set_data(sx, sy)
        
        # Draw mass
        col = 'red' if i == selected_idx else 'blue'
        sz = 10 + masses[i]*5
        mass_points[i].set_data([pos_x], [0])
        mass_points[i].set_color(col)
        mass_points[i].set_markersize(sz)
        
        # label? messy to update text every frame efficiently
        # skip label or add text objects list?
        # student code: skip label update or just accept it's missing labels now
        # OR: clear text only? nah.


    # 2. Plots
    # update existing lines from lines_hist
    
    # check if lines_hist needs rebuild (safety)
    if len(lines_hist) != len(masses):
        update_plot_objects()
    
    for i in range(len(masses)):
        if len(history_t) > 0 and len(history_x[i]) > 0:
            lines_hist[i].set_data(history_t, history_x[i])
            
    # auto scroll x
    if len(history_t)>0:
        ax_plot.set_xlim(min(history_t), max(history_t)+1)
        ax_plot.set_ylim(-5, 5) # enforce y limits so it doesnt jump
        
        # update energy plot
        ax_energy.cla()
        ax_energy.set_title(f"Energy ({plot_mode})")
        ax_energy.grid(True)
        ax_energy.set_xlim(min(history_t), max(history_t)+1)
        
        if plot_mode == 'system':
            # use totals
            if len(history_total_e) == len(history_t):
                ax_energy.plot(history_t, history_total_e, 'k--', label='Tot')
                ax_energy.plot(history_t, history_total_ke, 'r-', label='K')
                ax_energy.plot(history_t, history_total_pe, 'b-', label='U')
                max_e = max(history_total_e) if len(history_total_e)>0 else 1.0
                ax_energy.legend(loc='upper right', fontsize='x-small')
            else:
                max_e = 1.0
        else:
            # custom mode
            max_e = 0.1
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for label in selected_curves:
                # label is "m{i} K" or "m{i} U"
                parts = label.split()
                m_idx = int(parts[0][1:])
                e_type = parts[1]
                
                if m_idx < len(masses):
                    # color by mass index
                    col = colors[m_idx % len(colors)]
                    
                    if e_type == 'K':
                        data = history_ke[m_idx]
                        ls = '-'
                    else:
                        data = history_pe[m_idx]
                        ls = '--'
                        
                    if len(data)==len(history_t):
                        ax_energy.plot(history_t, data, label=label, color=col, linestyle=ls)
                        if len(data) > 0:
                            max_e = max(max_e,max(data))
                            
            ax_energy.legend(loc='upper right', fontsize='xx-small')

        # auto y for energy
        ax_energy.set_ylim(0,max_e*1.1)










# setup figure
fig = plt.figure(figsize=(8, 8))

# Grid layout: 
# Top: Plot (30%)
# Mid: System (30%)
# Bot: Controls (40%)

ax_plot = fig.add_axes([0.05, 0.7, 0.4, 0.25])
lines_hist = [] # init
history_x = [[]] # init for 1 mass

ax_energy = fig.add_axes([0.50, 0.7, 0.35, 0.25])
ax_energy.set_title("Energy")
ax_energy.grid(True)
line_et, = ax_energy.plot([], [], 'k--', label='Tot')
line_ek, = ax_energy.plot([], [], 'r-', label='K')
line_ep, = ax_energy.plot([], [], 'b-', label='U')
ax_energy.legend(loc='upper right', fontsize='x-small')

# radio for plot mode
ax_radio = plt.axes([0.86, 0.6, 0.13, 0.08])
ax_radio.set_facecolor('#f0f0f0')
radio = RadioButtons(ax_radio, ('system', 'custom'))
radio.on_clicked(change_plot_mode)


ax_main = fig.add_axes([0.1, 0.35, 0.8, 0.25])

# controls area
# Add mass button
ax_add = plt.axes([0.1, 0.2, 0.2, 0.05])
btn_add = Button(ax_add, "add mass")
btn_add.on_clicked(add_mass)

# Pause
ax_pause = plt.axes([0.4, 0.2, 0.2, 0.05])
btn_pause = Button(ax_pause, "pause")
btn_pause.on_clicked(toggle_pause)

# Selection controls
ax_prev = plt.axes([0.1, 0.12, 0.1, 0.05])
btn_prev = Button(ax_prev, "<")
btn_prev.on_clicked(select_prev)

ax_next = plt.axes([0.3, 0.12, 0.1, 0.05])
btn_next = Button(ax_next, ">")
btn_next.on_clicked(select_next)

# Text showing selection
sel_text = plt.text(0.2, 0.1, "editing: mass 0", transform=fig.transFigure, ha='center')

# Sliders for selected
ax_k = plt.axes([0.5, 0.12, 0.4, 0.03])
k_slider = Slider(ax_k, 'k', 1.0, 50.0, valinit=10.0)
k_slider.on_changed(update_k)

ax_m = plt.axes([0.5, 0.07, 0.4, 0.03])
m_slider = Slider(ax_m, 'mass', 0.1, 10.0, valinit=1.0)
m_slider.on_changed(update_m)


# Connect
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)


ani = FuncAnimation(fig, update, interval=30)
plt.show()

# graveyard
# k_matrix = [[0,0],[0,0]]
# eigenvalues = np.linalg.eig(k_matrix)
# print("modes", eigenvalues)
