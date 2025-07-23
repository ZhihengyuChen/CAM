import numpy as np
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display, clear_output

# Grid and neighbor definition
shape = (100, 100, 100) # up to 1,000,000 sites (only 50% occupied in fcu network)
grid = np.zeros(shape, dtype=int)
seeds = []

# fcu topology
neighbor_offsets = np.array([
    [ 1,  1,  0], [ 1, -1,  0], [-1,  1,  0], [-1, -1,  0],
    [ 1,  0,  1], [ 1,  0, -1], [-1,  0,  1], [-1,  0, -1],
    [ 0,  1,  1], [ 0,  1, -1], [ 0, -1,  1], [ 0, -1, -1]
])

# define the seed from here
def seed_custom():
    add_seed(shape[0]//2, shape[1]//2, shape[2]//2)
    add_seed(shape[0]//2, shape[1]//2+1, shape[2]//2+1)
    add_seed(shape[0]//2+1, shape[1]//2+1, shape[2]//2)


# Add node (crystal growth)
def add_seed(x, y, z):
    if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
        grid[x, y, z] = 1
        seeds.append((x, y, z))


# ---------- Core Function ----------
# Strict grow function 
def grow(grid, n_crit=3, n_max=6, p=1.0):
    shape_x, shape_y, shape_z = grid.shape
    active_mask = (grid == 1)

    coord_count = np.zeros_like(grid, dtype=int)
    for dx, dy, dz in neighbor_offsets:
        shifted = np.roll(active_mask, shift=(dx, dy, dz), axis=(0, 1, 2))
        coord_count += shifted

    spreading_mask = (grid == 1) & (coord_count < n_max)
    saturated_mask = (grid == 1) & (coord_count >= n_max)

    neighbor_count = np.zeros_like(grid, dtype=int)
    saturated_neighbors = np.zeros_like(grid, dtype=int)
    for dx, dy, dz in neighbor_offsets:
        neighbor_count += np.roll(spreading_mask, shift=(dx, dy, dz), axis=(0, 1, 2))
        saturated_neighbors += np.roll(saturated_mask, shift=(dx, dy, dz), axis=(0, 1, 2))

    candidate_sites = (grid == 0) & (neighbor_count >= n_crit) & (saturated_neighbors == 0)
    xs, ys, zs = np.where(candidate_sites)

    new_grid = grid.copy()
    for idx in np.random.permutation(len(xs)):
        x, y, z = xs[idx], ys[idx], zs[idx]
        if np.random.rand() > p:
            continue

        new_self_coord = 0
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape_x and 0 <= ny < shape_y and 0 <= nz < shape_z:
                if new_grid[nx, ny, nz] == 1:
                    new_self_coord += 1
        if new_self_coord > n_max:
            continue

        violate = False
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape_x and 0 <= ny < shape_y and 0 <= nz < shape_z:
                if new_grid[nx, ny, nz] == 1:
                    if coord_count[nx, ny, nz] + 1 > n_max:
                        violate = True
                        break
        if violate:
            continue

        new_grid[x, y, z] = 1
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape_x and 0 <= ny < shape_y and 0 <= nz < shape_z:
                if new_grid[nx, ny, nz] == 1:
                    coord_count[nx, ny, nz] += 1
        coord_count[x, y, z] = new_self_coord

    return new_grid

# ---------- Suppressed Sites ----------
def get_suppressed_sites(grid, n_crit=3, n_max=6):
    active_mask = (grid == 1)
    coord_count = np.zeros_like(grid, dtype=int)
    for dx, dy, dz in neighbor_offsets:
        coord_count += np.roll(active_mask, shift=(dx, dy, dz), axis=(0, 1, 2))

    spreading_mask = (grid == 1) & (coord_count < n_max)
    saturated_mask = (grid == 1) & (coord_count >= n_max)

    neighbor_count = np.zeros_like(grid, dtype=int)
    saturated_neighbors = np.zeros_like(grid, dtype=int)
    for dx, dy, dz in neighbor_offsets:
        neighbor_count += np.roll(spreading_mask, shift=(dx, dy, dz), axis=(0, 1, 2))
        saturated_neighbors += np.roll(saturated_mask, shift=(dx, dy, dz), axis=(0, 1, 2))

    return (grid == 0) & (neighbor_count >= n_crit) & (saturated_neighbors > 0)

# ---------- Plotting ----------
def plot_combined(grid, n_crit=2, n_max=6, show_active=True, show_suppressed=True):
    traces = []

    if show_active:
        active_mask = (grid == 1)
        coord_count = np.zeros_like(grid, dtype=int)
        for dx, dy, dz in neighbor_offsets:
            coord_count += np.roll(active_mask, shift=(dx, dy, dz), axis=(0, 1, 2))

        pos_active = np.argwhere(grid == 1)
        colors = coord_count[grid == 1]
        hover_text = [f"({x},{y},{z})<br>Coord: {c}" for (x, y, z), c in zip(pos_active, colors)]

        traces.append(go.Scatter3d(
            x=pos_active[:, 0], y=pos_active[:, 1], z=pos_active[:, 2],
            mode='markers',
            text=hover_text, hoverinfo='text',
            marker=dict(size=active_size_slider.value, color=colors,
                        colorscale='YlGnBu', cmin=0, cmax=12,
                        colorbar=dict(title='Coord.')),
            name='Active'
        ))

    if show_suppressed:
        pos_sup = np.argwhere(get_suppressed_sites(grid, n_crit, n_max))
        traces.append(go.Scatter3d(
            x=pos_sup[:, 0], y=pos_sup[:, 1], z=pos_sup[:, 2],
            mode='markers',
            marker=dict(size=suppressed_size_slider.value, color='red',
                        opacity=suppressed_opacity_slider.value),
            name='Suppressed'
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=800, height=700,
        title='Growth Visualization',
        scene=dict(aspectmode='cube',
                   camera=dict(projection=dict(type='orthographic')))
    )
    fig.show()

# ---------- Interaction Logic ----------
def grow_step(b):
    global grid
    grid = grow(grid, n_crit_slider.value, n_max_slider.value, p_slider.value)
    with run_output:
        clear_output(wait=True)
        plot_combined(grid, n_crit_slider.value, n_max_slider.value,
                      show_active_checkbox.value, show_suppressed_checkbox.value)

def reset_grid(b):
    global grid, seeds
    grid[:] = 0
    seeds.clear()
    seed_custom()
    with run_output:
        clear_output(wait=True)
        plot_combined(grid, n_crit_slider.value, n_max_slider.value,
                      show_active_checkbox.value, show_suppressed_checkbox.value)

def show_combined(b):
    with run_output:
        clear_output(wait=True)
        plot_combined(grid, n_crit_slider.value, n_max_slider.value,
                      show_active_checkbox.value, show_suppressed_checkbox.value)

# ---------- UI widgets ----------
n_crit_slider = widgets.IntSlider(value=2, min=1, max=12, description='Min Neighbors')
n_max_slider = widgets.IntSlider(value=6, min=1, max=12, description='Max Neighbors')
p_slider = widgets.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.05, description='Growth Prob')

active_size_slider = widgets.IntSlider(value=4, min=1, max=20, description='Active Size')
suppressed_size_slider = widgets.IntSlider(value=10, min=1, max=20, description='Suppressed Size')
suppressed_opacity_slider = widgets.FloatSlider(value=0.3, min=0.0, max=1.0, step=0.05, description='Suppressed Opacity')

show_active_checkbox = widgets.Checkbox(value=True, description="Show Active Sites")
show_suppressed_checkbox = widgets.Checkbox(value=True, description="Show Suppressed Sites")

step_button = widgets.Button(description='Grow Step')
reset_button = widgets.Button(description='Reset')
combined_button = widgets.Button(description='Show All')
export_button = widgets.Button(description='Export XYZ')

run_output = widgets.Output()

# ---------- Export to XYZ ----------
def export_xyz(grid, filename="exported_structure.xyz"):
    positions = np.argwhere(grid == 1)
    with open(filename, "w") as f:
        f.write(f"{len(positions)}\n")
        f.write("Atoms from cellular automaton model\n")
        for x, y, z in positions:
            f.write(f"C {x:.3f} {y:.3f} {z:.3f}\n")
    print(f"XYZ file saved as: {filename}")

def export_xyz_callback(b):
    export_xyz(grid)

export_button.on_click(export_xyz_callback)



step_button.on_click(grow_step)
reset_button.on_click(reset_grid)
combined_button.on_click(show_combined)

seed_custom()

# ---------- Display ----------
display(widgets.HBox([n_crit_slider, n_max_slider, p_slider]))
display(widgets.HBox([active_size_slider, suppressed_size_slider, suppressed_opacity_slider]))
display(widgets.HBox([show_active_checkbox, show_suppressed_checkbox]))
display(widgets.HBox([step_button, reset_button, combined_button, export_button]))
display(run_output)

plot_combined(grid)
