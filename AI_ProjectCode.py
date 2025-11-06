import time
import heapq
import random
from collections import deque
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------
# Configuration
# ------------------------
GRID_ROWS = 30
GRID_COLS = 40
OBSTACLE_DENSITY = 0.20  # fraction of cells that are obstacles (0..1)
RANDOM_SEED = 42

START = (2, 2)   # (row, col)
GOAL = (25, 35)  # (row, col)

USE_ASTAR = True           # True -> A*, False -> BFS
ASTAR_HEURISTIC = "manhattan"  # "manhattan" or "euclidean"

VISUALIZE_SEARCH = True    # visualize explored cells (slow for large grids)
ANIMATE_ROBOT = True       # animate the robot following the path
PAUSE_BETWEEN_FRAMES = 0.02  # pause between frames in seconds

# ------------------------
# Utilities
# ------------------------
def neighbors(cell, grid):
    """Return all free 4-neighbors (up/down/left/right) inside grid bounds."""
    (r, c) = cell
    deltas = [(-1,0),(1,0),(0,-1),(0,1)]
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if grid[nr, nc] == 0:  # 0 means free
                yield (nr, nc)

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ------------------------
# Search algorithms
# ------------------------
def bfs_search(grid, start, goal):
    """
    BFS search returning (came_from, visited_order, nodes_expanded, runtime).
    came_from: dict child -> parent
    visited_order: list of visited nodes in order (useful for visualization)
    """
    t0 = time.time()
    q = deque([start])
    came_from = {start: None}
    visited_order = []
    nodes_expanded = 0

    while q:
        cur = q.popleft()
        visited_order.append(cur)
        nodes_expanded += 1
        if cur == goal:
            break
        for nb in neighbors(cur, grid):
            if nb not in came_from:
                came_from[nb] = cur
                q.append(nb)

    t1 = time.time()
    return came_from, visited_order, nodes_expanded, t1 - t0

def astar_search(grid, start, goal, heuristic="manhattan"):
    """
    A* search returning (came_from, visited_order, nodes_expanded, runtime).
    Uses tie-breaking on f = g + h, and g (distance from start).
    """
    t0 = time.time()
    if heuristic == "manhattan":
        hfun = manhattan
    else:
        hfun = euclidean

    open_heap = []  # elements: (f, g, node, parent)
    entry_finder = {}  # node -> best g found
    came_from = {}
    visited_order = []
    nodes_expanded = 0

    g_start = 0
    f_start = g_start + hfun(start, goal)
    heapq.heappush(open_heap, (f_start, g_start, start))
    entry_finder[start] = g_start
    came_from[start] = None

    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        # If this entry is outdated (we found a better g earlier), skip it
        if entry_finder.get(cur, float('inf')) < g:
            continue

        visited_order.append(cur)
        nodes_expanded += 1
        if cur == goal:
            break

        for nb in neighbors(cur, grid):
            tentative_g = g + 1  # cost between neighbors is 1
            if tentative_g < entry_finder.get(nb, float('inf')):
                entry_finder[nb] = tentative_g
                came_from[nb] = cur
                f_nb = tentative_g + hfun(nb, goal)
                heapq.heappush(open_heap, (f_nb, tentative_g, nb))

    t1 = time.time()
    return came_from, visited_order, nodes_expanded, t1 - t0

# ------------------------
# Path reconstruction
# ------------------------
def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path

# ------------------------
# Map generation / helpers
# ------------------------
def generate_random_grid(rows, cols, obstacle_density=0.2, seed=None, start=None, goal=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    grid = np.zeros((rows, cols), dtype=int)
    # Place obstacles
    for r in range(rows):
        for c in range(cols):
            if random.random() < obstacle_density:
                grid[r, c] = 1
    # Clear start and goal
    if start:
        grid[start] = 0
    if goal:
        grid[goal] = 0
    return grid

def pretty_print_grid(grid, start=None, goal=None, path=None):
    rows, cols = grid.shape
    display = np.full((rows, cols), '.', dtype=str)
    display[grid==1] = '#'
    if path:
        for (r,c) in path:
            display[r,c] = 'o'
    if start:
        display[start] = 'S'
    if goal:
        display[goal] = 'G'
    for r in range(rows):
        print(''.join(display[r,:]))
    print()

# ------------------------
# Visualization & Animation
# ------------------------
def visualize_search(grid, start, goal, visited_order, path, title="Path Planning"):
    """
    Visualize search progress and final path using matplotlib.
    visited_order: list of nodes in order they were popped/visited by the search
    path: list of nodes forming the final path (or None)
    """
    rows, cols = grid.shape
    # Build base image: 0 free: white, 1 obstacle: black
    img = np.zeros((rows, cols, 3), dtype=float)
    img[grid==0] = [1.0, 1.0, 1.0]  # free -> white
    img[grid==1] = [0.0, 0.0, 0.0]  # obstacle -> black

    fig, ax = plt.subplots(figsize=(cols/4, rows/4))
    ax.set_title(title)
    im = ax.imshow(img, interpolation='nearest', origin='upper')

    def update_frame(i):
        frame_img = img.copy()
        # color visited nodes (light blue)
        for k in range(min(i, len(visited_order))):
            r,c = visited_order[k]
            frame_img[r,c] = [0.6, 0.9, 1.0]
        # color frontier/last visited (orange)
        if i > 0 and i <= len(visited_order):
            r,c = visited_order[min(i-1, len(visited_order)-1)]
            frame_img[r,c] = [1.0, 0.6, 0.0]
        # draw path if available (red)
        if path:
            for (r,c) in path:
                frame_img[r,c] = [1.0, 0.0, 0.0]
        # start and goal markers (green and purple)
        sr, sc = start
        gr, gc = goal
        frame_img[sr, sc] = [0.0, 1.0, 0.0]
        frame_img[gr, gc] = [0.6, 0.0, 0.8]
        im.set_data(frame_img)
        return (im,)

    frames = len(visited_order) + 5
    ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=PAUSE_BETWEEN_FRAMES*1000, blit=False, repeat=False)
    plt.show()
    return ani

def animate_robot_following_path(grid, path):
    if not path:
        print("No path to animate.")
        return
    rows, cols = grid.shape
    img = np.zeros((rows, cols, 3), dtype=float)
    img[grid==0] = [1.0, 1.0, 1.0]
    img[grid==1] = [0.0, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(cols/4, rows/4))
    im = ax.imshow(img, interpolation='nearest', origin='upper')
    ax.set_title("Robot following path (blue)")

    def update(i):
        frame = img.copy()
        # draw path in faint red
        for (r,c) in path:
            frame[r,c] = [1.0, 0.7, 0.7]
        # robot as blue filled cell at current position
        r, c = path[min(i, len(path)-1)]
        frame[r,c] = [0.0, 0.2, 1.0]
        # start and goal markers
        sr, sc = path[0]
        gr, gc = path[-1]
        frame[sr, sc] = [0.0, 1.0, 0.0]
        frame[gr, gc] = [0.6, 0.0, 0.8]
        im.set_data(frame)
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(path)+5, interval=150, blit=False, repeat=False)
    plt.show()
    return ani

# ------------------------
# Main demo
# ------------------------
def main():
    global USE_ASTAR, ASTAR_HEURISTIC
    # generate random grid
    grid = generate_random_grid(GRID_ROWS, GRID_COLS, obstacle_density=OBSTACLE_DENSITY,
                                seed=RANDOM_SEED, start=START, goal=GOAL)
    # ensure start/goal are free; if not, clear them
    grid[START] = 0
    grid[GOAL] = 0

    print("Warehouse Robot Path Planning Demo")
    print(f"Grid: {GRID_ROWS} x {GRID_COLS}, obstacle density: {OBSTACLE_DENSITY}")
    print(f"Start: {START}, Goal: {GOAL}")
    print("Algorithm:", "A*" if USE_ASTAR else "BFS", f"(heuristic={ASTAR_HEURISTIC})" if USE_ASTAR else "")

    # run selected search
    if USE_ASTAR:
        came_from, visited_order, nodes_expanded, runtime = astar_search(grid, START, GOAL, heuristic=ASTAR_HEURISTIC)
    else:
        came_from, visited_order, nodes_expanded, runtime = bfs_search(grid, START, GOAL)

    path = reconstruct_path(came_from, START, GOAL)
    if path:
        print(f"Path found! Length: {len(path)-1} moves, nodes expanded: {nodes_expanded}, time: {runtime:.6f}s")
    else:
        print(f"No path found. nodes expanded: {nodes_expanded}, time: {runtime:.6f}s")

    # text print small grids
    if GRID_ROWS <= 20 and GRID_COLS <= 40:
        print("\nMap (S=start, G=goal, #=obstacle, o=path):")
        pretty_print_grid(grid, start=START, goal=GOAL, path=path)

    # show search visualization
    if VISUALIZE_SEARCH:
        title = f"{'A*' if USE_ASTAR else 'BFS'} search visualization | nodes expanded: {nodes_expanded}"
        visualize_search(grid, START, GOAL, visited_order, path, title=title)

    # animate robot moving along path
    if ANIMATE_ROBOT and path:
        animate_robot_following_path(grid, path)

if __name__ == "__main__":
    main()
