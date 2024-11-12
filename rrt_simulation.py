import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Define the Node for RRT
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

# Calculate Euclidean distance
def distance(node1, node2):
    return np.hypot(node1.x - node2.x, node1.y - node2.y)

# Check for collision with static obstacles
def is_collision_free_static(node1, node2, obstacles, step_size=1.0):
    steps = int(distance(node1, node2) / step_size)
    for i in range(steps):
        x = node1.x + (node2.x - node1.x) * i / steps
        y = node1.y + (node2.y - node1.y) * i / steps
        for obs in obstacles:
            if distance(Node(x, y), Node(obs['position'][0], obs['position'][1])) <= obs['radius']:
                return False
    return True

# RRT Path Planning with visualization
def rrt_with_visualization(start, goal, obstacles, max_iter=500, step_size=5.0, goal_sample_rate=0.2):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    nodes = [start_node]
    
    for _ in range(max_iter):
        if random.random() < goal_sample_rate:
            rand_node = goal_node
        else:
            rand_node = Node(random.uniform(0, 100), random.uniform(0, 100))
        
        nearest_node = min(nodes, key=lambda node: distance(node, rand_node))
        
        theta = np.arctan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
        new_node = Node(nearest_node.x + step_size * np.cos(theta),
                        nearest_node.y + step_size * np.sin(theta))
        new_node.parent = nearest_node

        if is_collision_free_static(nearest_node, new_node, obstacles):
            nodes.append(new_node)
            # Plot the new branch of the RRT tree
            plt.plot([nearest_node.x, new_node.x], [nearest_node.y, new_node.y], "-g")
            plt.pause(0.01)

            if distance(new_node, goal_node) <= step_size:
                goal_node.parent = new_node
                nodes.append(goal_node)
                print("Goal reached!")
                return nodes

    print("Goal not reached within max iterations.")
    return nodes

# Extract path from RRT nodes
def extract_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

# Visualization function for RRT and obstacles
def plot_environment_with_rrt(robot_position, path, obstacles, goal_position):
    plt.clf()
    plt.plot(robot_position[0], robot_position[1], "bo", label="Start")
    plt.plot(goal_position[0], goal_position[1], "ro", label="Goal")
    for obs in obstacles:
        color = "gray" if obs['type'] == 'static' else "orange"
        circle = plt.Circle((obs['position'][0], obs['position'][1]), obs['radius'], color=color, alpha=0.5)
        plt.gca().add_patch(circle)
    if path:
        plt.plot([x for x, y in path], [y for x, y in path], '-r', linewidth=2, label="Final Path")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.pause(0.01)

# Main function
def main():
    start = (10, 10)
    goal = (90, 90)
    obstacles = [
        {'type': 'static', 'position': np.array([30, 30]), 'radius': 10},
        {'type': 'static', 'position': np.array([50, 70]), 'radius': 15},
        {'type': 'static', 'position': np.array([70, 20]), 'radius': 12},
    ]

    # Initial plot setup
    plt.figure()
    plot_environment_with_rrt(start, [], obstacles, goal)

    # Run RRT with visualization
    nodes = rrt_with_visualization(start, goal, obstacles)
    
    # Extract and plot the final path if the goal was reached
    goal_node = nodes[-1] if distance(nodes[-1], Node(goal[0], goal[1])) <= 5 else None
    if goal_node:
        path = extract_path(goal_node)
        plot_environment_with_rrt(start, path, obstacles, goal)
        print("Path found to goal!")
    else:
        print("No path found to the goal.")

    plt.show()

if __name__ == "__main__":
    main()
