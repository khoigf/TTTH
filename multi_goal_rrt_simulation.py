import numpy as np
import matplotlib.pyplot as plt
import random
import time

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def distance(node1, node2):
    return np.hypot(node1.x - node2.x, node1.y - node2.y)

def classify_obstacle(obstacle):
    return 'static' if obstacle['type'] == 'static' else 'dynamic'

def is_collision_free_static(node1, node2, obstacles, step_size=1.0):
    steps = int(distance(node1, node2) / step_size)
    for i in range(steps):
        x = node1.x + (node2.x - node1.x) * i / steps
        y = node1.y + (node2.y - node1.y) * i / steps
        for obs in obstacles:
            if classify_obstacle(obs) == 'static' and distance(Node(x, y), Node(obs['position'][0], obs['position'][1])) <= obs['radius']:
                return False
    return True

def will_collide_dynamic(robot_pos, robot_velocity, obstacle_pos, obstacle_velocity, safe_distance=5, buffer_distance=3):
    rel_pos = np.array(obstacle_pos) - np.array(robot_pos)
    rel_velocity = np.array(obstacle_velocity) - np.array(robot_velocity)
    tca = -np.dot(rel_pos, rel_velocity) / np.dot(rel_velocity, rel_velocity)
    tca = max(0, tca)
    closest_dist = np.linalg.norm(rel_pos + tca * rel_velocity)
    return closest_dist < (safe_distance + buffer_distance)

def avoid_obstacle_direction(robot_position, obstacle_position, direction):
    away_vector = robot_position - obstacle_position
    away_vector /= np.linalg.norm(away_vector)
    adjusted_direction = direction + 0.5 * away_vector
    return adjusted_direction / np.linalg.norm(adjusted_direction)

def rrt_with_obstacles(start, goal, obstacles, max_iter=500, step_size=5.0, goal_sample_rate=0.2):
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
            plt.plot([nearest_node.x, new_node.x], [nearest_node.y, new_node.y], "-g")
            plt.pause(0.01)
            
            if distance(new_node, goal_node) <= step_size:
                goal_node.parent = new_node
                nodes.append(goal_node)
                return nodes
    return nodes

def extract_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

def update_robot_position(robot_position, path, robot_velocity, speed, dynamic_obstacles):
    if len(path) > 1:
        next_point = np.array(path[1], dtype=np.float64)
        direction = next_point - robot_position
        direction /= np.linalg.norm(direction)

        for obs in dynamic_obstacles:
            if classify_obstacle(obs) == "dynamic" and will_collide_dynamic(robot_position, robot_velocity, obs['position'], obs['velocity']):
                direction = avoid_obstacle_direction(robot_position, obs['position'], direction)
                speed = speed / 2  # Reduce speed to avoid collision more accurately
                break

        move_step = direction * speed
        robot_position += move_step

        if np.linalg.norm(robot_position - next_point) < speed:
            path.pop(0)

    return robot_position

def plot_environment(robot_position, path, obstacles, goal_position):
    plt.clf()
    plt.plot(robot_position[0], robot_position[1], "bo", label="Robot")
    plt.plot(goal_position[0], goal_position[1], "ro", label="Goal")
    for obs in obstacles:
        color = "gray" if classify_obstacle(obs) == 'static' else "orange"
        circle = plt.Circle((obs['position'][0], obs['position'][1]), obs['radius'], color=color, alpha=0.5)
        plt.gca().add_patch(circle)
    if path:
        plt.plot([x for x, y in path], [y for x, y in path], '-r', linewidth=2, label="Path")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.pause(0.01)

def update_dynamic_obstacles(obstacles):
    for obs in obstacles:
        if classify_obstacle(obs) == "dynamic":
            obs['position'] += obs['velocity']
            if obs['position'][0] <= 0 or obs['position'][0] >= 100:
                obs['velocity'][0] *= -1
            if obs['position'][1] <= 0 or obs['position'][1] >= 100:
                obs['velocity'][1] *= -1

    for i in range(len(obstacles)):
        if classify_obstacle(obstacles[i]) == "dynamic":
            for j in range(i + 1, len(obstacles)):
                if classify_obstacle(obstacles[j]) == "dynamic":
                    dist = np.linalg.norm(obstacles[i]['position'] - obstacles[j]['position'])
                    min_dist = obstacles[i]['radius'] + obstacles[j]['radius']
                    if dist < min_dist:
                        direction = (obstacles[i]['position'] - obstacles[j]['position']) / dist
                        overlap = min_dist - dist
                        obstacles[i]['position'] += direction * (overlap / 2)
                        obstacles[j]['position'] -= direction * (overlap / 2)
                        obstacles[i]['velocity'] -= direction * 0.1
                        obstacles[j]['velocity'] += direction * 0.1

            for j in range(len(obstacles)):
                if classify_obstacle(obstacles[j]) == "static":
                    dist = np.linalg.norm(obstacles[i]['position'] - obstacles[j]['position'])
                    min_dist = obstacles[i]['radius'] + obstacles[j]['radius']
                    if dist < min_dist:
                        direction = (obstacles[i]['position'] - obstacles[j]['position']) / dist
                        overlap = min_dist - dist
                        obstacles[i]['position'] += direction * overlap
                        obstacles[i]['velocity'] += direction * 0.1

def main():
    start = (10, 10)
    goals = [(90, 90), (10, 80), (80, 30)]
    obstacles = [
        {'type': 'static', 'position': np.array([30, 30], dtype='float64'), 'radius': 10},
        {'type': 'static', 'position': np.array([50, 70], dtype='float64'), 'radius': 15},
        {'type': 'static', 'position': np.array([70, 20], dtype='float64'), 'radius': 12},
        {'type': 'dynamic', 'position': np.array([60, 60], dtype='float64'), 'velocity': np.array([-0.4, 0.3], dtype='float64'), 'radius': 5},
        {'type': 'dynamic', 'position': np.array([15, 80], dtype='float64'), 'velocity': np.array([0.2, -0.2], dtype='float64'), 'radius': 4},
        {'type': 'dynamic', 'position': np.array([32, 80], dtype='float64'), 'velocity': np.array([0, -0.3], dtype='float64'), 'radius': 4}
    ]

    robot_position = np.array(start, dtype=np.float64)
    robot_velocity = np.array([0.6, 0.6])
    speed = 1
    
    for goal in goals:
        plt.clf()
        plot_environment(robot_position, [], obstacles, goal)
        nodes = rrt_with_obstacles(robot_position, goal, obstacles)
        goal_node = nodes[-1]
        path = extract_path(goal_node)
    
        while len(path) > 1:
            update_dynamic_obstacles(obstacles)
            robot_position = update_robot_position(robot_position, path, robot_velocity, speed, obstacles)
            plot_environment(robot_position, path, obstacles, goal)
            time.sleep(0.1)

    plt.show()

if __name__ == '__main__':
    main()
