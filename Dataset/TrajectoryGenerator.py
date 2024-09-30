import numpy as np
import matplotlib.pyplot as plt


def getRandomGraph():
    # generate a random graph
    nodes = np.random.rand(10, 2)  # 10 nodes in 2D
    adj_mat = np.random.randint(0, 10, (10, 10))  # 10x10 adjacency matrix
    adj_mat = np.triu(adj_mat, 1) + np.triu(adj_mat, 1).T  # make it symmetric
    adj_mat = np.where(adj_mat > 4, 1, 0)  # make it binary
    adj_mat = np.triu(adj_mat, 1)  # keep only the upper triangle
    edges = np.argwhere(adj_mat == 1)  # get the edges

    # randomly choose a start node
    start_node = np.random.randint(0, 10)
    # randomly choose a target node
    target_node = np.random.randint(0, 10)
    # find a path from start to target node
    current_node = start_node
    visited = []
    queue = [current_node]
    while current_node != target_node:
        current_node = queue.pop(0)
        visited.append(current_node)
        neighbors = np.argwhere(edges[:, 0] == current_node).flatten()
        np.random.shuffle(neighbors)
        for neighbor in neighbors:
            if edges[neighbor, 1] not in visited:
                queue.append(edges[neighbor, 1])
        if len(queue) == 0:
            break
    visited.append(target_node)

    return nodes, edges, visited


def simulateTrajectory(visiting_nodes: np.ndarray, step_mean: float, step_std: float, noise_std: float):
    """
    Simulate a trajectory by walking along the path defined by the visiting nodes.
    :param visiting_nodes: (N, 2) array of nodes to visit
    :param step_mean: the mean of the step distance
    :param step_std: the standard deviation of the step distance
    :param noise_std: the standard deviation of the GPS noise
    :return: (M, 2) array of the simulated trajectory
    """
    # compute the distance from the start node to each node
    pairwise_dist = np.linalg.norm(visiting_nodes[1:] - visiting_nodes[:-1], axis=1)
    distances = np.cumsum(pairwise_dist)
    distances = np.insert(distances, 0, 0)

    # simulate the random walk
    walked_dist = 0
    current_pos = visiting_nodes[0]
    trajectory = [current_pos]
    while walked_dist < distances[-1]:
        next_step_distance = np.random.normal(step_mean, step_std)
        walked_dist += next_step_distance
        # find the position of the walker along the path
        for i in range(len(distances) - 1):
            if distances[i] <= walked_dist < distances[i + 1]:
                current_pos = visiting_nodes[i] + (walked_dist - distances[i]) / pairwise_dist[i] * (
                            visiting_nodes[i + 1] - visiting_nodes[i])
                break
        trajectory.append(current_pos)

    trajectory.append(visiting_nodes[-1])
    trajectory = np.array(trajectory)

    # simulate some GPS noise
    gps_noise = np.random.normal(0, noise_std, trajectory.shape)

    return trajectory + gps_noise



if __name__ == "__main__":
    nodes, edges, visited = getRandomGraph()
    start_node = visited[0]
    target_node = visited[-1]
    traj_dataset = []
    for i in range(20):
        traj_dataset.append(simulateTrajectory(nodes[visited], 0.15, 0.05, 0.01))

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(nodes[:, 0], nodes[:, 1], marker=".")
    for edge in edges:
        plt.plot(nodes[edge, 0], nodes[edge, 1], c='black', alpha=0.5, linewidth=0.5)

    # highlight start node
    plt.scatter(nodes[start_node, 0], nodes[start_node, 1], c='red', s=100, marker="$S$")
    # highlight target node
    plt.scatter(nodes[target_node, 0], nodes[target_node, 1], c='green', s=100, marker="$T$")
    # highlight the path
    for i in range(len(visited) - 1):
        plt.plot(nodes[[visited[i], visited[i + 1]], 0], nodes[[visited[i], visited[i + 1]], 1], c='blue', linewidth=2)

    # get xlim and ylim for plot 1
    xlim = plt.xlim()
    ylim = plt.ylim()

    # plot the trajectory
    plt.subplot(122)
    for trajectory in traj_dataset:
        plt.scatter(trajectory[:, 0], trajectory[:, 1], c='red', alpha=0.5, marker='.')
        plt.plot(trajectory[:, 0], trajectory[:, 1], c='gray', linewidth=1, alpha=0.1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()