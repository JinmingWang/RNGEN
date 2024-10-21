#include "LaDeDataset.hpp"


std::vector<char> getBytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}


LaDeDataset::LaDeDataset(std::string path, 
                         int graph_depth, 
                         int trajs_per_graph,
                         int max_segs_per_graph,
                         bool rotation, 
                         float scaling_range, 
                         float traj_step_mean, 
                         float traj_step_std, 
                         float traj_noise_std, 
                         int traj_len) {

    torch::IValue x = torch::pickle_load(getBytes(path));
    auto generic_list = x.toList();

    this->nodes = generic_list.get(1).toTensor().to(DEVICE);
    this->edges = generic_list.get(2).toTensor().to(DEVICE);
    this->degrees = generic_list.get(3).toTensor().to(DEVICE);

    removeBadNodes();

    this->graph_depth = graph_depth;
    this->trajs_per_graph = trajs_per_graph;
    this->max_segs_per_graph = max_segs_per_graph;
    this->rotation = rotation;
    this->scaling_range = scaling_range;
    this->traj_step_mean = traj_step_mean;
    this->traj_step_std = traj_step_std;
    this->traj_noise_std = traj_noise_std;
    this->traj_len = traj_len;

    this->candidate_nodes = torch::where(this->degrees > 2)[0];
    cout << "Number of candidate nodes: " << this->candidate_nodes.size(0) << endl;
}


void LaDeDataset::removeBadNodes() {
    // bad node definition: For node B, if it has only two neighbors A and C
    // and the angle between BA and BC is less than 30 degrees
    // this means ABC is almost a straight line, B is meaningless
    // In this case, remove node B from nodes, degrees and edges
    // Remove neighbor B from edges of A and C
    // If A and C are already connected, reduce their degree by 1
    // If A and C are not connected, connect them

    for (int i = 0; i < this->nodes.size(0); i++) {
        if (this->degrees[i].item<int>() == 2) {
            int neighbor_1 = this->edges[i][0].item<int>();
            int neighbor_2 = this->edges[i][1].item<int>();
            Tensor node = this->nodes[i];
            Tensor neighbor_1_node = this->nodes[neighbor_1];
            Tensor neighbor_2_node = this->nodes[neighbor_2];
            Tensor vec_1 = node - neighbor_1_node;
            Tensor vec_2 = node - neighbor_2_node;
            float cos_theta = torch::dot(vec_1, vec_2) / (torch::norm(vec_1) * torch::norm(vec_2)).item<float>();
            Tensor degrees = torch::acos(cos_theta) * 180.0 / M_PI;
            if (degrees < 30) {
                // It is not wise to remove elements from a tensor, since it involves index shifting
                // We can instead just remove edge AB and BC, then set the degree of B to 0
                // This will make B an isolated node, which will never be involved in all later operations
                this->edges[i] = torch::zeros_like(this->edges[i]);     // remove all edges of B
                this->degrees[i] = 0;
                // Remove B from the neighbors of A
                int B_pos = torch::where(this->edges[neighbor_1] == i)[0].item<int>();
                int last_pos = this->degrees[neighbor_1].item<int>() - 1;
                // Move the last element to the position of B, then set the last element to 0
                this->edges.index({neighbor_1, B_pos}) = this->edges.index({neighbor_1, last_pos});
                this->edges.index({neighbor_1, last_pos}) = 0;

                B_pos = torch::where(this->edges[neighbor_2] == i)[0].item<int>();
                last_pos = this->degrees[neighbor_2].item<int>() - 1;
                this->edges.index({neighbor_2, B_pos}) = this->edges.index({neighbor_2, last_pos});
                this->edges.index({neighbor_2, last_pos}) = 0;

                if (torch::where(this->edges[neighbor_1] == neighbor_2).size(0) == 0) {
                    // Connect A and C if they are not connected
                    // Degree does not change because B is removed
                    this->edges[neighbor_1][this->degrees[neighbor_1]] = neighbor_2;
                    this->edges[neighbor_2][this->degrees[neighbor_2]] = neighbor_1;
                } else {
                    this->degrees[neighbor_1] -= 1;
                    this->degrees[neighbor_2] -= 1;
                }
            }
        }
    }
}


mat<string, Tensor> LaDeDataset::get() {
    SegmentGraph graph = this->getGraph();
    Tensor angle = this->rotation ? torch::rand({1}, DEVICE) * 2.0 * M_PI : torch::zeros({1}, DEVICE);
    Tensor scale = torch::rand({1}, DEVICE) * (this->scaling_range * 2.0) - this->scaling_range + 1.0;
    graph.transform(angle, scale);
    graph.normalize();

    // graph.draw("#000000");
    // matplot::show();

    Tensor trajs;
    Tensor paths;
    Tensor traj_lengths;
    Tensor path_lengths;
    this->simulateTrajs(graph, trajs, paths, traj_lengths, path_lengths);

    Tensor graph_tensor = graph.toTensor(this->max_segs_per_graph);

    Tensor heatmap = this->getHeatmap(graph_tensor, trajs, 64, 64);

    return {{"graph", graph_tensor},
            {"trajs", trajs},
            {"paths", paths},
            {"traj_lengths", traj_lengths},
            {"path_lengths", path_lengths},
            {"heatmap", heatmap}};
}

SegmentGraph LaDeDataset::getGraph() {
    // Randomly select a node as the starting point
    int random_idx = torch::randint(0, this->candidate_nodes.size(0), {1}).item<int>();
    int node_idx = this->candidate_nodes[random_idx].item<int>();

    // fixed_set is used to keep track of the edges that have been visited
    std::set<std::set<int>> visited_edges;
    std::set<int> visited_nodes({node_idx});
    // frontier is used to keep track of the nodes that are currently being visited
    std::set<int> frontier({node_idx});
    int depth = 0;
    while (frontier.size() > 0 && depth < this->graph_depth) {
        std::set<int> new_frontier;
        // Iterate over the nodes in the frontier
        for (auto &node_id : frontier) {
            // Iterate over the edges connected to the node
            for (int i = 0; i < this->degrees[node_id].item<int>(); i++) {
                int other_node_id = this->edges[node_id][i].item<int>();
                std::set<int> edge({node_id, other_node_id});
                // If the edge has not been visited, add it to the graph
                if (visited_edges.find(edge) == visited_edges.end()) {
                    visited_edges.insert(edge);
                    // If the other node has not been visited, add it to the frontier
                    if (visited_nodes.find(other_node_id) == visited_nodes.end()) {
                        visited_nodes.insert(other_node_id);
                        new_frontier.insert(other_node_id);
                    }
                    // If the edge has not been visited but the other node has been visited
                    // Then the other node cannot be added to the frontier
                }
                if (visited_edges.size() == this->max_segs_per_graph) break;
            }
            if (visited_edges.size() == this->max_segs_per_graph) break;
        }
        if (visited_edges.size() == this->max_segs_per_graph) break;
        frontier = new_frontier;
        depth++;
    }

    std::vector<Tensor> segments;
    for (const std::set<int> &edge : visited_edges) {
        Tensor node_1 = this->nodes[*edge.begin()];
        Tensor node_2 = this->nodes[*edge.rbegin()];
        segments.emplace_back(torch::stack({node_1, node_2}, 0));
    }
    return SegmentGraph(segments);
}

void LaDeDataset::simulateTrajs(SegmentGraph &graph, Tensor &trajs, Tensor &paths, Tensor &traj_lengths, Tensor &path_lengths) {

    trajs = torch::zeros({this->trajs_per_graph, this->traj_len, 2}, DEVICE);
    paths = torch::zeros({this->trajs_per_graph, this->max_path_length, 2}, DEVICE);
    traj_lengths = torch::zeros({trajs_per_graph}, DEVICE);
    path_lengths = torch::zeros({trajs_per_graph}, DEVICE);

    for (int i = 0; i < trajs_per_graph; i++) {
        int path_length;
        graph.getRandomPath(paths[i], path_length);
        path_lengths[i] = path_length;

        int traj_len;
        simulateTraj(paths[i].index({Slice(None, path_length)}), trajs[i], traj_len);
        traj_lengths[i] = traj_len;
    }
}

void LaDeDataset::simulateTraj(Tensor &visiting_nodes, Tensor &traj, int &traj_len) {
    // visiting_node: (N, 2)
    // pairwise_dist: (N-1,)
    Tensor exclude_first = visiting_nodes.index({Slice(1, None, 1)});
    Tensor exclude_last = visiting_nodes.index({Slice(None, -1, 1)});

    Tensor pairwise_dist = torch::norm(exclude_first - exclude_last, 2, 1);
    Tensor total_dist = torch::sum(pairwise_dist);
//    Tensor distances = torch::cumsum(pairwise_dist, 0);
//    distances = torch::cat({torch::zeros({1}, DEVICE), distances}, 0);

    // Simulate the random walk
//    Tensor walked_dist = torch::zeros(1, DEVICE);
//    Tensor current_pos = visiting_nodes[0];
//    std::vector<Tensor> traj({current_pos});
//
//    while ((walked_dist < distances[-1]).item<bool>()) {
//        Tensor next_step_dist = torch::normal(this->traj_step_mean, this->traj_step_std, {1}).to(DEVICE);
//        // the distance cannot be negative
//        next_step_dist = torch::nn::functional::relu(next_step_dist);
//        walked_dist += next_step_dist;
//        // Find the position of the walker along the path
//        for (int i = 0; i < distances.size(0) - 1; i++) {
//            if (((distances[i] <= walked_dist).item<bool>() && (walked_dist < distances[i + 1]).item<bool>())) {
//                current_pos = visiting_nodes[i] + (walked_dist - distances[i]) / pairwise_dist[i] * (visiting_nodes[i + 1] - visiting_nodes[i]);
//                break;
//            }
//        }
//        traj.push_back(current_pos);
//    }
//
//    traj.push_back(visiting_nodes[-1]);
//    Tensor traj_tensor = torch::stack(traj);


    /*
    Let's change the logic. Originally, we simulate the trajectory by walking along the path.
    If the total distance of the path id D, each step moves a distance of N(μ, σ^2).
    Then the number of points to be generated is L = D / N(μ, σ^2).
    So we can first generate L points uniformly along the path, then add noise to jitter the distances.
    Finally apply the traj_noise_std to simulate the GPS noise.
    */
    traj_len = (int) (total_dist / (this->traj_step_mean + torch::randn({1}, DEVICE) * this->traj_step_std)).item<float>();
    // t is the percentage of where the point is on the path
    Tensor t = torch::linspace(0, 1, num_points).to(DEVICE) + torch::randn({num_points}, DEVICE) * this->traj_noise_std;
    // t must be sorted because noise adding may change the order
    t = torch::sort(t).values;
    Tensor traj_tensor = torch::interp(distances, visiting_nodes, t);

    Tensor gps_noise = torch::randn(traj_tensor.sizes(), DEVICE) * this->traj_noise_std;

    int simulated_traj_len = traj_tensor.size(0);
    if (simulated_traj_len >= this->traj_len) {
        traj = (traj_tensor + gps_noise).index({Slice(None, this->traj_len)});
    } else {
        int pad_size = this->traj_len - simulated_traj_len;
        traj = torch::nn::functional::pad(traj_tensor + gps_noise, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    }
}

Tensor LaDeDataset::getHeatmap(Tensor graph_tensor, Tensor traj_tensor, int H, int W) {
    // eliminate duplicate nodes
    Tensor nodes = std::get<0>(torch::unique_dim(graph_tensor.view({-1, 2}), 0));

    Tensor traj_points = traj_tensor.view({-1, 2});     // shape: (N, 2)
    // Eliminate points that are equal to (0, 0)
    Tensor mask_filter = torch::logical_not(torch::all(traj_points == 0, 1));
    traj_points = traj_points.index({mask_filter});
    
    Tensor min_point = std::get<0>(torch::min(torch::cat({traj_points, nodes}, 0), 0, true));
    Tensor max_point = std::get<0>(torch::max(torch::cat({traj_points, nodes}, 0), 0, true));
    traj_points = (traj_points - min_point) / (max_point - min_point);
    nodes = (nodes - min_point) / (max_point - min_point);

    Tensor x_ids = (traj_points.index({"...", 0}) * (W - 1)).toType(torch::kLong);
    Tensor y_ids = (traj_points.index({"...", 1}) * (H - 1)).toType(torch::kLong);
    Tensor heatmap_flat = torch::zeros(H * W, torch::kFloat32).to(DEVICE);
    Tensor flat_indices = y_ids * W + x_ids;
    heatmap_flat.scatter_add_(0, flat_indices, torch::ones_like(flat_indices, torch::kFloat32).to(DEVICE));

    x_ids = (traj_points.index({"...", 0}) * (W - 1)).toType(torch::kLong);
    y_ids = (traj_points.index({"...", 1}) * (H - 1)).toType(torch::kLong);
    Tensor nodemap_flat = torch::zeros(H * W, torch::kFloat32).to(DEVICE);
    flat_indices = y_ids * W + x_ids;
    nodemap_flat.scatter_add_(0, flat_indices, torch::ones_like(flat_indices, torch::kFloat32).to(DEVICE));

    return torch::stack({heatmap_flat, nodemap_flat}, 0).view({2, H, W});
}


