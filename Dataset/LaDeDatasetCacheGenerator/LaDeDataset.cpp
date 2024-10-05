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
}

std::tuple<Tensor, Tensor, Tensor, Tensor> LaDeDataset::get() {
    SegmentGraph graph = this->getGraph();
    // Tensor angle = this->rotation ? torch::rand({1}, DEVICE) * 2.0 * M_PI : torch::zeros({1}, DEVICE);
    // Tensor scale = torch::rand({1}, DEVICE) * (this->scaling_range * 2.0) - this->scaling_range + 1.0;
    // graph.transform(angle, scale);
    graph.normalize();

    // graph.draw("#000000");
    // matplot::show();

    Tensor trajs;
    Tensor paths;
    this->simulateTrajs(graph, trajs, paths);

    Tensor graph_tensor = graph.toTensor(this->max_segs_per_graph);

    Tensor heatmap = this->getHeatmap(graph_tensor, trajs, 128, 128);

    return std::make_tuple(trajs, paths, graph_tensor, heatmap);
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

void LaDeDataset::simulateTrajs(SegmentGraph &graph, Tensor &trajs, Tensor &paths) {

    int num_segments = graph.segments.size(0);

    std::vector<Tensor> traj_list;
    std::vector<Tensor> path_list;
    int max_path_len = this->graph_depth * 2 + 1;

    for (int i = 0; i < trajs_per_graph; i++) {
        int start_segment_id = rand() % num_segments;
        std::vector<Tensor> path = graph.getRandomPath(start_segment_id);
        Tensor visiting_nodes = torch::stack(path);     // (N, 2)
        traj_list.emplace_back(this->simulateTraj(visiting_nodes));
        auto pad_option = torch::nn::functional::PadFuncOptions({0, 0, 0, max_path_len - visiting_nodes.size(0)});
        path_list.emplace_back(torch::nn::functional::pad(visiting_nodes, pad_option));
    }

    trajs = torch::stack(traj_list, 0);
    paths = torch::stack(path_list, 0);
}

Tensor LaDeDataset::simulateTraj(Tensor &visiting_nodes) {
    // visiting_node: (N, 2)
    // pairwise_dist: (N-1,)
    Tensor exclude_first = visiting_nodes.index({Slice(1, None, 1)});
    Tensor exclude_last = visiting_nodes.index({Slice(None, -1, 1)});

    Tensor pairwise_dist = torch::norm(exclude_first - exclude_last, 2, 1);
    Tensor distances = torch::cumsum(pairwise_dist, 0);
    distances = torch::cat({torch::zeros({1}, DEVICE), distances}, 0);

    // Simulate the random walk
    Tensor walked_dist = torch::zeros(1, DEVICE);
    Tensor current_pos = visiting_nodes[0];
    std::vector<Tensor> traj({current_pos});

    while ((walked_dist < distances[-1]).item<bool>()) {
        Tensor next_step_dist = torch::normal(this->traj_step_mean, this->traj_step_std, {1}).to(DEVICE);
        walked_dist += next_step_dist;
        // Find the position of the walker along the path
        for (int i = 0; i < distances.size(0) - 1; i++) {
            if (((distances[i] <= walked_dist).item<bool>() && (walked_dist < distances[i + 1]).item<bool>())) {
                current_pos = visiting_nodes[i] + (walked_dist - distances[i]) / pairwise_dist[i] * (visiting_nodes[i + 1] - visiting_nodes[i]);
                break;
            }
        }
        traj.push_back(current_pos);
    }

    traj.push_back(visiting_nodes[-1]);
    Tensor traj_tensor = torch::stack(traj);

    Tensor gps_noise = torch::randn(traj_tensor.sizes(), DEVICE) * this->traj_noise_std;

    int simulated_traj_len = traj_tensor.size(0);
    if (simulated_traj_len >= this->traj_len) {
        return (traj_tensor + gps_noise).index({Slice(None, this->traj_len)});
    } else {
        int pad_size = this->traj_len - simulated_traj_len;
        return torch::nn::functional::pad(traj_tensor + gps_noise, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
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


