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
                         int N_trajs,
                         int max_L_traj,
                         int max_segs_per_graph,
                         float traj_step_mean,
                         float traj_noise_std) {

    torch::IValue x = torch::pickle_load(getBytes(path));
    auto generic_list = x.toList();

    this->nodes = generic_list.get(0).toTensor().to(DEVICE);
    this->edges = generic_list.get(1).toTensor().to(DEVICE);
    this->degrees = generic_list.get(2).toTensor().to(DEVICE);

    this->computeKeyNodeMask();

    // removeBadNodes();

    this->graph_depth = graph_depth;
    this->N_trajs = N_trajs;
    this->max_L_traj = max_L_traj;
    this->max_segs_per_graph = max_segs_per_graph;
    this->traj_step_mean = traj_step_mean;
    this->traj_noise_std = traj_noise_std;

    this->candidate_nodes = torch::where(this->degrees > 2)[0];
    cout << "Number of candidate nodes: " << this->candidate_nodes.size(0) << endl;
}


void LaDeDataset::removeBadNodes() {
    // bad node definition: For node B, if it has only two neighbors A and C
    // and the segment ABC is almost a straight line, then B is meaningless
    // In this case, remove node B from nodes, degrees and edges
    // Remove neighbor B from edges of A and C
    // If A and C are already connected, reduce their degree by 1
    // If A and C are not connected, connect them
    cout << "Removing Bad points" << endl;
    int points_removed = 0;
    // Iterate over all nodes in form A-B-C, node B has degree 2
    for (int B_id = 0; B_id < this->nodes.size(0); B_id++) {
        if (this->degrees[B_id].item<int>() == 2) {
            // Get the id of node A and C
            int A_id = this->edges[B_id][0].item<int>();
            int C_id = this->edges[B_id][1].item<int>();

            // Get node A, B, C
            Tensor node_A = this->nodes[A_id];
            Tensor node_B = this->nodes[B_id];
            Tensor node_C = this->nodes[C_id];

            // Calculate the angle between BA and BC
            Tensor vec_1 = node_B - node_A;
            Tensor vec_2 = node_B - node_C;
            Tensor cos_theta = torch::dot(vec_1, vec_2) / (torch::norm(vec_1) * torch::norm(vec_2));
            cos_theta = torch::clamp(cos_theta, -1.0, 1.0);
            float degrees = torch::acos(cos_theta).item<float>() * 180.0 / M_PI;

            // Check if ABC is almost a straight line
            // Then the angle will be close to 180
            if (degrees > 170) {
                points_removed ++;
                // Not good to remove elements, as it involves reassigning the index
                // Remove all edges related to B by setting all edges to -1, set degree of B to 0
                this->edges[B_id].fill_(-1);
                this->degrees[B_id] = 0;

                // Remove B from the neighbors of A
                // [... , B , ... , last , ...] -> [... , last , ... , -1 , ...]
                auto B_pos = torch::where(this->edges[A_id] == B_id)[0].item<int>();
                int last_pos = this->degrees[A_id].item<int>() - 1;
                this->edges.index({A_id, B_pos}) = this->edges.index({A_id, last_pos});
                this->edges.index({A_id, last_pos}) = -1;
                this->degrees[A_id] -= 1;

                // Move the last element to the position of B, then set the last element to 0
                B_pos = torch::where(this->edges[C_id] == B_id)[0].item<int>();
                last_pos = this->degrees[C_id].item<int>() - 1;
                this->edges.index({C_id, B_pos}) = this->edges.index({C_id, last_pos});
                this->edges.index({C_id, last_pos}) = -1;
                this->degrees[C_id] -= 1;

                // Because A and C are originally connected by B, after removing AB and BC, there shoud be AC
                // If A and C are not connected, then connect them
                if (torch::sum(this->edges[A_id] == C_id).item<int>() == 0) {
                    this->edges[A_id][this->degrees[A_id]] = C_id;
                    this->edges[C_id][this->degrees[C_id]] = A_id;
                    this->degrees[A_id] += 1;
                    this->degrees[C_id] += 1;
                }
            }
        }
    }
    // If we have eliminated some bad points, say we removed B in ABCD
    // Now we have ACD, but C in ACD can be a new bad point
    // However, ACD is not checked in the first round
    if (points_removed > 0) {
        cout << points_removed << " points removed, continue" << endl;
        removeBadNodes();
    }
}


void LaDeDataset::get(Tensor &segments, vector<Tensor> &trajs, vector<Tensor> &paths) {
    SegmentGraph graph = this->getGraph();  // graph.segments: (N_segs, 8, 2)
    this->simulateTrajs(graph, trajs, paths);

    for (int i = 0; i < this->N_trajs; i ++){
        trajs[i] = trajs[i].to(torch::kCPU);
        paths[i] = paths[i].to(torch::kCPU);
    }

    segments = graph.segments.to(torch::kCPU);
}


SegmentGraph LaDeDataset::getGraph() {
    // Randomly select a node as the starting point
    int random_idx = torch::randint(0, this->candidate_nodes.size(0), {1}).item<int>();
    int node_idx = this->candidate_nodes[random_idx].item<int>();

    // keep track of the edges that have been visited
    std::set<std::set<int>> visited_edges;
    // A hyper_seg is a path connecting two key nodes
    std::set<std::vector<int>> hyper_segs;
    // frontier is used to keep track of the key nodes that are being visited
    std::set<int> frontier({node_idx});
    int depth = 0;
    while (frontier.size() > 0 && depth < this->graph_depth) {
        std::set<int> new_frontier;
        // Iterate over the nodes in the frontier
        for (auto &node_id : frontier) {
            // Iterate over the neighbors
            for (int i = 0; i < this->degrees[node_id].item<int>(); i++) {
                // Try to reach out to another key node, the first step is to reach for a neighbor
                // if the edge has been visited, then skip
                int reaching_node_id = this->edges[node_id][i].item<int>();
                std::set<int> edge({node_id, reaching_node_id});
                if (visited_edges.find(edge) != visited_edges.end()) continue;

                // This edge has not been visited, which also means this hyper_seg has not been visited
                std::vector<int> hyper_seg = {node_id, reaching_node_id};
                visited_edges.insert(edge);

                // Reach out to another key node
                while (! this->keynode_mask[reaching_node_id].item<bool>()) {
                    // The reaching node is not a key node, go one step further
                    // The reaching node must have two neighbors
                    // One is the way we come from, the other is the way we go
                    if (this->edges[reaching_node_id][0].item<int>() == hyper_seg[hyper_seg.size() - 2]) {
                        // Then edges[reaching_node_id][0] is the way we come from, reach for edges[reaching_node_id][1]
                        hyper_seg.push_back(this->edges[reaching_node_id][1].item<int>());
                    } else {
                        hyper_seg.push_back(this->edges[reaching_node_id][0].item<int>());
                    }
                    // Update the reaching node
                    reaching_node_id = hyper_seg[hyper_seg.size() - 1];
                    // Update the edge
                    edge = {hyper_seg[hyper_seg.size() - 2], hyper_seg[hyper_seg.size() - 1]};
                    visited_edges.insert(edge);
                }

                // hyper_seg could be too long, we need to truncate it
                bool too_long = false;
                while (hyper_seg.size() > this->seg_interp_len) {
                    too_long = true;
                    hyper_seg.pop_back();
                }

                // Now hyper_segs = [node_id, ..., reaching_node_id]
                hyper_segs.insert(hyper_seg);
                // Add the reaching node to the new frontier if it is a key node
                // too_long is True, then the reaching node is not a key node
                if (! too_long) frontier.insert(reaching_node_id);

                if (hyper_segs.size() == this->max_segs_per_graph) break;
            }
            if (hyper_segs.size() == this->max_segs_per_graph) break;
        }
        if (hyper_segs.size() == this->max_segs_per_graph) break;
        frontier = new_frontier;
        depth++;
    }

    std::vector<Tensor> hyper_segs_tensor;
    for (auto &hyper_seg : hyper_segs) {
        std::vector<Tensor> node_arr;
        for (auto &node_id : hyper_seg) {
            node_arr.emplace_back(this->nodes[node_id]);
        }
        Tensor nodes = torch::stack(node_arr, 0);   // (N, 2)

        // Do interpolation to make the hyper_seg longer
        Tensor pairwise_dist = torch::norm(nodes.index({Slice(1, None)}) - nodes.index({Slice(None, -1)}), 2, 1);
        Tensor reach_dist = torch::nn::functional::pad(torch::cumsum(pairwise_dist, 0), torch::nn::functional::PadFuncOptions({1, 0}));

        // t is where the point is on the path
        Tensor t = torch::linspace(0, reach_dist[-1].item<float>(), this->seg_interp_len).to(DEVICE);

        // For i-th interpolation point, find the left and right nodes
        Tensor right_ids = torch::searchsorted(reach_dist, t);  // Is there a right=True parameter?
        Tensor left_ids = right_ids - 1;

        // interp = left_item * right_ratio + right_item * left_ratio
        Tensor left_ratio = (t - reach_dist.index({left_ids})) / (reach_dist.index({right_ids}) - reach_dist.index({left_ids}));
        Tensor right_ratio = 1 - left_ratio;

        Tensor interp_points = nodes.index({left_ids}) * right_ratio.unsqueeze(1) + nodes.index({right_ids}) * left_ratio.unsqueeze(1);
        hyper_segs_tensor.emplace_back(interp_points);
    }
    return SegmentGraph(hyper_segs_tensor);
}

void LaDeDataset::simulateTrajs(SegmentGraph &graph, vector<Tensor> &trajs, vector<Tensor> &paths) {
    for (int i = 0; i < this->N_trajs; i++) {
        Tensor path = graph.getRandomPath();
        paths.emplace_back(path);
        trajs.emplace_back(simulateTraj(path));
    }
}

Tensor LaDeDataset::simulateTraj(Tensor path) {
    // path: (N, 2)
    // pairwise_dist: (N-1,)
    Tensor exclude_first = path.index({Slice(1, None, 1)});
    Tensor exclude_last = path.index({Slice(None, -1, 1)});

    Tensor pairwise_dist = torch::norm(exclude_first - exclude_last, 2, 1);
    Tensor distances = torch::cumsum(pairwise_dist, 0);
    distances = torch::cat({torch::zeros({1}, DEVICE), distances}, 0);

    int num_points = (int) (distances[-1] / this->traj_step_mean + torch::randn({1}, DEVICE)).item<float>();
    if (num_points < 3) num_points = 3;
    if (num_points > this->max_L_traj) num_points = this->max_L_traj;
    // t is where the point is on the path
    // Tensor t = torch::linspace(0, distances[-1].item<float>(), num_points).to(DEVICE) + torch::randn({num_points}, DEVICE) * this->traj_step_std;
    // t = torch::nn::functional::hardtanh(t, torch::nn::HardtanhOptions().min_val(0.0).max_val(distances[-1].item<float>()));
    Tensor t = torch::rand({num_points}).to(DEVICE) * distances[-1].item<float>();
    // t must be sorted because noise adding may change the order
    t = std::get<0>(torch::sort(t));

//    // Iter all points
//    for (int i = 0; i < num_points; i++) {
//        Tensor current_t = t[i];
//        // Iter all segments
//        for (int j = 0; j < distances.size(0) - 1; j++) {
//            if (((distances[j] <= current_t).item<bool>() && (current_t < distances[j + 1]).item<bool>())) {
//                traj_tensor[i] = visiting_nodes[j] + (current_t - distances[j]) / pairwise_dist[j] * (visiting_nodes[j + 1] - visiting_nodes[j]);
//                break;
//            }
//        }
//    }

    /*
    inline at::Tensor at::searchsorted(
    const at::Tensor &sorted_sequence,
    const at::Scalar &self,
    bool out_int32 = false,
    bool right = false,
    ::std::optional<c10::string_view> side = ::std::nullopt,
    const ::std::optional<at::Tensor> &sorter = {})
    */
    Tensor right_ids = torch::searchsorted(distances, t, true, true);
    Tensor left_ids = right_ids - 1;
    Tensor left_ratio = (t - distances.index({left_ids})) / pairwise_dist.index({left_ids});
    Tensor right_ratio = 1 - left_ratio;
    Tensor traj = path.index({left_ids}) * right_ratio.unsqueeze(1) + path.index({right_ids}) * left_ratio.unsqueeze(1);

    Tensor gps_noise = torch::randn_like(traj) * this->traj_noise_std;

    return traj + gps_noise;
}

/*
Tensor LaDeDataset::getHeatmap(Tensor graph_tensor, vector<Tensor> traj_tensor, int H, int W) {
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
*/

void LaDeDataset::computeKeyNodeMask(){
    auto whole_dim = Slice();

    Tensor left_id = this->edges.index({whole_dim, 0});
    Tensor l_node = this->nodes.index({left_id});

    Tensor right_id = this->edges.index({whole_dim, 1});
    Tensor r_node = this->nodes.index({right_id});

    Tensor BA = l_node - this->nodes;
    Tensor BC = r_node - this->nodes;

    Tensor angle_rad = torch::atan2(BC.index({whole_dim, 1}), BC.index({whole_dim, 0})) - torch::atan2(BA.index({whole_dim, 1}), BA.index({whole_dim, 0}));
    Tensor angle_deg = torch::abs(angle_rad * 180 / 3.14159265359);
    angle_deg = torch::min(angle_deg, 360.0 - angle_deg);

    Tensor less_than_90 = angle_deg < 90;

    this->keynode_mask = (this->degrees == 1) | (this->degrees >= 3) | ((this->degrees == 2) & less_than_90);
}

