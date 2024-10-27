#include "SegmentGraph.hpp"


SegmentGraph::SegmentGraph() {
    this->segments = torch::empty({0, 2, 2}).to(DEVICE);
}

SegmentGraph::SegmentGraph(std::vector<Tensor> segment_list) {
    this->segments = torch::stack(segment_list, 0);
}

SegmentGraph::SegmentGraph(Tensor segments) {
    this->segments = segments;
}

std::vector<int> SegmentGraph::getNeighbors(Tensor node, std::vector<int> &exceptions) {
    std::vector<int> neighbors;
    for (int i = 0; i < this->segments.size(0); i++) {
        // If the segment is in the exceptions list, skip it
        if (std::find(exceptions.begin(), exceptions.end(), i) != exceptions.end()) {
            continue;
        }
        // If the segment contains the node, add it to the neighbors list
        if (this->segments[i][0].equal(node) || this->segments[i][1].equal(node)) {
            neighbors.emplace_back(i);
        }
    }
    return neighbors;
}

void SegmentGraph::getRandomPath(Tensor &path, int &path_length) {
    int num_segments = this->segments.size(0);
    int start_segment_id = rand() % num_segments;

    std::vector<int> visited_sid(1, start_segment_id);
    std::vector<Tensor> init_path({this->segments[start_segment_id][0], this->segments[start_segment_id][1]});
    std::vector<Tensor> result_path = this->growPath(visited_sid, init_path);

    path_length = result_path.size();
    path = torch::stack(result_path, 0);    // (path_length, 2)
    int path_len = this->max_path_length - path_length;
    if (path_len > 0) {
        auto pad_option = torch::nn::functional::PadFuncOptions({0, 0, 0, path_len});
        path = torch::nn::functional::pad(path, pad_option);
    }
}

std::vector<Tensor> SegmentGraph::growPath(std::vector<int> &visited_sid, std::vector<Tensor> path) {
    std::vector<int> left_neighbors = this->getNeighbors(path[0], visited_sid);
    if (left_neighbors.size() > 0) {
        // Randomly select a neighbor
        int left_neighbor = left_neighbors[rand() % left_neighbors.size()];
        visited_sid.push_back(left_neighbor);
        if (this->segments[left_neighbor][0].equal(path[0])) {
            path.insert(path.begin(), this->segments[left_neighbor][1]);
        } else {
            path.insert(path.begin(), this->segments[left_neighbor][0]);
        }
    }

    std::vector<int> right_neighbors = this->getNeighbors(path[path.size() - 1], visited_sid);
    if (right_neighbors.size() > 0) {
        // Randomly select a neighbor
        int right_neighbor = right_neighbors[rand() % right_neighbors.size()];
        visited_sid.push_back(right_neighbor);
        if (this->segments[right_neighbor][0].equal(path[path.size() - 1])) {
            path.push_back(this->segments[right_neighbor][1]);
        } else {
            path.push_back(this->segments[right_neighbor][0]);
        }
    }

    if ((left_neighbors.size() > 0 || right_neighbors.size() > 0) && path.size() < this->max_path_length) {
        return this->growPath(visited_sid, path);
    } else {
        return path;
    }
}

Tensor SegmentGraph::getCenter() {
    return torch::mean(this->segments.view({-1, 2}), 0, true);
}

void SegmentGraph::normalize() {
    Tensor nodes_tensor = this->segments.view({-1, 2});

    Tensor min_node = std::get<0>(torch::min(nodes_tensor, 0, true));   // (1, 2)
    Tensor max_node = std::get<0>(torch::max(nodes_tensor, 0, true));   // (1, 2)
    Tensor node_range = max_node - min_node;    // (1, 2)

    // segments: (N, 2, 2)
    int n_segs = this->segments.size(0);
    this->segments = ((this->segments.view({-1, 2}) - min_node) / node_range * 6.0 - 3.0).view({n_segs, 2, 2});
}

void SegmentGraph::transform(Tensor rotate_angle, Tensor scale_factor) {
    Tensor center = this->getCenter();

    Tensor cos = torch::cos(rotate_angle);
    Tensor sin = torch::sin(rotate_angle);
    Tensor rot_mat = torch::cat({torch::stack({cos, -sin}), torch::stack({sin, cos})}, 1).to(DEVICE);

    int n_segs = this->segments.size(0);
    this->segments = (torch::matmul(this->segments.view({-1, 2}) - center, rot_mat) *scale_factor + center).view({n_segs, 2, 2});
}

Tensor SegmentGraph::toTensor(int pad_to) {
    if (pad_to <= 0){
        return this->segments;
    } else {
        auto pad_option = torch::nn::functional::PadFuncOptions({0, 0, 0, 0, 0, pad_to - this->segments.size(0)});
        return torch::nn::functional::pad(this->segments, pad_option);
    }

}

void SegmentGraph::draw(std::string color) const {
    for (int i = 0; i < this->segments.size(0); i++) {
        std::vector<double> x = {this->segments[i][0][0].item<double>(), this->segments[i][1][0].item<double>()};
        std::vector<double> y = {this->segments[i][0][1].item<double>(), this->segments[i][1][1].item<double>()};
        matplot::plot(x, y)->color({0.0f, 0.0f, 0.0f});
        matplot::hold(on);
    }
    matplot::hold(off);
}