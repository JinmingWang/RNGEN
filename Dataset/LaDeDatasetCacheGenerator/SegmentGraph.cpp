#include "SegmentGraph.hpp"


SegmentGraph::SegmentGraph() {
    this->segments = torch::empty({0, 0, 2}).to(DEVICE);
}

SegmentGraph::SegmentGraph(std::vector<Tensor> segment_list) {
    this->segments = torch::stack(segment_list, 0);
}

SegmentGraph::SegmentGraph(Tensor segments) {
    this->segments = segments;
}

Tensor SegmentGraph::getRandomNeighbour(Tensor node) {
    std::vector<int> neighbors;
    for (int i = 0; i < this->segments.size(0); i++) {
        // first node of the segment == node, this segment is a neighbor
        if (this->segments[i][0].equal(node)) neighbors.emplace_back(i);
        // last node of the segment == node, this segment is a neighbor
        if (this->segments[i][-1].equal(node)) neighbors.emplace_back(i);
    }

    if (neighbors.size() == 0) {
        return torch::empty({0, 2}).to(DEVICE);
    } else {
        int neighbor_id = neighbors[rand() % neighbors.size()];
        return this->segments[neighbor_id];
    }
}

Tensor SegmentGraph::getRandomPath() {
    int num_segs = this->segments.size(0);
    int start_sid = rand() % num_segs;
    return this->growPath(this->segments[start_sid]);
}

Tensor SegmentGraph::growPath(Tensor path) {
    Tensor left_neighbor = this->getNeighbors(path[0]);
    if (left_neighbor.size(0) > 0) {
        if (left_neighbor[0].equal(path[0])) {
            // have to flip left_neighbor then concatenate
            path = torch::cat({left_neighbor.flip(0), path}, 0);
        } else {
            path = torch::cat({left_neighbor, path}, 0);
        }
    }

    Tensor right_neighbor = this->getNeighbors(path[-1]);
    if (right_neighbor.size(0) > 0) {
        if (right_neighbor[0].equal(path[-1])) {
            path = torch::cat({path, right_neighbor}, 0);
        } else {
            path = torch::cat({path, right_neighbor.flip(0)}, 0);
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

void SegmentGraph::transform(Tensor rotate_angle) {
    Tensor center = this->getCenter();

    Tensor cos = torch::cos(rotate_angle);
    Tensor sin = torch::sin(rotate_angle);
    Tensor rot_mat = torch::cat({torch::stack({cos, -sin}), torch::stack({sin, cos})}, 1).to(DEVICE);

    int n_segs = this->segments.size(0);
    this->segments = (torch::matmul(this->segments.view({-1, 2}) - center, rot_mat) + center).view({n_segs, 2, 2});
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