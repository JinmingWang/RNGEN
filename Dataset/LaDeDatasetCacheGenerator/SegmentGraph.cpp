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

int SegmentGraph::getRandomNeighbour(Tensor node, int this_sid) {
    // node: (2)
    std::vector<int> neighbors;
    for (int i = 0; i < this->segments.size(0); i++) {
        if (i == this_sid) continue;
        // first node of the segment == node, this segment is a neighbor
        if (this->segments[i][0].equal(node)) neighbors.emplace_back(i);
        // last node of the segment == node, this segment is a neighbor
        if (this->segments[i][-1].equal(node)) neighbors.emplace_back(i);
    }

    if (neighbors.size() == 0) {
        return -1;
    } else {
        return neighbors[rand() % neighbors.size()];
    }
}

Tensor SegmentGraph::getRandomPath() {
    int start_sid = rand() % this->segments.size(0);
    return this->growPath(this->segments[start_sid], start_sid);
}

Tensor SegmentGraph::growPath(Tensor path, int this_sid) {
    int left_neighbor_id = this->getRandomNeighbour(path[0], this_sid);
    int right_neighbor_id = this->getRandomNeighbour(path[-1], this_sid);

    if (path.size(0) >= this->max_path_length) {
        return path;
    }

    // Cannot grow anymore
    if (left_neighbor_id == -1 && right_neighbor_id == -1){
        return path;
    }

    bool grow_left;
    if (left_neighbor_id == -1) {   // cannot grow left, then grow right
        grow_left = false;
    } else if (right_neighbor_id == -1) {   // cannot grow right, then grow left
        grow_left = true;
    } else {    // can grow both side, randomly choose one
        auto gen = std::bind(std::uniform_int_distribution<>(0,1),std::default_random_engine());
        grow_left = gen();
    }

    if (grow_left) {
        Tensor left_neighbor = this->segments[left_neighbor_id];
        if (left_neighbor[0].equal(path[0])) {
            // have to flip left_neighbor then concatenate
            path = torch::cat({left_neighbor.flip(0), path}, 0);
        } else {
            path = torch::cat({left_neighbor, path}, 0);
        }
        this->growPath(path, left_neighbor_id);
    } else {
        Tensor right_neighbor = this->segments[right_neighbor_id];
        if (right_neighbor[0].equal(path[-1])) {
            path = torch::cat({path, right_neighbor}, 0);
        } else {
            path = torch::cat({path, right_neighbor.flip(0)}, 0);
        }
        this->growPath(path, right_neighbor_id);
    }

    return path;
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