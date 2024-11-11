#include "Utils.hpp"

class SegmentGraph {
public:
    Tensor segments;
    int min_path_length = 2;
    int max_path_length = 8;

    SegmentGraph();
    SegmentGraph(std::vector<Tensor> segment_list);
    SegmentGraph(Tensor segments);

    Tensor getRandomNeighbour(Tensor node);
    Tensor getRandomPath();
    Tensor getCenter();
    void normalize();
    void transform(Tensor rotate_angle);
    void draw(std::string color) const;

private:

    std::vector<Tensor> growPath(std::vector<Tensor> path);
};