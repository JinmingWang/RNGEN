#include "Utils.hpp"

class SegmentGraph {
public:
    Tensor segments;
    int min_path_length = 2;
    int max_path_length = 32;

    SegmentGraph();
    SegmentGraph(std::vector<Tensor> segment_list);
    SegmentGraph(Tensor segments);

    int getRandomNeighbour(Tensor node, int this_sid);
    Tensor getRandomPath();
    Tensor getCenter();
    void normalize();
    void transform(Tensor rotate_angle);
    void draw(std::string color) const;

private:

    Tensor growPath(Tensor path, int this_sid);
};