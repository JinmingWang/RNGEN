#include "Utils.hpp"

class SegmentGraph {
public:
    Tensor segments;

    SegmentGraph();
    SegmentGraph(std::vector<Tensor> segment_list);
    SegmentGraph(Tensor segments);

    std::vector<int> getNeighbors(Tensor node, std::vector<int> &exceptions);
    std::vector<Tensor> getRandomPath(int start_segment_id);
    Tensor getCenter();
    void normalize();
    void transform(Tensor rotate_angle, Tensor scale_factor);
    Tensor toTensor(int pad_to = -1);
    void draw(std::string color) const;

private:
    std::vector<Tensor> growPath(std::vector<int> &visited_sid, std::vector<Tensor> path);
};