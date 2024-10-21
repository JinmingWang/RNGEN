#include "Utils.hpp"

class SegmentGraph {
public:
    Tensor segments;

    SegmentGraph();
    SegmentGraph(std::vector<Tensor> segment_list);
    SegmentGraph(Tensor segments);

    std::vector<int> getNeighbors(Tensor node, std::vector<int> &exceptions);
    void getRandomPath(Tensor &path, int &path_length);
    Tensor getCenter();
    void normalize();
    void transform(Tensor rotate_angle, Tensor scale_factor);
    Tensor toTensor(int pad_to = -1);
    void draw(std::string color) const;

private:
    int min_path_length = 3;
    int max_path_length = 10;

    std::vector<Tensor> growPath(std::vector<int> &visited_sid, std::vector<Tensor> path);
};