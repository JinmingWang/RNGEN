#include "SegmentGraph.hpp"

class LaDeDataset {
public:
    LaDeDataset(std::string path,
                int graph_depth = 5, 
                int trajs_per_graph = 64,
                int max_segs_per_graph = 64,
                bool rotation = true, 
                float scaling_range = 0.2, 
                float traj_step_mean = 0.1,
                float traj_step_std = 0.03,
                float traj_noise_std = 0.03,
                int traj_len = 128);

    std::tuple<Tensor, Tensor, Tensor, Tensor> get();

private:

    Tensor nodes;
    Tensor edges;
    Tensor degrees;
    int graph_depth;
    int trajs_per_graph;
    int max_segs_per_graph;
    bool rotation;
    float scaling_range;
    float traj_step_mean;
    float traj_step_std;
    float traj_noise_std;
    int traj_len;

    Tensor candidate_nodes;

    SegmentGraph getGraph();
    void simulateTrajs(SegmentGraph &graph, Tensor &trajs, Tensor &paths);
    Tensor simulateTraj(Tensor &visiting_nodes);
    Tensor getHeatmap(Tensor graph_tensor, Tensor traj_tensor, int H, int W);
};
    