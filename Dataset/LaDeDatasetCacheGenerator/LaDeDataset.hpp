#include "SegmentGraph.hpp"

class LaDeDataset {
public:
    LaDeDataset(std::string path,
                int graph_depth = 5, 
                int N_trajs = 64,
                int max_L_traj = 64,
                int max_segs_per_graph = 64,
                float traj_step_mean = 0.1,
                float traj_noise_std = 0.03);

    void get(Tensor &graph, vector<Tensor> &trajs, vector<Tensor> &paths);

private:

    Tensor nodes;
    Tensor edges;
    Tensor degrees;
    Tensor keynode_mask;
    int graph_depth;
    int N_trajs;
    int max_L_traj;
    int max_segs_per_graph;
    int seg_interp_len = 8;
    float traj_step_mean;
    float traj_step_std;
    float traj_noise_std;

    Tensor candidate_nodes;

    SegmentGraph getGraph();
    void removeBadNodes();
    void simulateTrajs(SegmentGraph &graph, vector<Tensor> &trajs, vector<Tensor> &paths);
    Tensor simulateTraj(Tensor path);
    /*
    Tensor getHeatmap(Tensor graph_tensor, vector<Tensor> traj_tensor, int H, int W);
    */
    void computeKeyNodeMask();
};
    