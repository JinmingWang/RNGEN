#include "LaDeDataset.hpp"
#include <time.h>

/* torch::pickle_save() supporting data container:
 *
 * C++              Python
 * ---              ------
 * std::vector      list
 * std::tuple       tuple
 * c10::Dict        dict
 * torch::Tensor    torch.Tensor
 */

void saveTensors(vector<Tensor> tensor, std::string path) {
    std::vector<char> f = torch::pickle_save(tensor);
    std::ofstream out(path, std::ios::out | std::ios::binary);
    out.write(f.data(), f.size());
    out.close();
}


int main(int argc, char const *argv[]) {
    int count = 10;
    if (argc == 2) {
        count = std::stoi(argv[1]);
    } else if (argc > 2) { 
        std::cerr << "Usage: " << argv[0] << " [count]" << std::endl;
        return 1;
    } else {
        std::cout << "Using default count: " << count << std::endl;
    }

    // Get the number of digits in the count
    int count_w = std::to_string(count).length();

    std::string path = "processed.pt";
    int graph_depth = 5;
    int trajs_per_graph = 64;
    int max_segs_per_graph = 64;
    bool rotation = true;
    float scaling_range = 0.2;
    float traj_step_mean = 0.3;
    float traj_step_std = 0.15;
    float traj_noise_std = 0.07;
    float traj_len = 64;

    // Easy
    LaDeDataset dataset(path, graph_depth, trajs_per_graph, max_segs_per_graph, rotation,
                        scaling_range, traj_step_mean, traj_step_std, traj_noise_std, traj_len);
    vector<Tensor> trajs;
    vector<Tensor> paths;
    vector<Tensor> graphs;
    vector<Tensor> heatmaps;

    // record start time
    clock_t start = clock();

    for (int i = 0; i < count; i++) {
        auto [traj, path, graph, heatmap] = dataset.get();
        trajs.emplace_back(traj.to(torch::kCPU));
        paths.emplace_back(path.to(torch::kCPU));
        graphs.emplace_back(graph.to(torch::kCPU));
        heatmaps.emplace_back(heatmap.to(torch::kCPU));

        double elapsed = double(clock() - start) / CLOCKS_PER_SEC;
        double estimated_total = elapsed / (i + 1) * count;
        double estimated_remaining = estimated_total - elapsed;

        std::cout << "\rGenerating " << std::setw(count_w) << i + 1 << " / " << count << " trajs." << std::setw(2)
                  << " Elapsed: " << int(elapsed/60) << "m" << int(elapsed) % 60 << "s" << std::setw(2) << " Remaining: "
                    << int(estimated_remaining/60) << "m" << int(estimated_remaining) % 60 << "s    " << std::flush;
    }
    std::cout << std::endl;

    saveTensors(trajs, "./trajs.pth");          // Each element: (64, 128, 2)
    saveTensors(paths, "./paths.pth");          // Each element: (64, 11, 2)
    saveTensors(graphs, "./graphs.pth");        // Each element: (64, 2, 2)
    saveTensors(heatmaps, "./heatmaps.pth");    // Each element: (2, 64, 64)
    
    return 0;
}