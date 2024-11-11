#include "LaDeDataset.hpp"
#include <time.h>
#include <fstream>
#include <sstream>
#include <unordered_map>

/* torch::pickle_save() supporting data container:
 *
 * C++              Python
 * ---              ------
 * std::vector      list
 * std::tuple       tuple
 * c10::Dict        dict
 * torch::Tensor    torch.Tensor
 */

int data_count = 10;
std::string path = "/home/jimmy/Data/LaDe/processed_roads_Shanghai.pt";
int graph_depth = 5;
int N_trajs = 64;
int max_segs_per_graph = 64;
float traj_step_mean = 0.3;
float traj_step_std = 0.15;
float traj_noise_std = 0.07;

template <typename T>
void saveTensors(T tensors, std::string path) {
    std::vector<char> f = torch::pickle_save(tensors);
    std::ofstream out(path, std::ios::out | std::ios::binary);
    out.write(f.data(), f.size());
    out.close();
}


// Function to parse the configuration file
void parseConfigFile(const std::string& filename) {
    std::unordered_map<std::string, std::string> configMap;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            configMap[key] = value;
        }
    }

    file.close();

    data_count = std::stoi(configMap.at("data_count"));
    path = configMap.at("path");
    graph_depth = std::stoi(configMap.at("graph_depth"));
    N_trajs = std::stoi(configMap.at("N_trajs"));
    max_segs_per_graph = std::stoi(configMap.at("max_segs_per_graph"));
    scaling_range = std::stof(configMap.at("scaling_range"));
    traj_step_mean = std::stof(configMap.at("traj_step_mean"));
    traj_step_std = std::stof(configMap.at("traj_step_std"));
    traj_noise_std = std::stof(configMap.at("traj_noise_std"));
}


int main(int argc, char const *argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_args_file>" << std::endl;
        return 1;
    }

    std::string configFile = argv[1];

    // Parse the config file and assign values
    parseConfigFile(configFile);

    // Get the number of digits in the count
    int count_w = std::to_string(data_count).length();

    // Easy
    LaDeDataset dataset(path, graph_depth, N_trajs, max_segs_per_graph,
                        traj_step_mean, traj_step_std, traj_noise_std);
    vector<vector<Tensor>> all_trajs;
    vector<vector<Tensor>> all_paths;
    vector<Tensor> all_segs;
    //vector<Tensor> heatmaps;

    // record start time
    clock_t start = clock();

    for (int i = 0; i < data_count; i++) {
        vector<Tensor> trajs;   // (N_trajs, L_traj, 2)
        vector<Tensor> paths;   // (N_trajs, L_path, 2)
        Tensor segs;    // (N_segs, 8, 2)
        dataset.get(segs, trajs, paths);

        all_trajs.emplace_back(trajs.to(torch::kCPU));
        all_paths.emplace_back(paths.to(torch::kCPU));
        all_segs.emplace_back(segs.to(torch::kCPU));

        double elapsed = double(clock() - start) / CLOCKS_PER_SEC;
        double estimated_total = elapsed / (i + 1) * data_count;
        double estimated_remaining = estimated_total - elapsed;

        std::cout << "\rGenerating " << std::setw(count_w) << i + 1 << " / " << data_count << " trajs." << std::setw(2)
                  << " Elapsed: " << int(elapsed/60) << "m" << int(elapsed) % 60 << "s" << std::setw(2) << " Remaining: "
                    << int(estimated_remaining/60) << "m" << int(estimated_remaining) % 60 << "s    " << std::flush;
    }
    std::cout << std::endl;

    saveTensors<vector<vector<Tensor>>>(all_trajs, "./CACHE/trajs.pth");
    saveTensors<vector<vector<Tensor>>>(all_paths, "./CACHE/paths.pth");
    saveTensors<vector<Tensor>>(all_segs, "./CACHE/segs.pth");
    // saveTensors(heatmaps, "./CACHE/heatmaps.pth");    // Each element: (2, 64, 64)
    
    return 0;
}