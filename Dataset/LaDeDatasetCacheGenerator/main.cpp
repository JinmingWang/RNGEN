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

int count = 10;
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

void saveTensors(vector<Tensor> tensor, std::string path) {
    std::vector<char> f = torch::pickle_save(tensor);
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
        return configMap;
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

    count = std::stoi(configMap.at("count"));
    path = configMap.at("path");
    graph_depth = std::stoi(configMap.at("graph_depth"));
    trajs_per_graph = std::stoi(configMap.at("trajs_per_graph"));
    max_segs_per_graph = std::stoi(configMap.at("max_segs_per_graph"));
    rotation = configMap.at("rotation") == "true";
    scaling_range = std::stof(configMap.at("scaling_range"));
    traj_step_mean = std::stof(configMap.at("traj_step_mean"));
    traj_step_std = std::stof(configMap.at("traj_step_std"));
    traj_noise_std = std::stof(configMap.at("traj_noise_std"));
    traj_len = std::stoi(configMap.at("traj_len"));
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
    int count_w = std::to_string(count).length();

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