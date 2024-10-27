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
    int count_w = std::to_string(data_count).length();

    // Easy
    LaDeDataset dataset(path, graph_depth, trajs_per_graph, max_segs_per_graph, rotation,
                        scaling_range, traj_step_mean, traj_step_std, traj_noise_std, traj_len);
    vector<Tensor> trajs;
    vector<Tensor> paths;
    vector<Tensor> trajs_lengths;
    vector<Tensor> paths_lengths;
    vector<Tensor> graphs;
    vector<Tensor> graphs_segs;
    vector<Tensor> heatmaps;

    // record start time
    clock_t start = clock();

    for (int i = 0; i < data_count; i++) {
        map<string, Tensor> data = dataset.get();
        trajs.emplace_back(data["trajs"].to(torch::kCPU));
        paths.emplace_back(data["paths"].to(torch::kCPU));
        trajs_lengths.emplace_back(data["traj_lengths"].to(torch::kCPU));
        paths_lengths.emplace_back(data["path_lengths"].to(torch::kCPU));
        graphs_segs.emplace_back(data["num_segments"].to(torch::kCPU));
        graphs.emplace_back(data["graph"].to(torch::kCPU));
        heatmaps.emplace_back(data["heatmap"].to(torch::kCPU));

        double elapsed = double(clock() - start) / CLOCKS_PER_SEC;
        double estimated_total = elapsed / (i + 1) * data_count;
        double estimated_remaining = estimated_total - elapsed;

        std::cout << "\rGenerating " << std::setw(count_w) << i + 1 << " / " << data_count << " trajs." << std::setw(2)
                  << " Elapsed: " << int(elapsed/60) << "m" << int(elapsed) % 60 << "s" << std::setw(2) << " Remaining: "
                    << int(estimated_remaining/60) << "m" << int(estimated_remaining) % 60 << "s    " << std::flush;
    }
    std::cout << std::endl;

    saveTensors(trajs, "./CACHE/trajs.pth");          // Each element: (64, 128, 2)
    saveTensors(paths, "./CACHE/paths.pth");          // Each element: (64, 11, 2)
    saveTensors(trajs_lengths, "./CACHE/trajs_lengths.pth");  // Each element: (64)
    saveTensors(paths_lengths, "./CACHE/paths_lengths.pth");  // Each element: (64)
    saveTensors(graphs_segs, "./CACHE/segs_count.pth");  // Each element: (64)
    saveTensors(graphs, "./CACHE/graphs.pth");        // Each element: (64, 2, 2)
    saveTensors(heatmaps, "./CACHE/heatmaps.pth");    // Each element: (2, 64, 64)
    
    return 0;
}