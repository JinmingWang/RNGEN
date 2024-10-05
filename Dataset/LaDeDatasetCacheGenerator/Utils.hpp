#include<torch/torch.h>
#include<torch/script.h>
#include<c10/core/Scalar.h>
#include<iostream>
#include<string>
#include<vector>
#include<matplot/matplot.h>
#include<random>
#include<valarray>

using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace c10;
using namespace matplot;

#define DEVICE torch::cuda::is_available() ? torch::kCUDA : torch::kCPU