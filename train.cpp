#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include "Net.h"
#include "Config.h"



torch::Tensor read_data(const std::string& loc)
{
    std::ofstream dataFile(loc);
    if(!dataFile.is_open())
        throw std::runtime_error("Could not open file");
    std::string line;
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        printf("*****************%s", ss);

     //    torch::Tensor tensor = ...

    // Here you need to get your data.

    return tensor
};

class IRIS : public torch::data::Dataset<MyDataset>
{
private:
    torch::Tensor states_, labels_;

public:
    explicit MyDataset(const std::string& loc){
        //read in c++ and set states and labels.
        read_data(loc);
    };

    torch::data::Example<> get(size_t index) override;
};

torch::data::Example<> MyDataset::get(size_t index)
{
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    return {states_[index], labels_[index]};
}

void train(Config config) {
    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type)

    Net model;
    model.to(device)

    auto dataset = torch::data::datasets::IRIS("/home/yrazeghi/data/IRIS/iris.data")
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());
    Net net(config.seed);

}

int main() {
    Config config;
    train(config);
}
