#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include "Net.h"
#include "Config.h"

    class IRIS : public torch::data::Dataset<IRIS> {
    private:
        torch::Tensor states_, labels_;
        int data_size = 0;

    public:
        explicit IRIS(const std::string &loc) {
            //read in c++ and set states and labels.
            read_data(loc);


        };

        torch::data::Example<> get(size_t index) override{
            // You may for example also read in a .csv file that stores locations
            // to your data and then read in the data at this step. Be creative.
            return {states_[index], labels_[index]};
        };
        torch::optional<size_t> size() const override {

            return data_size;
        };
        void read_data(const std::string &loc){
            std::fstream dataFile;
            dataFile.open(loc);
            std::vector<std::vector<double>> rows_double(150, std::vector<double>(4, 0.0));;
            if (!dataFile.is_open())
                throw std::runtime_error("Could not open file");
            std::string line, word = "";
            std::vector<std::string> row ;
            std::vector<double> row_d;
            std::vector<double> labels_v;
            int line_number=-1;
            while (!dataFile.eof()) {
                line_number++;
                row.clear();
                row_d.clear();
                getline(dataFile, line);
                if(line == "")
                    continue;
                std::stringstream s(line);
                int counter = 0;
                while ( getline(s, word, ',') ) {
                    if (counter==4){
                        if (word=="Iris-setosa")
                            labels_v.push_back(0);
                        else if (word=="Iris-versicolor")
                            labels_v.push_back(1);
                        else if (word=="Iris-virginica")
                            labels_v.push_back(2);
                        else{
                            std::cerr << "this should not happen" << word << "******" <<'\n';
                        }
                    }
                    else {
                        rows_double[line_number][counter] = (double) atof(word.c_str());
                    }
                    row.push_back(word);
                    counter++;
                }
            }
            dataFile.close();
            data_size = rows_double.size();
            auto empty_tensor = torch::empty(rows_double.size()*rows_double[0].size());
            float* myData = empty_tensor.data_ptr<float>();

            //creating tensor from vector
            for(int i=0; i<rows_double.size(); i++)
                for(int j =0; j<rows_double[0].size(); j++)
                    *myData++ = (float)rows_double[i][j];
            states_ = empty_tensor.resize_({rows_double.size(),rows_double[0].size()}).clone();
            labels_ =  torch::from_blob(labels_v.data(), {labels_v.size()}, torch::kFloat64).clone();
        };
    };

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
        torch::Device device(device_type);
        auto model = Net(4);
        model->to(device);
        model->train();

        auto dataset = IRIS("/home/yrazeghi/data/IRIS/iris.data")
                .map(torch::data::transforms::Stack<>());
        int64_t batch_size = 5;
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset,batch_size);
        torch::optim::SGD optimizer(model->parameters(), /*lr=*/0.1);
        // Train the network.
        int64_t n_epochs = 300;
        int64_t log_interval = 20;
        int dataset_size = dataset.size().value();
        float best_mse = std::numeric_limits<float>::max();
        for (int epoch = 1; epoch <= n_epochs; epoch++) {
            // Track loss.
            size_t batch_idx = 0;
            float mse = 0.; // mean squared error
            int count = 0;

            for (auto &batch : *data_loader) {
                auto imgs = batch.data;
                auto labels = batch.target.squeeze();
                imgs = imgs.to(torch::kF32);
                labels = labels.to(torch::kInt64);
                optimizer.zero_grad();
                auto output = model->forward(imgs);
                auto loss = torch::nll_loss(output, labels);
                loss.backward();
                optimizer.step();
                mse += loss.template item<float>();
                batch_idx++;
//                if (batch_idx % log_interval == 0) {
//                    std::printf(
//                            "Train Epoch: %d/%ld [%5ld/%5d] Loss: %.4f \n",
//                            epoch,
//                            n_epochs,
//                            batch_idx * batch.data.size(0),
//                            dataset_size,
//                            loss.template item<float>());
//                }

                count++;
            }
            mse /= (float)count;
            printf(" Mean squared error: %f\n", mse);

            if (mse < best_mse)
            {
                torch::save(model, "../best_model.pt");
                best_mse = mse;
            }
        }




    }

    int main() {
        Config config;
        train(config);
        return 0;
    }