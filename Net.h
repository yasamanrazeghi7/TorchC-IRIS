#ifndef BE_NET_H
#define BE_NET_H
#include <torch/torch.h>
#include <iostream>


struct NetImpl : torch::nn::Module {
    NetImpl(int64_t Input_size)
            : linear1(torch::nn::Linear(4, 50)),
              linear2(torch::nn::Linear(50, 20)),
              linear3(torch::nn::Linear(20, 3))
    {
        register_module("linear1", linear1);
        register_module("linear2", linear2);
        register_module("linear3", linear3);
    }
    torch::Tensor forward(torch::Tensor input) {

        torch::Tensor x = torch::relu(linear1(input));
        x = torch::relu(linear2(x));
        x = torch::log_softmax(linear3(x), 1);
        return x;
    }
    torch::nn::Linear linear1, linear2, linear3;
    torch::nn::Dropout dropout;
};
TORCH_MODULE(Net);


#endif //BE_NET_H
