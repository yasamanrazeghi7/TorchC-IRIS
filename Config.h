//
// Created by yasaman razeghi on 7/13/20.
//

#ifndef BE_CONFIG_H
#define BE_CONFIG_H
#include <string>


class Config {
public:
    std::string mode = "test";
    std::string device = "cpu";
    int  seed = 2;
    int epochs = 10;
    int bach_size = 256;
    bool save_results = true;
    bool testing = true;
    std::string dataset = "grid_benchmark";
    std::string save_dir =  "./";
};


#endif //BE_CONFIG_H
