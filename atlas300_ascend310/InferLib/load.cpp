#include"cnpy.h"
#include <unistd.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <complex>

int main(){
    cnpy::NpyArray arr = cnpy::npy_load("data/5000_X_val.npy");
    std::complex<float>* loaded_data = arr.data<std::complex<float>>();

    //make sure the loaded data matches the saved data
    assert(arr.word_size == sizeof(std::complex<float>));
    //assert(arr.shape.size() == 3 && arr.shape[0] == Nz && arr.shape[1] == Ny && arr.shape[2] == Nx);
    for(int i = 0; i < arr.shape.size();i++) std::cout << arr.shape[i] << std::endl;

    //append the same data to file
    //npy array on file now has shape (Nz+Nz,Ny,Nx)
    //cnpy::npy_save("arr1.npy",&data[0],{Nz,Ny,Nx},"a");

}
