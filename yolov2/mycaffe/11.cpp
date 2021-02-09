#include<fstream>
#include<iostream>
#include<chrono>

int main(){
   auto start=std::chrono::high_resolution_clock::now();
    std::ofstream file1;
    file1.open("times.txt", std::ios::app);
    file1<< "1111" << "\n";
    file1.close();

   auto end=std::chrono::high_resolution_clock::now();
    auto elapsed=std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
   std::ofstream file;
    file.open("times.txt", std::ios::app);
    file << "1111=="+std::to_string(elapsed.count()) << "\n";
    file.close();
return 0;
}
