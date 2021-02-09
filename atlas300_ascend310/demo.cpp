#include <iostream>
using namespace std;
extern "C"{
    void demo1(float *matrix,int len1,int len2){
        for(int i=0;i<len1;i++){
           int start=i*len2;
           for(int j=0;j<len2;j++){
                //std::cout<<matrix[start+j]<<std::endl;
                matrix[start+j]=matrix[start+j]+1;

           }
        }
    }
}
