#include <iostream>

//extern "C++"{
using namespace std;
static const uint32_t INPUT_ENGINE_ID = 2;
static const uint32_t OUTPUT_ENGINE_ID = 3;
char input_data_name[200];
std::string dtype_name;
long long total_len;

    void infer(float *matrix, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7){
    //void demo1(float *matrix,int len1,int len2){
    	struct  timeval start;
    	struct  timeval end;
    	double  diff;

    	//HIAI_StatusT ret = HIAI_OK;
    	//int iter = 100;
    	long long have_sent;


    	int BS 	    = arg1;
    	int CHANNEL = arg2;
    	int MODEL_W = arg3;
    	int MODEL_H = arg4;
    	int dtype_idx = arg5;
    	int total_len = arg7;
	int len1 = arg2;
	int len2 = arg3;


    	int count = 0;
    	int model_width = MODEL_W;
    	int model_height = MODEL_H;
    	int pic_size = model_width * model_height * CHANNEL;
    	float *data = new float[pic_size*BS];

        for(int i=0;i<len1;i++){
           int start=i*len2;
           for(int j=0;j<len2;j++){
                //std::cout<<matrix[start+j]<<std::endl;
                matrix[start+j]=matrix[start+j]+1;

           }
        }
        delete data;
    }
//}
