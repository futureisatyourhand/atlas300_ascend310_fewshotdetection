#include <iostream>
#include<fstream>
#include "common/err_code.hpp"
#include "common/BatchImageParaWithScale.h"
#include "output_engine.hpp"

//extern int dtype_size;
 
HIAI_IMPL_ENGINE_PROCESS("OutputEngine", OutputEngine, 1) {
    std::cout << "[OutputEngine] Post process start.." << std::endl;
    std::shared_ptr<EngineTransT> tran_data_ptr = 
        std::static_pointer_cast<EngineTransT>(arg0);
    std::vector<OutputT> output_data_vec = tran_data_ptr->output_data_vec;
    //printf("return result size %d status=%d\n", tran_data_ptr->status, output_data_vec.size());

    if (nullptr == tran_data_ptr) {
        std::cout << "[OutputEngine]failed to process invalid message.." << std::endl;
        return HIAI_INVALID_INPUT_MSG;
    }

    static int result_len;
    //static float *fl = NULL; 

    if(tran_data_ptr->status==false)
    	std::cout << "[OutputEngine] error msg: " << tran_data_ptr->msg;
    else{
    	std::cout << "[OutputEngine] inference time msg: " << tran_data_ptr->msg;

    	//fl = (float *)tran_data_ptr->b_info.timestamp[0];
    	result_len = tran_data_ptr->b_info.timestamp[1];
    	int out_size = output_data_vec[0].size / sizeof(float); 
        std::ofstream file;//elapsed=elapse.count(); 
        file.open("./output.txt", std::ios::app);  
        for (int ii=0; ii<out_size; ii++){
            //std::cout<<output_data_vec[0].size<<std::endl;        
            file<<std::to_string((output_data_vec[0].data.get())[ii])+" ";
    //	    printf("return %d result %d : %s = %lf;id2222=%lf \n",out_size, ii, output_data_vec[0].name.c_str(), (output_data_vec[0].data.get())[ii],(output_data_vec[0].data.get())[int(ii+out_size/2)]);
	    //*(fl+ii) = (output_data_vec[0].data.get())[ii];
            
        }
        file.close();
    }

    tran_data_ptr->output_data_vec.clear();
    hiai::Engine::SendData(0, "EngineTransT", 
            std::static_pointer_cast<void>(tran_data_ptr));
    //hiai::Graph::DestroyGraph(1);
    return HIAI_OK;
}
