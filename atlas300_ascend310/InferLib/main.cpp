#include <unistd.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "main.hpp"
#include "hiaiengine/api.h"
#include <string.h>
#include <opencv2/opencv.hpp>
#include "BatchImageParaWithScale.h"
#include "CommandLine.h"
#include <sys/time.h>
#include "cnpy.h"
#include <complex>
#include <stdio.h>
#include <stdlib.h>
//#include "NumCpp.hpp"


using namespace std;

static const uint32_t MAX_SLEEP_TIMER = 10;
static uint32_t CHANNEL = 3;
static uint32_t MODEL_W = 416;
static uint32_t MODEL_H = 416;
static uint32_t BS = 1; 
static uint32_t GRAPH_ID = 1;
static const uint32_t INPUT_ENGINE_ID = 2;
static const uint32_t OUTPUT_ENGINE_ID = 3;
//char input_data_name[200];
std::string dtype_name;
long long total_len;
int dtype_size;
int dtype_idx;
int pic_size; 
float* data;

bool g_test_flag = false;
std::mutex g_test_mutex;
std::condition_variable g_test_cv;



HIAI_REGISTER_DATA_TYPE("BatchImageParaWithScaleT", BatchImageParaWithScaleT);
HIAI_REGISTER_DATA_TYPE("EngineTransT", EngineTransT);
HIAI_StatusT CustomDataRecvInterface::RecvData(const std::shared_ptr<void>& message)
{
    //std::cout << "[RecvData] called" << std::endl;
    std::unique_lock<std::mutex> lck(g_test_mutex);
    g_test_flag = true;
    g_test_cv.notify_all();

    return HIAI_OK;
}

/**
* @ingroup FasterRcnnDataRecvInterface
* @brief RecvData RecvData回调，保存文�?* @param [in]
*/

// Init and create graph
HIAI_StatusT HIAI_InitAndStartGraph()
{
    // Step1: Global System Initialization before using HIAI Engine
    HIAI_StatusT status = HIAI_Init(0);
    // Step2: Create and Start the Graph
    status = hiai::Graph::CreateGraph("./graph0.config");

    if (status != HIAI_OK)
    {   std::cout << "[HIAI_InitAndStartGraph] Fail to start graph" << std::endl;
        HIAI_ENGINE_LOG(status, "Fail to start graph");
        return status;
    }

    // Step3
    std::shared_ptr<hiai::Graph> graph = hiai::Graph::GetInstance(GRAPH_ID);
    if (nullptr == graph)
    {
        HIAI_ENGINE_LOG("Fail to get the graph-%u", GRAPH_ID);
        return status;
    }

    hiai::EnginePortID target_port_config;
    target_port_config.graph_id = GRAPH_ID;
    target_port_config.engine_id = OUTPUT_ENGINE_ID;
    target_port_config.port_id = 0;
    graph->SetDataRecvFunctor(target_port_config,
    std::shared_ptr<CustomDataRecvInterface>( new CustomDataRecvInterface()));

    return HIAI_OK;
}



uint8_t* load_npy(cnpy::NpyArray &arr){
	const string s1 = "int8";
	const string s2 = "int16";
	const string s3 = "int32";
	const string s4 = "int64";
	const string s5 = "float32";
	const string s6 = "float64";
	uint8_t *mat_dat;
	
	std::cout << "Set type: " << dtype_name << ",   NpyArray Val num: " << arr.num_vals << ",  Word size: " << arr.word_size << std::endl;
	if (dtype_name==s1){	
			if (arr.word_size != sizeof(char)){
				cout << "whb npy' s data_type size is not equal to " << dtype_name << ':' << sizeof(char) << endl;
				return NULL;
			}
		 dtype_size = 1;
		 dtype_idx = 1;
    		 mat_dat = (uint8_t *)arr.data<std::complex<char>>();

		}
	else if (dtype_name==s2)
		{	
			if (arr.word_size != sizeof(short)){
				cout << "npy' s data_type size is not equal to " << dtype_name << ':' << sizeof(short) << endl;
				return NULL;
			}
		 dtype_size = 2;
		 dtype_idx = 2;
    		 mat_dat = (uint8_t *)arr.data<std::complex<short>>();
		}
	else if (dtype_name==s3)
		{	
			if (arr.word_size != sizeof(int)){
				cout << "npy' s data_type size is not equal to " << dtype_name << ':' << sizeof(int) << endl;
				return NULL;
			}
		 dtype_size = 4;
		 dtype_idx = 3;
    		 mat_dat = (uint8_t *)arr.data<std::complex<int>>();
		}
	else if (dtype_name==s4)
		{	
			if (arr.word_size != sizeof(long)){
				cout << "npy' s data_type size is not equal to " << dtype_name << ':' << sizeof(long) << endl;
				return NULL;
			}
		 dtype_size = 8;
		 dtype_idx = 4;
    		 mat_dat = (uint8_t *)arr.data<std::complex<long>>();
		}
	else if (dtype_name==s5)
		{	
			if (arr.word_size != sizeof(float)){
				cout << "npy' s data_type size is not equal to " << dtype_name << ':' << sizeof(float) << endl;
				return NULL;
			}
		 dtype_size = 4;
		 dtype_idx = 5;
    		 mat_dat = (uint8_t *)arr.data<std::complex<float>>();
		}
	else if (dtype_name==s6)
		{	
			if (arr.word_size != sizeof(double)){
				cout << "npy' s data_type size is not equal to " << dtype_name << ':' << sizeof(double) << endl;
				return NULL;
			}
		 dtype_size = 8;
		 dtype_idx = 6;
    		 mat_dat= (uint8_t *)arr.data<std::complex<double>>();
		}
	else{
		 dtype_size = 0;
		 dtype_idx = 0;
		 cout << "npy' s data_type not found! \n";
		 return NULL;
	}	
    
    //make sure the loaded data matches the saved data
    //assert(arr.word_size == sizeof(std::complex<float>));
    //assert(arr.shape.size() == 3 && arr.shape[0] == Nz && arr.shape[1] == Ny && arr.shape[2] == Nx);
	cout << "Input data shape: [" ;
	total_len = 1;
    	for(int i = 0; i < arr.shape.size();i++){ 
		std::cout << arr.shape[i] << ", ";
		total_len *= arr.shape[i];
	}
	std::cout << ']' << std::endl;
	return mat_dat;	
}

long long SendData(std::shared_ptr<hiai::Graph> graph, hiai::EnginePortID engine_id, cv::Mat orig_img, int model_width,
				int model_height, uint8_t *mat_dat, long long sent_cnt, uint8_t *res_ptr,int res_len)
{
    if (graph == nullptr)
    {
        std::cout << "[SendData] graph is null." << std::endl;
        return -1;
    }

    std::shared_ptr<BatchImageParaWithScaleT> image_handle = std::make_shared<BatchImageParaWithScaleT>();
    NewImageParaT imgData;
    imgData.img.channel = CHANNEL;

    //To prepare the destination image, the original image will be resize to the top left square area of the destination image.
    //The rest space of the destination area is payed without initialized.
    static long long bs_num;
    int jj=0;

    switch (dtype_idx){
        case 1:
    		for (jj=0; jj<pic_size*BS && jj+sent_cnt < total_len; jj++)
    		    	*(data+jj) =  (float)(*((char*)(mat_dat + (sent_cnt+jj)*dtype_size)));
		break;
        case 2:
    		for (jj=0; jj<pic_size*BS && jj+sent_cnt < total_len; jj++)
    		    	 *(data+jj) = (float)(*((short*)(mat_dat + (sent_cnt+jj)*dtype_size)));
		break;
        case 3:
    		for (jj=0; jj<pic_size*BS && jj+sent_cnt < total_len; jj++)
    		    	 *(data+jj) = (float)(*((int*)(mat_dat + (sent_cnt+jj)*dtype_size)));
		break;
        case 4:
    		for (jj=0; jj<pic_size*BS && jj+sent_cnt < total_len; jj++)
    		    	 *(data+jj) = (float)(*((long*)(mat_dat + (sent_cnt+jj)*dtype_size)));
		break;
        case 5:
		data = (float*)mat_dat;
    		//for (jj=0; jj<pic_size*BS && jj+sent_cnt < total_len; jj++){
    		//    	 *(data+jj) = (*((float*)(mat_dat + (sent_cnt+jj)*dtype_size)));
		//}
		break;
        case 6:
    		for (jj=0; jj<pic_size*BS && jj+sent_cnt < total_len; jj++){
    		    	 //*(int*)(data+jj) = (int)((*((double*)(mat_dat + (sent_cnt+jj)*dtype_size)))*40000000);
    		    	 *(data+jj) = (float)(*((double*)(mat_dat + (sent_cnt+jj)*dtype_size)));
		}
		break;
	default:
		return sent_cnt; 
    }

    bs_num++;
    //std::cout << "NO.BS: " << bs_num << "   model_width:" << MODEL_W << "    model_height:" << MODEL_H << "    image size:" << pic_size << std::endl;

    float *ptr = data;
    int cnt;
    static bool res_alloced = false;

    imgData.img.size = pic_size;
    imgData.img.width = MODEL_W;
    imgData.img.height = MODEL_H;
    //image_handle->b_info.batch_size = BS;
    //image_handle->b_info.is_first = true;
    //image_handle->b_info.is_last = true;
    //image_handle->b_info.max_batch_size = BS;
    //image_handle->b_info.batch_ID = 0;

    /* we store the result array ptr address for output engine */
    image_handle->b_info.timestamp.clear();
    image_handle->b_info.timestamp.push_back((uint64_t)res_ptr);
    image_handle->b_info.timestamp.push_back((uint64_t)res_len);


    //std::shared_ptr<float> data1(new float[pic_size], std::default_delete<float[]>());
    std::shared_ptr<float> data1;
    for (cnt=0; cnt < BS && cnt+sent_cnt < total_len ; cnt++){
	//whb call here!!
        data1.reset(new float[pic_size], std::default_delete<float[]>());
        memcpy(&(*data1), ptr + cnt*(imgData.img.size), imgData.img.size*sizeof(float));
    	imgData.img.data = data1;//std::shared_ptr<float_t> (ptr + cnt*(imgData.img.size));
	//imgData.f_info.is_first = cnt==0 ? true : false; 
	//imgData.f_info.is_last = cnt==(BS-1) ? true : false; 
	imgData.f_info.frame_ID = cnt;
    	image_handle->v_img.push_back(imgData);
    	image_handle->b_info.frame_ID.push_back(cnt);
    }

    if( HIAI_OK != graph->SendData(engine_id, "BatchImageParaWithScaleT", std::static_pointer_cast<void>(image_handle)))
    {
        std::cerr << "[SendData] send data error" << std::endl;
        return -1;
    }

	return sent_cnt + pic_size*cnt;
}



//int main(int argc, char* argv[])
int infer(float *matrix, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, float *res, int arg8)
{
    struct  timeval start;
    struct  timeval end;
    double  diff;
    int iter = 1;
    long long have_sent;
    int count = 0;
    int times;
    float *float_buffer;
    //float *matrix;

    HIAI_StatusT ret = HIAI_OK;

    BS = arg1;
    CHANNEL = 3;
    MODEL_W = 416;
    MODEL_H = 416;
    total_len = arg5;
    dtype_name = "float32";
    dtype_size = arg6;
    dtype_idx = arg7;
    times = arg8;

    pic_size = MODEL_W * MODEL_H * CHANNEL;
    data = float_buffer = new float[pic_size*BS];

    // ------------------------------ Parsing and validation of input args ---------------------------------
    char buf[80];   
    getcwd(buf,sizeof(buf));   

    gettimeofday(&start,NULL);

    // 1.create graph
    //if(times==1 || times==3){	//first time to come in and initial the graph
    //if(times==0){	//first time to come in and initial the graph
    if(true){	//first time to come in and initial the graph
        ret = HIAI_InitAndStartGraph();
        if (HIAI_OK != ret)
        {
            std::cout << "[main] Fail to start graph" << std::endl;
            HIAI_ENGINE_LOG("Fail to start graph");;
            return -1;
        }
        printf("[main]Init the graph.\n");    
    }

    std::shared_ptr<hiai::Graph> graph = hiai::Graph::GetInstance(GRAPH_ID);
    printf("[main]current working directory: %s %d\n", buf, graph); 

    if (nullptr == graph)
    {
        std::cout << "[main] Fail to get the graph !!" << std::endl;
        HIAI_ENGINE_LOG("Fail to get the graph-%u", GRAPH_ID);
        return -1;
    }

    hiai::EnginePortID engine_id;
    engine_id.graph_id = GRAPH_ID;
    engine_id.engine_id = INPUT_ENGINE_ID;
    engine_id.port_id = 0;
	
    //cnpy::NpyArray arr = cnpy::npy_load("/home/liqian/bioavailability_model/tmp.npy");
    //matrix = (float*)(uint8_t*)load_npy("/home/huawei/chems/bioavailability_model/InferLib/data/atoms.npy");


    if (matrix == NULL){
        printf("Destroy !!!! Finish inference.\n");    
        hiai::Graph::DestroyGraph(GRAPH_ID);
        return 0;
    }

    gettimeofday(&end,NULL);
    have_sent = 0;
    do
    {
        cv::Mat mat_img;
	/************************************** time start **************************************/

    	//cv::Mat mat_img = cv::imread(FLAGS_i.c_str());

        have_sent = SendData(graph, engine_id, mat_img, MODEL_W, MODEL_H, (uint8_t*)matrix, have_sent, (uint8_t*)res,arg8);
	if (have_sent<0) break;
	/************************************** time end ***************************************/
	diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
        std::cout << "[main] to send image usetime = " << std::to_string(diff/1000) << "ms " << std::endl;
        //printf("%f, %f\n", matrix[0], matrix[1]);

        std::unique_lock<std::mutex> lck(g_test_mutex);
        g_test_cv.wait_for(lck, std::chrono::seconds(MAX_SLEEP_TIMER), [] {return g_test_flag;});
        g_test_flag = false;
        count++;
        break;
    } while(have_sent<total_len && count<iter);

    printf("[main]Finish inference.\n");    
    //if (times == 2 || times == 3){	//the last time, going to leave and destroy the graph
    //if (times == -1){	//the last time, going to leave and destroy the graph
    if (true){	//the last time, going to leave and destroy the graph
        printf("[main]Destroy the graph.\n");    
    	hiai::Graph::DestroyGraph(GRAPH_ID);
    }

    delete float_buffer;
    return 0;
}
