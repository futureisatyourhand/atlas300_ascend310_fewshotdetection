
#ifndef BatchImageParaWithScale_H_
#define BatchImageParaWithScale_H_

#include "hiaiengine/data_type.h"
#include "hiaiengine/data_type_reg.h"

using hiai::BatchInfo;
using hiai::IMAGEFORMAT;
using hiai::ImageData;

typedef struct ScaleInfo {
    float scale_width = 1;
    float scale_height = 1;
} ScaleInfoT;
template <class Archive>
void serialize(Archive& ar, ScaleInfoT& data) {
    ar(data.scale_width, data.scale_height);
}

typedef struct ResizeInfo {
    uint32_t resize_width = 0;
    uint32_t resize_height = 0;
} ResizeInfoT;
template <class Archive>
void serialize(Archive& ar, ResizeInfo& data) {
    ar(data.resize_width, data.resize_height);
}

typedef struct CropInfo {
    int point_x = 0;
    int point_y = 0;
    int crop_width = 0;
    int crop_height = 0;
} CropInfoT;
template <class Archive>
void serialize(Archive& ar, CropInfo& data) {
    ar(data.point_x, data.point_y, data.crop_width, data.crop_height);
}



typedef struct NewImagePara {
    hiai::FrameInfo f_info;
//    hiai::ImageData<uint8_t> img;
    hiai::ImageData<float> img;
    ScaleInfoT scale_info;
    ResizeInfo resize_info;
    CropInfo crop_info;
} NewImageParaT;

template <class Archive>
void serialize(Archive& ar, NewImageParaT& data) {
    ar(data.f_info, data.img, data.scale_info,data.resize_info, data.crop_info);
}

typedef struct NewImagePara2 {
    hiai::FrameInfo f_info;
    hiai::ImageData<float> img;
    ScaleInfoT scale_info;
} NewImageParaT2;

template <class Archive>
void serialize(Archive& ar, NewImageParaT2& data) {
    ar(data.f_info, data.img, data.scale_info);
}

typedef struct BatchImageParaWithScale {
    hiai::BatchInfo b_info;
    std::vector<NewImageParaT> v_img;
    //int  result_data;
    //int result_len;
} BatchImageParaWithScaleT;


typedef struct BatchImageParaWithScale2 {
    hiai::BatchInfo b_info;
    std::vector<NewImageParaT2> v_img;
} BatchImageParaWithScaleT2;


template <class Archive>
void serialize(Archive& ar, BatchImageParaWithScaleT& data) {
    ar(data.b_info, data.v_img);
    //ar(data.result_data, data.result_len, data.b_info, data.v_img);
}

template <class Archive>
void serialize(Archive& ar, BatchImageParaWithScaleT2& data) {
    ar(data.b_info, data.v_img);
}

struct ImageAll {
    int width_org;
    int height_org;
    int channal_org;
    ImageData<float> image;
};

template <class Archive>
void serialize(Archive& ar, ImageAll& data) {
    ar(data.width_org, data.height_org, data.channal_org, data.image);
}

struct BatchImageParaScale {
    BatchInfo b_info;             
    std::vector<ImageAll> v_img; 
};

template <class Archive>
void serialize(Archive& ar, BatchImageParaScale& data) {
    ar(data.b_info, data.v_img);
}

typedef enum ImageType{
    IMAGE_TYPE_RAW = -1,
    IMAGE_TYPE_NV12 = 0,
    IMAGE_TYPE_JPEG,
    IMAGE_TYPE_PNG,
    IMAGE_TYPE_BMP,
    IMAGE_TYPE_TIFF,
    IMAGE_TYPE_VIDEO = 100
}ImageTypeT;

struct EvbImageInfo{
    bool is_first;
    bool is_last;
    uint32_t batch_size;
    uint32_t batch_index;
    uint32_t max_batch_size;
    uint32_t batch_ID;
    uint32_t frame_ID;
    int format;
    uint32_t width  = 0;
    uint32_t height = 0;
    uint32_t size = 0;
    u_int8_t* pucImageData;
};

const int SEND_DATA_INTERVAL_MS = 200000;


inline bool isSentinelImage(const std::shared_ptr<BatchImageParaWithScaleT> image_handle){
    if (image_handle && image_handle->b_info.batch_ID == -1){
        return true;
    }
    return false;
}

inline bool isSentinelImage(const std::shared_ptr<BatchImageParaWithScaleT2> image_handle){
    if (image_handle && image_handle->b_info.batch_ID == -1){
        return true;
    }
    return false;
}


typedef struct Output
{
    int32_t size;
    std::string name;
    std::shared_ptr<float> data;
}OutputT;
template<class Archive>
void serialize(Archive& ar, OutputT& data)
{
    ar(data.size);
    ar(data.name);
    if (data.size > 0 && data.data.get() == nullptr)
    {
        data.data.reset(new float[data.size]);
    }

    ar(cereal::binary_data(data.data.get(), data.size * sizeof(float)));
}


typedef struct EngineTrans
{
    bool status;
    std::string msg;
    hiai::BatchInfo b_info;
    uint32_t size;
    std::vector<OutputT> output_data_vec;
    std::vector<NewImageParaT> v_img;   
    //int  result_data;
    //int result_len;
}EngineTransT;

template<class Archive>
void serialize(Archive& ar, EngineTransT& data)
{
   ar(data.status, data.msg, data.b_info, data.size, data.output_data_vec, data.v_img);
   //ar(data.result_data, data.result_len, data.status, data.msg, data.b_info, data.size, data.output_data_vec, data.v_img);
   //ar(data.status, data.msg, data.b_info, data.size, data.output_data_vec, data.v_img, data.result_data, data.result_len);
}

typedef struct {
	std::string tfilename;
	int format;
	int height;
	int width;
}ImageInfor;


#endif
