
#ifndef INC_DATA_RECV_H_
#define INC_DATA_RECV_H_
#include <hiaiengine/api.h>
#include <string>
#include <stdint.h>

extern std::mutex g_test_mutex;
extern std::condition_variable g_test_cv;
extern bool g_test_flag;


class CustomDataRecvInterface : public hiai::DataRecvInterface
{
 public:
    /**
    * @ingroup FasterRcnnDataRecvInterface
    * @brief init
    * @param [in]desc:std::string
    */
    CustomDataRecvInterface() {}

    /**
    * @ingroup FasterRcnnDataRecvInterface
    * @brief RecvData RecvData
    * @param [in]
    */
    HIAI_StatusT RecvData(const std::shared_ptr<void>& message);

 private:
    std::string file_name_;
};
#endif  // INC_DATA_RECV_H_
