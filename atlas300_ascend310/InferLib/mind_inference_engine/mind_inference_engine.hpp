
#ifndef MindINFERENCEENGINE_ENGINE_H_
#define MindINFERENCEENGINE_ENGINE_H_
#include "hiaiengine/api.h"
#include "hiaiengine/ai_model_manager.h"
#include "hiaiengine/ai_types.h"
#include "hiaiengine/data_type.h"
#include "hiaiengine/engine.h"
#include "hiaiengine/multitype_queue.h"
#include "hiaiengine/data_type_reg.h"
#include "hiaiengine/ai_tensor.h"
#include "common/BatchImageParaWithScale.h"

#define INPUT_SIZE 1
#define OUTPUT_SIZE 1
using hiai::Engine;

class MindInferenceEngine : public Engine {
public:
    MindInferenceEngine() {}
    HIAI_StatusT Init(const hiai::AIConfig& config, const  std::vector<hiai::AIModelDescription>& model_desc);
    /**
    * @ingroup hiaiengine
    * @brief HIAI_DEFINE_PROCESS : override Engine Process logic
    * @[in]: define a input port, a output port
    */
   HIAI_DEFINE_PROCESS(INPUT_SIZE, OUTPUT_SIZE);
private:
    int batch_size;
    int state;
    std::shared_ptr<hiai::AIModelManager> ai_model_manager_;
};


#endif
