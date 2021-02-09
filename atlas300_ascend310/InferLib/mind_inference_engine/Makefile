all : libmind_inference_engine.so
#DEVICE COMPILER		
CC := $(DDK_HOME)/toolchains/aarch64-linux-gcc6.3/bin/aarch64-linux-gnu-g++
#???????????????????????????
LOCAL_DIR  := ./

SRC_DIR = $(LOCAL_DIR)
BUILD_DIR = tmp
OUT_DIR = ../out
OBJ_DIR = $(BUILD_DIR)/obj
DEPS_DIR  = $(BUILD_DIR)/deps

#???????????
INC_DIR = \
	-I$(SRC_DIR) \
	-I$(DDK_HOME)/include/inc \
	-I$(DDK_HOME)/include/third_party/protobuf/include \
	-I$(DDK_HOME)/include/third_party/cereal/include \
	-I$(DDK_HOME)/include/libc_sec/include \
	-I../ \
	-I/usr/include/python2.7

#????????
#CC_FLAGS := $(INC_DIR) -g -std=c++11 -fPIC -DCPU_ONLY
CC_FLAGS := $(INC_DIR) -std=c++11 -fPIC -DCPU_ONLY -g
LNK_FLAGS := -shared \
	#-L$(DDK_HOME)/lib/aarch64-linux-gcc6.3 \
	#-lDvpp_api \
	#-lDvpp_jpeg_decoder \
	#-lDvpp_jpeg_encoder \
	#-lDvpp_vpc \


#??????3????
DIRS := $(shell find $(SRC_DIR) -maxdepth 3 -type d)
CUSTOM_DIRS := $(shell find $(SRC_DIR) -maxdepth 3 -type d)

#?????????????
VPATH = $(DIRS)

#??src_dir??????????cpp??
SOURCES  = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
CUSTOM_SOURCES  = $(foreach dir, $(CUSTOM_DIRS), $(wildcard $(dir)/*.cpp))
OBJS   = $(addprefix $(OBJ_DIR)/,$(patsubst %.cpp,%.o,$(notdir $(SOURCES))))
OBJS_customop = $(addprefix $(OBJ_DIR)/,$(patsubst %.cpp,%.o,$(notdir $(CUSTOM_SOURCES))))
OBJS_no_customop := $(filter-out $(OBJS_customop), $(OBJS))
DEPS  = $(addprefix $(DEPS_DIR)/, $(patsubst %.cpp,%.d,$(notdir $(SOURCES))))

# ?????????so

libmind_inference_engine.so: $(OBJS_customop)
	$(CC) $^ $(LNK_FLAGS) -o $@
	rm -rf $(BUILD_DIR)

#???????OBJ?????????
$(OBJ_DIR)/%.o:%.cpp
	@if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi;
	$(CC) -c $(CC_FLAGS) -o $@ $<


#???????DEPS?????????
#???@???????????
$(DEPS_DIR)/%.d:%.cpp
	@if [ ! -d $(DEPS_DIR) ]; then mkdir -p $(DEPS_DIR); fi;
	set -e; rm -f $@;
	$(CC) -MM $(CC_FLAGS) $< > $@.$$$$;
	sed 's,\($*\)\.o[ :]*,$(OBJ_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@;
	rm -f $@.$$$$

#???-??????
ifneq ($(MAKECMDGOALS), clean)
	-include $(DEPS)
endif

.PHONY : clean install
clean:
	rm -rf $(BUILD_DIR) lib*.so *.o $(OUT_DIR)/libmind_inference_engine.so
install: libmind_inference_engine.so
	mv *.so $(OUT_DIR)

