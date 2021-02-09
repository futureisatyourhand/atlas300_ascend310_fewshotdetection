TOPDIR      := $(patsubst %,%,$(CURDIR))
CC := g++
CPP := g++


#LOCAL_MODULE_NAME := main
#CC_FLAGS := -std=c++11 
LOCAL_MODULE_NAME := infer.so
CC_FLAGS := -std=c++11 -g -shared -fPIC

local_src_files := \
	$(TOPDIR)/main.cpp
	
local_inc_dirs := \
	$(TOPDIR) \
	$(DDK_HOME)/include/inc \
	$(DDK_HOME)/include/third_party/protobuf/include \
	$(DDK_HOME)/include/third_party/cereal/include \
	$(DDK_HOME)/include/libc_sec/include \
	$(DDK_HOME)/include/third_party/opencv/include \
  $(DDK_HOME)/include/third_party/gflags/include \
	$(TOPDIR)/common

local_shared_libs_dirs := \
	$(DDK_HOME)/uihost/lib/ \
	$(DDK_HOME)/ \
	out/
	
local_shared_libs := \
	matrix \
	drvdevdrv \
	drvhdc_host \
	mmpa \
	crypto \
	c_sec \
	pthread \
	protobuf \
	profilerclient \
	slog \
	cnpy \
	dl \
  gflags \
	opencv_world

Q := @
		
FULL_SRC_FILES        := $(local_src_files)
FULL_INC_DIRS         := $(foreach inc_dir, $(local_inc_dirs), -I$(inc_dir))
SHARED_LIBRARIES      := $(foreach shared_lib, $(local_shared_libs), -l$(shared_lib))
SHARED_LIBRARIES_DIRS := $(foreach shared_lib_dir, $(local_shared_libs_dirs), -L$(shared_lib_dir))

LOCAL_OBJ_PATH        := $(TOPDIR)/out
LOCAL_LIBRARY         := $(LOCAL_OBJ_PATH)/$(LOCAL_MODULE_NAME)
FULL_C_SRCS           := $(filter %.c,$(FULL_SRC_FILES))
FULL_C_OBJS           := $(patsubst $(TOPDIR)/%.c,$(LOCAL_OBJ_PATH)/%.o, $(FULL_C_SRCS))
FULL_CPP_SRCS         := $(filter %.cpp,$(FULL_SRC_FILES))
FULL_CPP_OBJS         := $(patsubst $(TOPDIR)/%.cpp,$(LOCAL_OBJ_PATH)/%.o, $(FULL_CPP_SRCS))

all: do_pre_build do_build

do_pre_build:
	$(Q)echo - do [$@]
	$(Q)mkdir -p $(LOCAL_OBJ_PATH)
		
do_build: $(LOCAL_LIBRARY) | do_pre_build
	$(Q)echo - do [$@]

$(LOCAL_LIBRARY): $(FULL_C_OBJS) $(FULL_CPP_OBJS) | do_pre_build
	$(Q)echo [LD] $@
	#$(Q)$(CPP) $(CC_FLAGS) -o $(LOCAL_LIBRARY) $(FULL_C_OBJS) $(FULL_CPP_OBJS) -Wl,--whole-archive  -Wl,--no-whole-archive -Wl,--start-group  -Wl,--end-group $(SHARED_LIBRARIES_DIRS) $(SHARED_LIBRARIES)
	$(Q)$(CPP) $(CC_FLAGS) -o $(LOCAL_LIBRARY) $(FULL_C_OBJS) $(FULL_CPP_OBJS) -Wl,--whole-archive  -Wl,--no-whole-archive -Wl,--start-group  -Wl,--end-group $(SHARED_LIBRARIES_DIRS) $(SHARED_LIBRARIES)

$(FULL_C_OBJS): $(LOCAL_OBJ_PATH)/%.o : $(TOPDIR)/%.c $(TOPDIR)/%.h  | do_pre_build
	$(Q)echo [CC] $@
	$(Q)mkdir -p $(dir $@)
	#$(Q)$(CPP) $(CC_FLAGS)  $(FULL_INC_DIRS) -c $< -o $@
	$(Q)$(CPP) $(CC_FLAGS)  $(FULL_INC_DIRS) -c $< -o $@

$(FULL_CPP_OBJS): $(LOCAL_OBJ_PATH)/%.o : $(TOPDIR)/%.cpp $(TOPDIR)/%.hpp  | do_pre_build
	$(Q)echo [CC] $@
	$(Q)mkdir -p $(dir $@)
	#$(Q)$(CPP) $(CC_FLAGS) $(FULL_INC_DIRS) -c $< -o $@
	$(Q)$(CPP) $(CC_FLAGS) $(FULL_INC_DIRS) -c $< -o $@

 
