#!/bin/sh
if [ -z $DDK_HOME ];then
	echo "[ERROR] DDK_HOME does not exist! Please set environment variable: export DDK_HOME=<root folder of ddk>"
	echo "eg:  export DDK_HOME=/home/HwHiAiUser/tools/che/ddk/ddk/"
	exit 0
fi
export LD_LIBRARY_PATH=${DDK_HOME}/uihost/lib/:${DDK_HOME}/lib/x86_64-linux-gcc5.4/
TOP_DIR=${PWD}
cd ${TOP_DIR} && make
cp graph.config ./out/
for file in ${TOP_DIR}/*
do
if [ -d "$file" ]
then
  if [ -f "$file/Makefile" ];then
    cd $file && make install
  fi
fi
done
