#  ---------------------------------------
#  Make.inc for Force2Vec kernel generator 
#  ---------------------------------------

   SHELL = /bin/sh 

#  ---------------------------------------------------
#  Set the path of the kgen directory manually for now
#  ---------------------------------------------------
TOP = /home/msujon/git/Force2Vec/sample/kgen/
#TOP = $(dir $(firstword $(MAKEFILE_LIST)))

#  ----------------------------------------
#  Paths to various directories 
#  ---------------------------------------
   
   BINdir = $(TOP)/bin
   BLDdir = $(TOP)/obj
   SRCdir = $(TOP)/src
   CGENdir = $(TOP)/codegen
   INCSdir = $(TOP)/include
   SIMDdir = $(TOP)/simd
   LIBdir = $(TOP)/lib
   RUNdir= $(TOP)/script
   GENdir = $(TOP)/generated
   GENSRCdir = $(GENdir)/src
   GENINCdir = $(GENdir)/include

#  ----------------------------------------
#  General purpose compiler 
#  ---------------------------------------
   CC = gcc
   CFLAGS = -O3 -w 

#  ----------------------------------------
#  Kernel compiler 
#  ---------------------------------------
   KCC = gcc
   #KCCFLAGS = -O3 -mnative
   # KCCFLAGS = -O3 -mavx -mfma 
   KCCFLAGS = -O3 -mavx512f -mavx512dq
   #KCCFLAGS = -O3 -march=armv8-a  

#  -----------------------------------------------------
#  Architecture related, we will probe the system later 
#  See $(INCSdir)/simd.h for details 
#  -----------------------------------------------------
   #ARCH = BLC_ARM64      # for ARM64 
   #SIMD = BLC_ARM_ASIMD  # for ASIMD of ARM64
   ARCH = BLC_X86
   #SIMD = BLC_AVX2
   SIMD = BLC_AVXZ  # AVX512 
   ARCHFLAGS = -DBLC_ARCH -D$(ARCH) -D$(SIMD)  
