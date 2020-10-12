BIN = ./bin
Kdir = ../
KLIBdir = $(Kdir)/lib
KINCdir = $(Kdir)/include
INCdir=../../

# indextype : int64_t or int32_t
# NOTE: when comparing with MKL, use ibit=64 since we are using MKL_ILP64
ibit=64
#ibit=32

# valuetype precision : double single 
#pre=d
#vlen=4
pre=s
vlen=16
#vlen=8
#
# SIMD width on system: 
#    Update ARCH, SIMD variable in kernels/make.inc 
#    See kernels/include/simd.h for details    
#
# max dimension or max value of K for generated library 
mdim=128  
regblk=bacrb 
#regblk=acrb 
#regblk=crb 

kruntime=1
bestK=128    # needed when kruntime=1 

#setup flags based on type 
ifeq ($(pre), d)
   dtyp=-DDREAL
else
   dtyp=-DSREAL
endif
TYPFLAGS = -DINDEXTYPE=int$(ibit)_t -DINT$(ibit) $(dtyp)

# Library info  
sLIBS=$(KLIBdir)/$(pre)libgForce2Vec_sequential.a 
ptLIBS=$(KLIBdir)/$(pre)libgForce2Vec_pt.a 

KINCS=$(KINCdir)/kernels.h 

#
# tester/timer's compiler 
#
CC = g++
FLAGS = -fopenmp -O3 -march=native -std=c++11

#
# My parallel flags 
#
ldb=l
NTHREADS=48
#NTHREADS=6
LDB=LOAD_BALANCE 
MYPT_FLAG = -DPTTIME -DNTHREADS=$(NTHREADS) -D$(LDB)  
#MYPT_FLAG = -DPTTIME -DNTHREADS=$(NTHREADS) -DSTATIC  

# =========================================================================
# 	Target 
# =========================================================================

all: $(BIN)/x$(pre)Force2Vectime_pt        
#
#   serial version
#
$(BIN)/$(pre)Force2Vectime.o: Force2Vectime.cpp $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(TYPFLAGS) -I$(KINCdir) -I$(INCdir) -DCPP -c Force2Vectime.cpp -o $@   
$(BIN)/x$(pre)Force2Vectime: $(BIN)/$(pre)Force2Vectime.o $(sLIBS)  
	$(CC) $(FLAGS) -o $@ $^ $(sLIBS) -lm
#
#  parallel version 
#
$(BIN)/$(pre)Force2Vectime_pt.o: Force2Vectime.cpp $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(TYPFLAGS) -I$(KINCdir) -I$(INCdir) $(MYPT_FLAG) -DCPP \
		-c Force2Vectime.cpp -o $@   
$(BIN)/x$(pre)Force2Vectime_pt: $(BIN)/$(pre)Force2Vectime_pt.o $(ptLIBS)  
	$(CC) $(FLAGS) -o $@ $^ $(ptLIBS) -lm 

# ===========================================================================
# To generate Force2Vec kernels 
# =========================================================================

$(sLIBS)  : $(ptLIBS)
$(ptLIBS) : $(Kdir)/rungen.sh  
	cd $(Kdir) ; ./rungen.sh -p $(pre) -i $(ibit) -s $(vlen) -e $(mdim) \
	   -v $(vlen) -t $(NTHREADS) -r $(regblk) -k $(kruntime) -b $(bestK)
#
# cleanup 
#

clean:
	rm -rf ./bin/*

killlib: 
	cd $(Kdir) ; make clean pre=$(pre) 

