include Make.inc 

# precision float=s, double=d 
pre = d
ibit=64
vlen =   # see simd.h to get the vlen for a system
dim =    # generate src for this dim value  

# setup for tuning, k-compile time
kruntime=0
mdim = 128  # max dim for k-compile time 

# setup for final code generation: kruntime  
#kruntime=1
bestK=64    # needed when kruntime = 1

nthds =  
# Register blocking strategy: 
#    bacrb: regblk all access, acrb: regblk a and c, crb: regblk only c
#regblk=bacrb
#regblk=acrb
regblk=crb

header: $(GENINCdir)/$(pre)gkernels_tdist.h \
   $(GENINCdir)/$(pre)gkernels_sigmoid.h \
   $(GENINCdir)/$(pre)gmisc.h

gmakefile : $(GENdir)/Makefile

srcfile : $(GENSRCdir)/$(pre)gForce2Vec_K$(dim)_tdist_csr.c  \
   $(GENSRCdir)/$(pre)gForce2Vec_K$(dim)_sigmoid_csr.c 

libfile : staticlibs  

$(BINdir)/xextract: $(CGENdir)/extract.c 
	make -p $(CGENdir) 
	$(CC) $(CFLAGS) -o $@ $(CGENdir)/extract.c 

$(GENdir)/Makefile : $(BINdir)/xextract $(CGENdir)/genmake.base 
	$(BINdir)/xextract -b $(CGENdir)/genmake.base -langM -def MDIM $(mdim) \
	   pre=$(pre) -def VLEN $(vlen) -def ityp $(ibit) -def nthds $(nthds) \
	   -o $@  

$(GENINCdir)/$(pre)gmisc.h : $(BINdir)/xextract $(CGENdir)/genheader.base 
	$(BINdir)/xextract -b $(CGENdir)/genheader.base -langC \
	   pre=$(pre) rout=misc -o $@  

# will add a generator to write this main makefile later 
$(GENINCdir)/$(pre)gkernels_tdist.h : $(BINdir)/xextract $(CGENdir)/genheader.base 
	$(BINdir)/xextract -b $(CGENdir)/genheader.base -langC \
	   -def MDIM $(mdim) pre=$(pre) -def VLEN $(vlen) rout=ghead \
	   -def frc tdist -def kruntime $(kruntime) -def bestK $(bestK) -o $@  
$(GENINCdir)/$(pre)gkernels_sigmoid.h : $(BINdir)/xextract $(CGENdir)/genheader.base 
	$(BINdir)/xextract -b $(CGENdir)/genheader.base -langC \
	   -def MDIM $(mdim) pre=$(pre) -def VLEN $(vlen) rout=ghead \
	   -def frc sigmoid -def kruntime $(kruntime) -def bestK $(bestK) -o $@  

$(GENSRCdir)/$(pre)gForce2Vec_K$(dim)_tdist_csr.c : $(BINdir)/xextract $(CGENdir)/genkern.base
	$(BINdir)/xextract -b $(CGENdir)/genkern.base -langC -def DIM $(dim) \
	   pre=$(pre) rblk=$(regblk) -def VLEN $(vlen) rout=tdist \
	   -def kruntime $(kruntime) -o $@  
$(GENSRCdir)/$(pre)gForce2Vec_K$(dim)_sigmoid_csr.c : $(BINdir)/xextract $(CGENdir)/genkern.base
	$(BINdir)/xextract -b $(CGENdir)/genkern.base -langC -def DIM $(dim) \
	   pre=$(pre) rblk=$(regblk) -def VLEN $(vlen) rout=sigmoid \
	   -def kruntime $(kruntime) -o $@  

staticlibs: 
	cd $(GENdir) ; make 

clean:
	rm -f $(GENSRCdir)/$(pre)*.c $(GENSRCdir)/$(pre)*.o $(GENdir)/Makefile \
		$(GENINCdir)/*.h $(LIBdir)/$(pre)*.a \
		$(BINdir)/$(pre)*.o 


kill:
	rm -f $(BINdir)/xextract 
