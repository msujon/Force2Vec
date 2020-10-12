#include <cstdio>
#include <cstdint>
#include <random>
#include <cassert>
#include <omp.h>
#include "CSC.h"
#include "CSR.h"
#include "commonutility.h"
#include "utility.h"

#include"../include/kernels.h"
/*
 * select data type in Makefile
 *    pre=[s,d]
 *    ityp=[int64_t,int32_t]
 */

#ifdef DREAL
   #define VALUETYPE double
#else
   #define VALUETYPE float
#endif

#ifdef INT64
   #ifndef INT64_MAX 
      #error "64bit integer not supported in this architecture!!!"
   #endif
#endif
#ifdef INT32
   #ifndef INT32_MAX 
      #error "32bit integer not supported in this architecture!!!"
   #endif
#endif

/*
 * some misc definition for timer : from ATLAS 
 */
#define ATL_MaxMalloc 268435456UL
#define ATL_Cachelen 64
   #define ATL_MulByCachelen(N_) ( (N_) << 6 )
   #define ATL_DivByCachelen(N_) ( (N_) >> 6 )

#define ATL_AlignPtr(vp) (void*) \
        ATL_MulByCachelen(ATL_DivByCachelen((((size_t)(vp))+ATL_Cachelen-1)))

/*
 * ===========================================================================
 * Defining API for our new kernel 
 * input: 
 *    Dense matrices: A -> MxK B -> NxK C -> MxD 
 *    Sparse: S -> MxN 
 * output:
 *    C -> MxK 
 *
 *    dot / sum /subtraction / t-dist
 *       - dot : scalar, tdist 
 *       - sum / subtraction : vector  
 *    sigmoid / scal 
 *
 *
 *
 *    Meta descriptor: 
 *      
 *
 * ============================================================================
 */

/* based on CSR */
typedef void (*csr_mm_t) 
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
);

/* based on CSC */
typedef void (*csc_mm_t) 
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
);

/* ============================================================================
 *    sample kernels 
 * 
 *============================================================================*/

/*
 * Trusted kernels from gl/src/array/cpu/sddmmspmm 
 */
#define SM_TABLE_SIZE 2048
#define SM_BOUND 5.0
#define SM_RESOLUTION SM_TABLE_SIZE/(2.0 * SM_BOUND)

template <typename DType>
DType scale(DType v){
	if(v > SM_BOUND) return SM_BOUND;
        else if(v < -SM_BOUND) return -SM_BOUND;
        return v;
}

template <typename DType>
DType fast_SM(DType v, DType *sm_table){
        if(v > SM_BOUND) return 1.0;
        else if(v < -SM_BOUND) return 0.0;
        return sm_table[(int)((v + SM_BOUND) * SM_RESOLUTION)];
}

template <typename IdType, typename DType>
void init_SM_TABLE(DType *sm_table){
        DType x;
        for(IdType i = 0; i < SM_TABLE_SIZE; i++){
                x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
                sm_table[i] = 1.0 / (1 + exp(-x));
        }
}

template <typename IdType, typename DType>
void SDDMMSPMMCsrTdist
(  const IdType *indptr, 
   const IdType *indices, 
   const IdType *edges,
   const DType *X, 
   const DType *Y, 
   DType *O, 
   const IdType N, 
   const int64_t dim) 
{

#ifdef PTTIME 
#pragma omp parallel for
#endif
for (IdType rid = 0; rid < N; ++rid) {
        const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
        const IdType iindex = rid * dim;
	DType T[dim];
        for (IdType j = row_start; j < row_end; ++j){
                const IdType cid = indices[j];
                const IdType jindex = cid * dim;
                DType attrc = 0;
                for (int64_t k = 0; k < dim; ++k) {
			T[k] = X[iindex + k] - Y[jindex + k];
                        attrc += T[k] * T[k];
                }
                DType d1 = -2.0 / (1.0 + attrc);
                for (int64_t k = 0; k < dim; ++k) {
			T[k] = scale<DType>(T[k] * d1);
                        O[iindex+k] = O[iindex+k]  + T[k];
                }
        }
}

}

template <typename IdType, typename DType>
void SDDMMSPMMCsrSigmoid(const IdType *indptr, const IdType *indices, const IdType *edges, 
		const DType *X, const DType *Y, DType *O, const IdType N, const int64_t dim) {

DType *sm_table;
sm_table = static_cast<DType *> (::operator new (sizeof(DType[SM_TABLE_SIZE])));
init_SM_TABLE<IdType, DType>(sm_table);

//for(IdType i = 0; i < SM_TABLE_SIZE; i++) cout << sm_table[i] << " "; cout << endl;

#ifdef PTTIME 
#pragma omp parallel for
#endif
for (IdType rid = 0; rid < N; ++rid){
        const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
        const IdType iindex = rid * dim;
	for (IdType j = row_start; j < row_end; ++j){
		const IdType cid = indices[j];
                const IdType jindex = cid * dim;
                DType attrc = 0;
                for (int64_t k = 0; k < dim; ++k) {
                        attrc += X[iindex + k] * Y[jindex + k];
                }
                //DType d1 = 1.0 / (1.0 + exp(-attrc));
                DType d1 = fast_SM<DType>(attrc, sm_table);
		//printf("");
		for (int64_t k = 0; k < dim; ++k) {
                        O[iindex+k] = O[iindex+k]  + (1.0 - d1) * Y[jindex + k];
                }
        }
}		

}

void mytrusted_csr 
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
)
{
   switch(tkern)
   {
      case 't' : // t-dist 
         //fprintf(stderr, "***Applying trusted t-dist kernel: (m,k) = %d, %d\n", m, k);
         SDDMMSPMMCsrTdist<INDEXTYPE, VALUETYPE> (pntrb, indx, NULL, a, b, c, 
               m, k);
         break;
      case 's' : // sigmoid
         //fprintf(stderr, "***Applying trusted sigmoid kernel\n");
         SDDMMSPMMCsrSigmoid<INDEXTYPE, VALUETYPE> (pntrb, indx, NULL, a, b, c, 
               m, k);
         break;
      default:
         printf("unknown trusted kernel, timing is exiting... ... ...\n");
         exit(1);
   }
}

/*=============================================================================
 *          Tester framework 
 * We will redesign it with tester class later... just using template here
 * ============================================================================
 */
// from ATLAS; ATL_epsilon.c 
template <typename NT> 
NT Epsilon(void)
{
   static NT eps; 
   const NT half=0.5; 
   volatile NT maxval, f1=0.5; 

   do
   {
      eps = f1;
      f1 *= half;
      maxval = 1.0 + f1;
   }
   while(maxval != 1.0);
   return(eps);
}

template <typename IT, typename NT>
int doChecking(IT NNZA, IT M, IT N, NT *C, NT *D, IT ldc)
{
   IT i, j, k;
   NT diff, EPS; 
   double ErrBound; 

   int nerr = 0;
/*
 * Error bound : total computation = K*NNZ + K*NNZ FMAC = 4*K*NNZ
 *               flop per element of C = 4*K*NNZ / M*K
 *
 */
   EPS = Epsilon<NT>();
   // the idea is how many flop one element needs, should be max degree
   // NOTE: avg degree will not do, since some rows may have more non-zero  
   ErrBound = 4 * (NNZA) * EPS; /* considering upper bound for now*/ 
   cout << "--- EPS = " << EPS << " ErrBound = " << ErrBound << endl; 
   //cout << "--- ErrBound = " << ErrBound << " NNZ(A) = " << NNZA << " N = " << N  <<endl; 
   // row major! 
   for (i=0; i < M; i++)
   {
      for (j=0; j < N; j++)
      {
         k = i*ldc + j;
         diff = C[k] - D[k];
         if (diff < 0.0) diff = -diff; 
         if (diff > ErrBound)
         {
      #if 0
            fprintf(stderr, "C(%d,%d) : expected=%e, got=%e, diff=%e\n",
                    i, j, C[k], D[k], diff);
      #else // print single value... 
            if (!i && !j)
               fprintf(stderr, "C(%ld,%ld) : expected=%e, got=%e, diff=%e\n",
                       i, j, C[k], D[k], diff);
      #endif
            nerr++;
         }
         else if (D[k] != D[k]) /* test for NaNs */
         {
            fprintf(stderr, "C(%ld,%ld) : expected=%e, got=%e\n",
                    i, j, C[k], D[k]);
            nerr++;

         }
      }
   }
   return(nerr);
}

template <csr_mm_t trusted, csr_mm_t test>
int doTesting_Acsr
(
   CSR<INDEXTYPE,VALUETYPE> &S, 
   INDEXTYPE M, 
   INDEXTYPE N, 
   INDEXTYPE K, 
   VALUETYPE alpha, 
   VALUETYPE beta,
   int tkern
)
{
   int nerr, szAligned; 
   size_t i, j, szA, szB, szC, lda, ldc, ldb; 
   VALUETYPE *pb, *b, *pc0, *c0, *pc, *c, *pa, *a, *values;

   std::default_random_engine generator;
   std::uniform_real_distribution<VALUETYPE> distribution(0.0,1.0);
/*
 * NOTE: we are considering only row major A, B and C storage now
 *       A -> MxK, B->NxK, C->MxD  
 */
   lda = ldb = ldc = K; // both row major, K multiple of VLEN 
/*
 * NOTE: not sure about system's VLEN from this user code. So, make it cacheline
 * size aligned ....
 */
   szAligned = ATL_Cachelen / sizeof(VALUETYPE);
   szA = ((M*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szB = ((N*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szC = ((M*ldc+szAligned-1)/szAligned)*szAligned;  // szC in element 
   
   pa = (VALUETYPE*)malloc(szA*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pa);
   a = (VALUETYPE*) ATL_AlignPtr(pa);
   
   pb = (VALUETYPE*)malloc(szB*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pb);
   b = (VALUETYPE*) ATL_AlignPtr(pb);

   pc0 = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pc0);
   c0 = (VALUETYPE*) ATL_AlignPtr(pc0); 
      
   pc = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pc);
   c = (VALUETYPE*) ATL_AlignPtr(pc); 
   
   // init   
   for (i=0; i < szA; i++)
   {
   #if 1
      a[i] = distribution(generator);  
   #else
      //a[i] = 1.0*i;  
      a[i] = 0.5;  
   #endif
   }
   for (i=0; i < szB; i++)
   {
   #if 1
      b[i] = distribution(generator);  
   #else
      //b[i] = 1.0*i;  
      b[i] = 0.5;  
   #endif
   }
   for (i=0; i < szC; i++)
   {
   #if 0
      c[i] = c0[i] = distribution(generator);  
   #else  /* to test beta0 case */
      c[i] = 0.0; c0[i] = 0.0;
   #endif
   }
  
   if (M > S.rows) M = S.rows; // M can't be greater than A.rows  
/*
 *    csr may consists all 1 as values... init with random values
 */
   values = (VALUETYPE*)malloc(S.nnz*sizeof(VALUETYPE));
   assert(values);
   for (i=0; i < S.nnz; i++)
      values[i] = distribution(generator);  
/*
 * Let's apply trusted and test kernels 
 */
   fprintf(stdout, "Applying trusted kernel\n");
   //trusted(tkern, M, N, K, alpha, S.nnz, S.rows, S.cols, 
   //        S.colids, S.rowptr, S.rowptr+1, a, lda, b, ldb, beta, c0, ldc);   
   trusted(tkern, M, N, K, alpha, S.nnz, S.rows, S.cols, values, 
           S.colids, S.rowptr, S.rowptr+1, a, lda, b, ldb, beta, c0, ldc);   
   
   fprintf(stdout, "Applying test kernel\n");
   //test(tkern, M, N, K, alpha, S.nnz, S.rows, S.cols, 
   //      S.colids, S.rowptr, S.rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   test(tkern, M, N, K, alpha, S.nnz, S.rows, S.cols, values, 
         S.colids, S.rowptr, S.rowptr+1, a, lda, b, ldb, beta, c, ldc);   
/*
 * check for errors 
 */
   nerr = doChecking<INDEXTYPE, VALUETYPE>(S.nnz, M, K, c0, c, ldc);

   free(values);
   free(pc0);
   free(pc);
   free(pb);
   free(pa);

   return(nerr);
}
/*==============================================================================
 *    Timer:  
 *
 *============================================================================*/
/*
 * NOTE: kernel timer prototype, typedef template function pointer   
 */
template <typename IT>
using csr_timer_t = vector<double> (*) 
(
   const int tkern,         // kernel type
   const int nrep,         // number of repeatation 
   const IT M,      
   const IT N,     
   const IT K,    
   const VALUETYPE alpha,  // alpha
   const IT nnz,
   const IT rows,
   const IT cols,
   VALUETYPE *values,      // values
   IT *rowptr,   
   IT *colids,
   const VALUETYPE *a,
   const IT lda,
   const VALUETYPE *b,
   const IT ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const IT ldc
);
/*
 * Kernel timer wrapper for trusted kernel.. 
 * This wrapper handles all extra setup needed to call a library, like: MKL
 */
vector<double> callTimerTrusted_Acsr
(
   const int tkern,      // ROW_MAJOR, INDEX_BASE_ZERO 
   const int nrep,      // number of repeatation 
   const INDEXTYPE M,
   const INDEXTYPE N,
   const INDEXTYPE K, // A.cols
   const VALUETYPE alpha,
   const INDEXTYPE nnz,
   const INDEXTYPE rows,
   const INDEXTYPE cols,
   VALUETYPE *values, 
   INDEXTYPE *rowptr,
   INDEXTYPE *colids,
   const VALUETYPE *a,     
   const INDEXTYPE lda,   
   const VALUETYPE *b,
   const INDEXTYPE ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const INDEXTYPE ldc
)
{
   double start, end;
   vector <double> results;  // don't use single precision, use double  
/*
 * NOTE: 
 *    flag can be used to select different option, like: ROW_MAJOR, 
 *    INDEX_BASE_ZERO. For now, we only support following options (no checking):
 *       SPARSE_INDEX_BASE_ZERO
 *       SPARSE_OPERATION_NON_TRANSPOSE
 *       SPARSE_LAYOUT_ROW_MAJOR
 */
   // timing inspector phase 
   {
      results.push_back(0.0); // no inspection phase 
   }
   
   mytrusted_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
              colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   
   start = omp_get_wtime();
   for (int i=0; i < nrep; i++)
   {
      mytrusted_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
                 colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   }
   end = omp_get_wtime();
   results.push_back((end-start)/((double)nrep)); // execution time 

   return(results);
}


/*
 * timer wrapper for test kernel 
 */
vector<double> callTimerTest_Acsr
(
   const int tkern,      // kernel type  
   const int nrep,      // number of repeatation 
   const INDEXTYPE M,
   const INDEXTYPE N,
   const INDEXTYPE K, 
   const VALUETYPE alpha,
   const INDEXTYPE nnz,
   const INDEXTYPE rows,
   const INDEXTYPE cols,
   VALUETYPE *values, 
   INDEXTYPE *rowptr,
   INDEXTYPE *colids,
   const VALUETYPE *a,     
   const INDEXTYPE lda,   
   const VALUETYPE *b,
   const INDEXTYPE ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const INDEXTYPE ldc
)
{
   double start, end;
   vector <double> results;  // don't use single precision, use double  
/*
 * NOTE: 
 *    flag can be used to select different option, like: ROW_MAJOR, 
 *    INDEX_BASE_ZERO. For now, we only support following options (no checking):
 *       SPARSE_INDEX_BASE_ZERO
 *       SPARSE_OPERATION_NON_TRANSPOSE
 *       SPARSE_LAYOUT_ROW_MAJOR
 */
   // timing inspector phase 
   {
      results.push_back(0.0); // no inspection phase 
   }
  
#ifdef DREAL
   dgForce2Vec_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
              colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
#else
   sgForce2Vec_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
              colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
#endif
   start = omp_get_wtime();
   for (int i=0; i < nrep; i++)
   {
   #ifdef DREAL
      dgForce2Vec_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
                 colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   #else
      sgForce2Vec_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
                 colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   #endif
   }
   end = omp_get_wtime();
   results.push_back((end-start)/((double)nrep)); // execution time 

   return(results);
}

/*
 * Assuming large working set, sizeof B+D > L3 cache 
 */
template<typename IT, csr_timer_t<IT> CSR_TIMER>
vector <double> doTiming_Acsr
(
 const CSR<INDEXTYPE, VALUETYPE> &S, 
 IT M, 
 IT N, 
 IT K,
 const VALUETYPE alpha,
 const VALUETYPE beta,
 const int csKB,
 const int nrep,
 const int tkern
 )
{
   int szAligned; 
   IT i, j;
   vector <double> results; 
   double start, end;
   IT nnz, rows, cols;
   //size_t szB, szC, ldb, ldc; 
   IT szA, szB, szC, lda, ldb, ldc; 
   VALUETYPE *pa, *a, *pb, *b, *pc, *c, *values;
   IT *rowptr, *colids;

#if defined(PTTIME) && defined(NTHREADS)
   omp_set_num_threads(NTHREADS);
#endif

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);

   lda = ldb = ldc = K; // considering both row-major   

   szAligned = ATL_Cachelen / sizeof(VALUETYPE);
   szA = ((M*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szB = ((N*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szC = ((M*ldc+szAligned-1)/szAligned)*szAligned;  // szC in element 

   pa = (VALUETYPE*)malloc(szA*sizeof(VALUETYPE)+ATL_Cachelen);
   assert(pa);
   a = (VALUETYPE*) ATL_AlignPtr(pa);
   
   pb = (VALUETYPE*)malloc(szB*sizeof(VALUETYPE)+ATL_Cachelen);
   assert(pb);
   b = (VALUETYPE*) ATL_AlignPtr(pb);
   
   pc = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+ATL_Cachelen);
   assert(pc);
   c = (VALUETYPE*) ATL_AlignPtr(pc); 
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (i=0; i < szA; i++)
      a[i] = distribution(generator);  
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (i=0; i < szB; i++)
      b[i] = distribution(generator);  
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (i=0; i < szC; i++)
      c[i] = distribution(generator);  
/*
 *    To make rowptr, colids, values non-readonly 
 *    We may use it later if we introduce an inspector phase 
 *    NOTE: MKL uses diff type system ..
 */
      rowptr = (IT*) malloc((M+1)*sizeof(IT));
      assert(rowptr);
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
      for (i=0; i < M+1; i++)
         rowptr[i] = S.rowptr[i];
   
      colids = (IT*) malloc(S.nnz*sizeof(IT));
      assert(colids);
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
      for (i=0; i < S.nnz; i++)
         colids[i] = S.colids[i]; 
      
      values = (VALUETYPE*) malloc(S.nnz*sizeof(VALUETYPE));
      assert(values);
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
      for (i=0; i < S.nnz; i++)
         values[i] = distribution(generator);  
/*
 *    NOTE: with small working set, we should not skip the first iteration 
 *    (warm cache), because we want to time out of cache... 
 *    We run this timer either for in-cache data or large working set
 *    So we can safely skip 1st iteration... C will be in cache then
 */

   results = CSR_TIMER(tkern, nrep, M, N, K, alpha, nnz, rows, cols, values, 
                       rowptr, colids, a, lda, b, ldb, beta, c, ldc); 
   free(rowptr);
   free(colids);
   free(values);
   free(pb);
   free(pc);
   
   return(results);
}

/*
 * Run both trusted and test timer and compare results 
 */
void GetSpeedup(string inputfile, int option, INDEXTYPE M, 
      INDEXTYPE K, int csKB, int nrep, int isTest, int skipHeader, 
      VALUETYPE alpha, VALUETYPE beta, int tkern)
{
   int nerr, norandom;
   INDEXTYPE i;
   vector<double> res0, res1; 
   double exeTime0, exeTime1, inspTime0, inspTime1; 
   INDEXTYPE N, blkid; /* A->MxN, B-> NxD, C-> MxD */
   vector <INDEXTYPE> rblkids;
   CSR<INDEXTYPE, VALUETYPE> S_csr0; 
   CSR<INDEXTYPE, VALUETYPE> S_csr1; 
   CSC<INDEXTYPE, VALUETYPE> S_csc;
   

   SetInputMatricesAsCSC(S_csc, inputfile);
   S_csc.Sorted(); 
   N = S_csc.cols; 
   
   //cout << "K = " << K << endl; 
   // genetare CSR version of A  
   S_csr0.make_empty(); 
   S_csr0 = *(new CSR<INDEXTYPE, VALUETYPE>(S_csc));
   S_csr0.Sorted();
  /*
   * check for valid M.
   * NOTE: rows and cols of sparse matrix can be different 
   */
   if (!M || M > S_csr0.rows)
      M = S_csr0.rows;
/*
 * test the result if mandated 
 * NOTE: general notation: 
 *          Sparse Matrix : S -> MxN 
 *          Dense Matrix  : A->MxK B->NxK, C->MxK
 */
   assert(N && M && K);
   if (isTest)
   {
      //nerr = doTesting_Acsr<mytrusted_csr, mytest_csr>
      //                         (S_csr0, M, N, K, alpha, beta, tkern); 
   #ifdef DREAL
      nerr = doTesting_Acsr<mytrusted_csr, dgForce2Vec_csr>
                               (S_csr0, M, N, K, alpha, beta, tkern); 
   #else
      nerr = doTesting_Acsr<mytrusted_csr, sgForce2Vec_csr>
                               (S_csr0, M, N, K, alpha, beta, tkern); 
   #endif
      // error checking 
      if (!nerr)
         fprintf(stdout, "PASSED TEST\n");
      else
      {
         fprintf(stdout, "FAILED TEST, %d ELEMENTS\n", nerr);
         exit(1); // test failed, not timed 
      }

   }
/*
 * Now, it's time to add timer 
 */
   inspTime0 = inspTime1 = exeTime0 = exeTime1 = 0.0;
   {
      // call Trusted ... c code 
      res0 = doTiming_Acsr<INDEXTYPE, callTimerTrusted_Acsr>(S_csr0, M, N, K, 
                  alpha, beta, csKB, nrep, tkern);
      inspTime0 += res0[0];
      exeTime0 += res0[1];
      
      res1 = doTiming_Acsr<INDEXTYPE, callTimerTest_Acsr>(S_csr0, M, N, K, 
                  alpha, beta, csKB, nrep, tkern);
      //cout << "      blkid = " << blkid << " ExeTime = " << res1[1] << endl;    
      inspTime1 += res1[0];
      exeTime1 += res1[1];
   }
   //inspTime0 /= nrblk; 
   //inspTime1 /= nrblk; 
   
    //exeTime0 /= nrblk; 
   //exeTime1 /= nrblk; 
   
   if(!skipHeader) 
   {
      cout << "Filename,"
         << "NNZ,"
         << "M,"
         << "N,"
         << "K,"
         << "Trusted_exe_time,"
         << "Test_exe_time,"
         << "Speedup_exe_time,"
         << endl;
   }
   cout << inputfile << "," 
        << S_csr0.nnz << "," 
        << M << "," 
        << N << "," 
        << K << "," << std::scientific
        << exeTime0 << "," 
        << exeTime1 << "," 
        << std::fixed << std::showpoint
        << exeTime0/exeTime1
        << endl;
}

void Usage()
{
   printf("\n");
   printf("Usage for CompAlgo:\n");
   printf("-input <string>, full path of input file (required).\n");
   printf("-M <number>, rows of S (can be less than actual rows of S).\n");
   printf("-K <number>, number of cols of A, B and C \n");
   printf("-C <number>, Cachesize in KB to flush it for small workset \n");
   printf("-nrep <number>, number of repeatation \n");
   printf("-nrblk <number>, number of random blk with row M, 0/-1: all  \n");
   printf("-T <0,1>, 1 means, run tester as well  \n");
   printf("-t <t,s>, t : t-distribution, s : sigmoid  \n");
   printf("-skHd<1>, 1 means, skip header of the printed results  \n");
   printf("-ialpha <1, 0, 2>, alpha respectively 1.0, 0.0, X  \n");
   printf("-ibeta <1, 0, 2>, beta respectively 1.0, 0.0, X \n");
   printf("-h, show this usage message  \n");

}
void GetFlags(int narg, char **argv, string &inputfile, int &option, 
      INDEXTYPE &M, INDEXTYPE &K, int &csKB, int &nrep, 
      int &isTest, int &skHd, VALUETYPE &alpha, VALUETYPE &beta, char &tkern)
{
   int ialpha, ibeta; 
/*
 * default values 
 */
   option = 1; 
   inputfile = "";
   K = 128; 
   M = 0;
   tkern = 's';
   isTest = 0; 
   nrep = 20;
   //nrblk = 1;
   skHd = 0; // by default print header
   csKB = 25344; // L3 in KB 
   // alphaX, betaX would be the worst case for our implementation  
   ialpha=1; 
   alpha=1.0; 
   ibeta=1; 
   //beta = 1.0;
   
   for(int p = 1; p < narg; p++)
   {
      if(strcmp(argv[p], "-input") == 0)
      {
	 inputfile = argv[p+1];
      }
      else if(strcmp(argv[p], "-option") == 0)
      {
	 option = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-K") == 0)
      {
	 K = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-M") == 0)
      {
	 M = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-C") == 0)
      {
	 csKB = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-nrep") == 0)
      {
	 nrep = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-T") == 0)
      {
	 isTest = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-t") == 0)
      {
	 tkern = argv[p+1][0];
      }
      else if(strcmp(argv[p], "-skHd") == 0)
      {
	 skHd = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-ialpha") == 0)
      {
	 ialpha = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-ibeta") == 0)
      {
	 ibeta = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-h") == 0)
      {
         Usage();
         exit(1);
      }
   }
   if (inputfile == "")
   {
      cout << "Need input file ??? " << endl;
      exit(1);
   }
/*
 * set alpha beta
 */
#if 0 
   if (ialpha == 1 && ibeta == 1)
   {
      alpha = 1.0; 
      beta = 1.0;
   }
   else if (ialpha == 2 && ibeta == 2 )
   {
      alpha = 2.0; 
      beta = 2.0;
   }
   else
   {
      cout << "ialpha =  " << ialpha << " ibeta = " << ibeta << " not supported"
         << endl;
      exit(1);
   }
#endif
/*
 * supported beta = 0 and beta = 1 case
 */
   if (ibeta = 0)
      beta = 0.0;
   else
      beta = 1.0;

}
int main(int narg, char **argv)
{
   INDEXTYPE M, K;
   VALUETYPE alpha, beta;
   int option, csKB, nrep, isTest, skHd, nrblk;
   char tkern;
   string inputfile; 
   GetFlags(narg, argv, inputfile, option, M, K, csKB, nrep, isTest, skHd, 
            alpha, beta, tkern);
   GetSpeedup(inputfile, option, M, K, csKB, nrep, isTest, skHd, alpha, beta, 
         tkern);
   return 0;
}
