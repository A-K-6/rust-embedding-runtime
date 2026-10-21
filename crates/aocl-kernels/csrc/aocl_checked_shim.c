#define _POSIX_C_SOURCE 200809L
#include "aocl_checked_shim.h"
#include <aocl_dlp.h>
#include <limits.h>
#include <stdlib.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

_Static_assert(sizeof(md_t)==8, "AOCL dimension ABI must be int64");
_Static_assert(sizeof(msz_t)==8, "AOCL size ABI must be uint64");
#if AOCL_DLP_VERSION_MAJOR != 5 || AOCL_DLP_VERSION_MINOR != 3 || AOCL_DLP_VERSION_PATCH != 2
#error "The checked AOCL shim requires pinned AOCL-DLP 5.3.2 headers"
#endif
#ifndef DLP_ENABLE_OPENMP
#error "The checked AOCL shim requires an OpenMP AOCL-DLP build"
#endif
struct runtime_aocl_packed { float *data; int64_t n,k; int threads; };
static int checked_product(int64_t x,int64_t y,size_t *out) {
    if(x<=0 || y<=0 || (uint64_t)x>SIZE_MAX/(uint64_t)y/sizeof(float)) return -1;
    *out=(size_t)x*(size_t)y;
    return 0;
}
int runtime_aocl_available(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}
int runtime_aocl_release_unused_memory(void) {
#if defined(__GLIBC__)
    return malloc_trim(0) != 0;
#else
    return 0;
#endif
}
static int set_threads(int n) {
    /* Only changes this calling thread's DLP settings. */
    dlp_thread_set_ways_local(-1,-1);
    dlp_thread_set_num_threads_local((md_t)n);
    return dlp_thread_get_num_threads_active()==n ? 0 : -5;
}
int runtime_aocl_packed_create(const float *w,size_t w_len,int64_t n,int64_t k,
    int threads,runtime_aocl_packed **out,uint64_t *packed_bytes) {
    size_t expected;
    if(!out || !packed_bytes) return -1;
    *out=NULL;*packed_bytes=0;
    if(!w || n>32768 || k>32768 || threads<1 || threads>64 ||
        checked_product(n,k,&expected)!=0 || w_len!=expected) return -1;
    if(!runtime_aocl_available())return -4;
    int status=set_threads(threads);if(status!=0)return status;
    dlp_metadata_t md={0};
    msz_t bytes=aocl_get_reorder_buf_size_f32f32f32of32('R','T','B',k,n,&md);
    if(md.error_hndl.error_code!=DLP_CLSC_SUCCESS)return md.error_hndl.error_code;
    if(bytes==0 || bytes>SIZE_MAX || bytes<expected*sizeof(float))return -3;
    runtime_aocl_packed *h=calloc(1,sizeof(*h));if(!h)return -2;
    void *buf=NULL;
    if(posix_memalign(&buf,64,(size_t)bytes)!=0 || !buf){free(h);return -2;}
    h->data=buf;h->n=n;h->k=k;h->threads=threads;
    aocl_reorder_f32f32f32of32('R','T','B',w,h->data,k,n,k,&md);
    if(md.error_hndl.error_code!=DLP_CLSC_SUCCESS){
        status=md.error_hndl.error_code;free(buf);free(h);return status;
    }
    *packed_bytes=bytes;*out=h;return 0;
}
int runtime_aocl_packed_forward(const runtime_aocl_packed *h,const float *a,
    size_t a_len,int64_t m,float *c,size_t c_len) {
    size_t expected_a,expected_c;
    if(!h || !a || !c || m>1024 || checked_product(m,h->k,&expected_a)!=0 ||
       checked_product(m,h->n,&expected_c)!=0 || a_len!=expected_a ||
       c_len!=expected_c)return -1;
    int status=set_threads(h->threads);if(status!=0)return status;
    dlp_metadata_t md={0};
    aocl_gemm_f32f32f32of32('R','N','T',m,h->n,h->k,1.0f,a,h->k,'N',
        h->data,h->k,'R',0.0f,c,h->n,&md);
    return md.error_hndl.error_code;
}
void runtime_aocl_packed_destroy(runtime_aocl_packed *h) {
    if(h){free(h->data);free(h);}
}
