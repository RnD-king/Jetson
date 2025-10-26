#include "postprocess.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

// --------------------- 워크스페이스 ---------------------
namespace {
    int g_capN = 0, g_capDet = 0;

    // 후보 버퍼(최대 N)
    float *d_x1 = nullptr, *d_y1 = nullptr, *d_x2 = nullptr, *d_y2 = nullptr, *d_s = nullptr, *d_area = nullptr;
    int   *d_cntCand = nullptr;

    // NMS에 쓰는 flag
    int   *d_suppressed = nullptr;

    // 결과 버퍼(최대 max_det)
    float *d_out_boxes = nullptr, *d_out_scores = nullptr;
    int   *d_cntSel = nullptr;
}

// --------------------- 유틸 ---------------------
static __device__ __forceinline__ float iou_xyxy(
    float ax1, float ay1, float ax2, float ay2,
    float bx1, float by1, float bx2, float by2,
    float aA,  float bA)
{
    float xx1 = fmaxf(ax1, bx1);
    float yy1 = fmaxf(ay1, by1);
    float xx2 = fminf(ax2, bx2);
    float yy2 = fminf(ay2, by2);
    float w   = fmaxf(0.f, xx2 - xx1);
    float h   = fmaxf(0.f, yy2 - yy1);
    float inter = w * h;
    float uni   = aA + bA - inter;
    return inter / (uni + 1e-6f);
}

// --------------------- Kernel 1: 후보 압축 ---------------------
// RAW[5,N] -> 후보 배열에 conf>=conf_t만 원본좌표(xyxy)로 저장. 원본 크기로 역레터박스.
__global__ void k_make_candidates(
    const float* raw, int N,
    int inW, int inH, int imgW, int imgH,
    float conf_t, int* d_cnt,
    float* x1, float* y1, float* x2, float* y2, float* s, float* area)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    float cx = raw[0 * N + i];
    float cy = raw[1 * N + i];
    float w  = raw[2 * N + i];
    float h  = raw[3 * N + i];
    float sc = raw[4 * N + i];
    if (sc < conf_t) return;

    float scale = fminf((float)inW / imgW, (float)inH / imgH);
    float pad_x = (inW - imgW * scale) * 0.5f;
    float pad_y = (inH - imgH * scale) * 0.5f;

    // model-space xyxy
    float mx1 = cx - 0.5f * w;
    float my1 = cy - 0.5f * h;
    float mx2 = cx + 0.5f * w;
    float my2 = cy + 0.5f * h;

    // inverse letterbox -> original image space
    float ox1 = (mx1 - pad_x) / scale;
    float oy1 = (my1 - pad_y) / scale;
    float ox2 = (mx2 - pad_x) / scale;
    float oy2 = (my2 - pad_y) / scale;

    ox1 = fminf(fmaxf(ox1, 0.f), (float)imgW);
    oy1 = fminf(fmaxf(oy1, 0.f), (float)imgH);
    ox2 = fminf(fmaxf(ox2, 0.f), (float)imgW);
    oy2 = fminf(fmaxf(oy2, 0.f), (float)imgH);
    if (ox2 <= ox1 || oy2 <= oy1) return;

    int idx = atomicAdd(d_cnt, 1);
    x1[idx] = ox1; y1[idx] = oy1; x2[idx] = ox2; y2[idx] = oy2; s[idx] = sc;
    area[idx] = (ox2 - ox1) * (oy2 - oy1);
}

// --------------------- Kernel 2: 점수 내림차순 In-place Sort ---------------------
// 단일 블록 odd-even sort (K<=4096 가정, max_det 훨씬 작음). K가 크더라도 conf=0.70이면 보통 수백 내.
__global__ void k_sort_desc(float* s, float* x1, float* y1, float* x2, float* y2, float* area, int K)
{
    for (int p = 0; p < K; ++p) {
        int parity = p & 1;
        int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + parity;
        if (i + 1 < K) {
            if (s[i] < s[i + 1]) {
                float ts = s[i]; s[i] = s[i + 1]; s[i + 1] = ts;
                float tx = x1[i]; x1[i] = x1[i + 1]; x1[i + 1] = tx;
                float ty = y1[i]; y1[i] = y1[i + 1]; y1[i + 1] = ty;
                float tx2= x2[i]; x2[i] = x2[i + 1]; x2[i + 1] = tx2;
                float ty2= y2[i]; y2[i] = y2[i + 1]; y2[i + 1] = ty2;
                float ta = area[i]; area[i] = area[i + 1]; area[i + 1] = ta;
            }
        }
        __syncthreads();
    }
}

// --------------------- Kernel 3: Greedy NMS (완전 GPU, 순차 but 병렬 내루프) ---------------------
__global__ void k_greedy_nms(
    const float* x1, const float* y1, const float* x2, const float* y2,
    const float* s, const float* area,
    int K, float iou_t, int max_det,
    int* suppressed, int* d_sel_cnt,
    float* out_boxes, float* out_scores)
{
    int tid = threadIdx.x;
    if (tid == 0) *d_sel_cnt = 0;
    __syncthreads();

    for (int i = 0; i < K; ++i) {
        __syncthreads();
        if (suppressed[i]) continue;

        // 선택
        if (tid == 0) {
            int w = atomicAdd(d_sel_cnt, 1);
            if (w < max_det) {
                out_boxes[w*4+0] = x1[i];
                out_boxes[w*4+1] = y1[i];
                out_boxes[w*4+2] = x2[i];
                out_boxes[w*4+3] = y2[i];
                out_scores[w]    = s[i];
            }
        }
        __syncthreads();

        // 선택된 i와의 IoU로 뒤쪽 후보 억제 (병렬 분배)
        int base = i + 1;
        for (int j = base + tid; j < K; j += blockDim.x) {
            if (suppressed[j]) continue;
            float ov = iou_xyxy(
                x1[i], y1[i], x2[i], y2[i],
                x1[j], y1[j], x2[j], y2[j],
                area[i], area[j]
            );
            if (ov > iou_t) suppressed[j] = 1;
        }
        __syncthreads();

        // 최대 개수 도달 시 조기 종료
        if (tid == 0 && (*d_sel_cnt) >= max_det) break;
        __syncthreads();
    }
}

// --------------------- Host 함수 ---------------------
bool InitRawNMSWorkspace(int max_candidates, int max_det)
{
    if (g_capN == max_candidates && g_capDet == max_det) return true;
    FreeRawNMSWorkspace();

    g_capN = max_candidates;
    g_capDet = max_det;

    auto ck = [](cudaError_t e){ return e == cudaSuccess; };

    if (!ck(cudaMalloc(&d_x1, sizeof(float)*g_capN))) return false;
    if (!ck(cudaMalloc(&d_y1, sizeof(float)*g_capN))) return false;
    if (!ck(cudaMalloc(&d_x2, sizeof(float)*g_capN))) return false;
    if (!ck(cudaMalloc(&d_y2, sizeof(float)*g_capN))) return false;
    if (!ck(cudaMalloc(&d_s,  sizeof(float)*g_capN))) return false;
    if (!ck(cudaMalloc(&d_area,sizeof(float)*g_capN))) return false;

    if (!ck(cudaMalloc(&d_cntCand, sizeof(int)))) return false;
    if (!ck(cudaMalloc(&d_suppressed, sizeof(int)*g_capN))) return false;

    if (!ck(cudaMalloc(&d_out_boxes,  sizeof(float)*g_capDet*4))) return false;
    if (!ck(cudaMalloc(&d_out_scores, sizeof(float)*g_capDet  ))) return false;
    if (!ck(cudaMalloc(&d_cntSel, sizeof(int)))) return false;

    return true;
}

void FreeRawNMSWorkspace()
{
    #define FREED(p) if (p) { cudaFree(p); p=nullptr; }
    FREED(d_x1); FREED(d_y1); FREED(d_x2); FREED(d_y2);
    FREED(d_s); FREED(d_area);
    FREED(d_cntCand); FREED(d_suppressed);
    FREED(d_out_boxes); FREED(d_out_scores); FREED(d_cntSel);
    g_capN = g_capDet = 0;
    #undef FREED
}

bool RunYoloRawNMS(
    const float* d_raw, int N,
    int inW, int inH, int imgW, int imgH,
    float conf_t, float iou_t, int max_det,
    cudaStream_t stream,
    int* h_num, float* h_boxes, float* h_scores)
{
    if (N <= 0 || max_det <= 0) return false;
    if (!InitRawNMSWorkspace(N, max_det)) return false;

    cudaMemsetAsync(d_cntCand, 0, sizeof(int), stream);

    dim3 blk(256);
    dim3 grd((N + blk.x - 1) / blk.x);
    k_make_candidates<<<grd, blk, 0, stream>>>(
        d_raw, N, inW, inH, imgW, imgH, conf_t, d_cntCand,
        d_x1, d_y1, d_x2, d_y2, d_s, d_area
    );

    // 후보 수 조회
    int K = 0;
    cudaMemcpyAsync(&K, d_cntCand, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (K <= 0) { *h_num = 0; return true; }

    // 정렬(점수 내림차순)
    int threads = 512;
    int blocks  = 1; // odd-even sort는 블록간 sync가 안되므로 한 블록 권장
    k_sort_desc<<<blocks, threads, 0, stream>>>(d_s, d_x1, d_y1, d_x2, d_y2, d_area, K);

    // NMS
    cudaMemsetAsync(d_suppressed, 0, sizeof(int)*K, stream);
    k_greedy_nms<<<1, 512, 0, stream>>>(
        d_x1, d_y1, d_x2, d_y2, d_s, d_area,
        K, iou_t, max_det, d_suppressed, d_cntSel,
        d_out_boxes, d_out_scores
    );

    int sel = 0;
    cudaMemcpyAsync(&sel, d_cntSel, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    sel = std::min(sel, max_det);

    // Host 복사
    cudaMemcpy(h_boxes,  d_out_boxes,  sizeof(float)*sel*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scores, d_out_scores, sizeof(float)*sel,   cudaMemcpyDeviceToHost);
    *h_num = sel;
    return true;
}

