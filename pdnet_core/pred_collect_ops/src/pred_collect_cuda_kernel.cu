/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modify from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;
const int kMaxGridNum = 2147483647;


inline int GET_BLOCKS(const int N){
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

template <typename scalar_t>
__device__ scalar_t dmcn_im2col_bilinear(const scalar_t *bottom_data, const int data_width,
                                         const int height, const int width, scalar_t h, scalar_t w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


template <typename scalar_t>
__device__ void dmcn_get_both_coordinate_weight(scalar_t argmax_h, scalar_t argmax_w,
                                               const int height, const int width, const scalar_t *im_data,
                                               const int data_width, scalar_t & weight_h, scalar_t & weight_w){
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width){
    //empty
    weight_h = 0;
    weight_w = 0;
    return;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t data_ = 0;
  weight_h = 0;
  weight_w = 0;

  if (argmax_h_low >= 0 && argmax_w_low >= 0){
    data_ = im_data[argmax_h_low * data_width + argmax_w_low];
    weight_h += -1 * (argmax_w_low + 1 - argmax_w) * data_;
    weight_w += -1 * (argmax_h_low + 1 - argmax_h) * data_;
  }

  if (argmax_h_low >= 0 && argmax_w_high <= width - 1){
    data_ = im_data[argmax_h_low * data_width + argmax_w_high];
    weight_h += -1 * (argmax_w - argmax_w_low) * data_;
    weight_w += (argmax_h_low + 1 - argmax_h) * data_;
  }
  if (argmax_h_high <= height - 1 && argmax_w_low >= 0){
    data_ = im_data[argmax_h_high * data_width + argmax_w_low];
    weight_h += (argmax_w_low + 1 - argmax_w) * data_;
    weight_w += -1 * (argmax_h - argmax_h_low) * data_;
  }
  if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1){
    data_ = im_data[argmax_h_high * data_width + argmax_w_high];
    weight_h += (argmax_w - argmax_w_low) * data_;
    weight_w += (argmax_h - argmax_h_low) * data_;
  }
  return;
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w,
                                             const int h, const int w, const int height, const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}




template <typename scalar_t>
__global__ void deformable_im2merge_gpu_kernel(const int n, const scalar_t *data_im, const scalar_t *data_location,
                                                const int batch_size, const int channels, const int height, const int width,
                                                const int channels_out, const int height_out, const int width_out,
                                                const int num_points, const int channels_offset, scalar_t *data_out) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_out = index % width_out;
    const int h_out = (index / width_out) % height_out;
    const int c_out = (index / width_out / height_out) % channels_out;
    const int b_out = (index / width_out / height_out) / channels_out;

    const int c_im = c_out;

    scalar_t *data_out_ptr = data_out + ((b_out * channels_out + c_out) * height_out + h_out) * width_out + w_out;
    const scalar_t *data_im_ptr = data_im + (b_out * channels + c_im) * height * width;
    const scalar_t *data_location_ptr = data_location + (b_out * 2 * num_points * height_out + h_out) * width_out + w_out;

    const int location_ptr_offset = height_out * width_out;
    const int im_ptr_offset = channels_offset * height * width;

    scalar_t val = static_cast<scalar_t>(0);
    for (int i = 0; i < num_points; ++i){
        const scalar_t h_im = data_location_ptr[2 * i * location_ptr_offset];
        const scalar_t w_im = data_location_ptr[(2 * i + 1) * location_ptr_offset];
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width){
          val += dmcn_im2col_bilinear(data_im_ptr + i * im_ptr_offset, width, height, width, h_im, w_im);
        }
    }
    *data_out_ptr = val;
  }
}


template <typename scalar_t>
__global__ void deformable_merge2im_coord_gpu_kernel(const int n,
                                                    const scalar_t *data_out, const scalar_t *data_im, const scalar_t *data_location,
                                                    const int batch_size, const int channels, const int height, const int width,
                                                    const int channels_out, const int height_out, const int width_out,
                                                    const int kernel_size, const int channels_offset, scalar_t *grad_location) {
  CUDA_KERNEL_LOOP(index, n){
    // num_kernels = batch_size * kernel_size * height_out * width_out
    scalar_t val_y = 0;
    scalar_t val_x = 0;
    int w = index % width_out;
    int h = (index / width_out) % height_out;
    int i = (index / width_out / height_out) % kernel_size;   // 0 ~ (kernel_size - 1)
    int b = (index / width_out / height_out) / kernel_size;
    // compute the start and end of the output

    const scalar_t *data_out_ptr = data_out + (b * channels_out * height_out + h) * width_out + w;
    const scalar_t *data_im_ptr = data_im + (b * channels + i * channels_offset) * height * width;
    const scalar_t *data_location_ptr = data_location + b * 2 * kernel_size * height_out * width_out;
    scalar_t *grad_location_ptr = grad_location + b * 2 * kernel_size * height_out * width_out;

    const int data_location_h_ptr = (((2 * i) * height_out + h) * width_out + w);
    const int data_location_w_ptr = (((2 * i + 1) * height_out + h) * width_out + w);
    const scalar_t location_h = data_location_ptr[data_location_h_ptr];
    const scalar_t location_w = data_location_ptr[data_location_w_ptr];
    scalar_t inv_h = location_h;
    scalar_t inv_w = location_w;

    const int out_ptr_offset = height_out * width_out;
    const int im_ptr_offset = height * width;
    for (int out_c = 0; out_c < channels_out; ++out_c){
      // const int out_pos = ((b * channels_out + out_c) * height_out + h) * width_out + w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      }
      scalar_t weight_y = 0;
      scalar_t weight_x = 0;
      dmcn_get_both_coordinate_weight(inv_h, inv_w, height, width, data_im_ptr + out_c * im_ptr_offset, width, weight_y, weight_x);
      const scalar_t data_out_val = data_out_ptr[out_c * out_ptr_offset];
      val_y += weight_y * data_out_val;
      val_x += weight_x * data_out_val;
    }
    grad_location_ptr[data_location_h_ptr] = val_y;
    grad_location_ptr[data_location_w_ptr] = val_x;
  }
}


template <typename scalar_t>
__global__ void deformable_merge2im_gpu_kernel(const int n,
                                                const scalar_t *data_out, const scalar_t *data_location,
                                                const int batch_size, const int channels, const int height, const int width,
                                                const int channels_out, const int height_out, const int width_out,
                                                const int kernel_size, const int channels_offset, scalar_t *grad_im) {
  CUDA_KERNEL_LOOP(index, n){
    const int w_out = index % width_out;
    const int h_out = (index / width_out) % height_out;
    const int c = index / width_out / height_out % channels_out;
    const int i = (index / width_out / height_out / channels_out) % kernel_size;
    const int b = (index / width_out / height_out / channels_out) / kernel_size;

    const scalar_t *data_location_ptr = data_location + b * 2 * kernel_size * height_out * width_out;
    scalar_t *grad_im_ptr = grad_im + (b * channels + i * channels_offset + c) * height * width;

    const int data_location_h_ptr = ((2 * i) * height_out + h_out) * width_out + w_out;
    const int data_location_w_ptr = ((2 * i + 1) * height_out + h_out) * width_out + w_out;
    const scalar_t location_h = data_location_ptr[data_location_h_ptr];
    const scalar_t location_w = data_location_ptr[data_location_w_ptr];

    const int index_out = ((b * channels_out + c) * height_out + h_out) * width_out + w_out;
    const scalar_t cur_top_grad = data_out[index_out];

    const int location_h_low = floor(location_h);
    const int location_w_low = floor(location_w);
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dx = 0; dx <= 1; ++dx) {
        const int cur_h = location_h_low + dy;
        const int cur_w = location_w_low + dx;
        if (cur_h >= 0 && cur_h < height && cur_w >= 0 && cur_w < width) {
          // interpolation weight: from the dist between [location_h, location_w], [cur_h, cur_w]
          scalar_t weight = dmcn_get_gradient_weight(location_h, location_w, cur_h, cur_w, height, width);
          atomicAdd(grad_im_ptr + cur_h * width + cur_w, weight * cur_top_grad);
        }
      }
    }
  }
}


void deformable_im2merge_cuda(
    const at::Tensor data_im, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_offset, at::Tensor data_out) {
  // num_axes should be smaller than block size

  const int num_kernels = batch_size * channels_out * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.type(), "deformable_im2merge_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_location_ = data_location.data<scalar_t>();
        scalar_t *data_out_ = data_out.data<scalar_t>();

        deformable_im2merge_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im_, data_location_,
            batch_size, channels, height_im, width_im,
            channels_out, height_out, width_out,
            kernel_size, channels_offset, data_out_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("error in deformable_im2merge_cuda: %s\n", cudaGetErrorString(err));
  }
}


void deformable_merge2im_cuda(
    const at::Tensor data_out, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_offset, at::Tensor grad_im) {

  const int num_kernels = batch_size * kernel_size * channels_out * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_out.type(), "deformable_merge2im_gpu", ([&] {
        const scalar_t *data_out_ = data_out.data<scalar_t>();
        const scalar_t *data_location_ = data_location.data<scalar_t>();
        scalar_t *grad_im_ = grad_im.data<scalar_t>();

        deformable_merge2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_out_, data_location_,
            batch_size, channels, height_im, width_im,
            channels_out, height_out, width_out,
            kernel_size, channels_offset, grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("error in deformable_merge2im_cuda: %s\n", cudaGetErrorString(err));
  }
}


void deformable_merge2im_coord_cuda(
    const at::Tensor data_out, const at::Tensor data_im, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_offset, at::Tensor grad_location){
  const int num_kernels = batch_size * kernel_size * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_out.type(), "deformable_merge2im_coord_gpu", ([&] {
        const scalar_t *data_out_ = data_out.data<scalar_t>();
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_location_ = data_location.data<scalar_t>();
        scalar_t *grad_location_ = grad_location.data<scalar_t>();

        deformable_merge2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_out_, data_im_, data_location_,
            batch_size, channels, height_im, width_im,
            channels_out, height_out, width_out,
            kernel_size, channels_offset, grad_location_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("error in deformable_merge2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}




template <typename scalar_t>
__global__ void deformable_im2concat_gpu_kernel(const int n, const scalar_t *data_im, const scalar_t *data_location,
                                                const int batch_size, const int channels, const int height, const int width,
                                                const int channels_out, const int height_out, const int width_out,
                                                const int kernel_size, const int channels_per_point, const int channels_offset, scalar_t *data_out) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_out = index % width_out;
    const int h_out = (index / width_out) % height_out;
    const int i = (index / width_out / height_out) % kernel_size;
    const int b_out = (index / width_out / height_out) / kernel_size;

    const int c_out = i * channels_per_point;
    const int c_im = i * channels_offset;

    scalar_t *data_out_ptr = data_out + ((b_out * channels_out + c_out) * height_out + h_out) * width_out + w_out;
    const scalar_t *data_im_ptr = data_im + (b_out * channels + c_im) * height * width;
    const scalar_t *data_location_ptr = data_location + (b_out * 2 * kernel_size * height_out + h_out) * width_out + w_out;

    const int location_ptr_offset = height_out * width_out;
    const scalar_t h_im = data_location_ptr[2 * i * location_ptr_offset];
    const scalar_t w_im = data_location_ptr[(2 * i + 1) * location_ptr_offset];

    const int im_ptr_offset = height * width;
    const int out_ptr_offset = height_out * width_out;
    for (int c = 0; c < channels_per_point; ++c){
        scalar_t val = static_cast<scalar_t>(0);
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width){
          val = dmcn_im2col_bilinear(data_im_ptr + c * im_ptr_offset, width, height, width, h_im, w_im);
        }
        *data_out_ptr = val;
        data_out_ptr += out_ptr_offset;
    }
  }
}



template <typename scalar_t>
__global__ void deformable_concat2im_coord_gpu_kernel(const int n,
                                                    const scalar_t *data_out, const scalar_t *data_im, const scalar_t *data_location,
                                                    const int batch_size, const int channels, const int height, const int width,
                                                    const int channels_out, const int height_out, const int width_out,
                                                    const int kernel_size, const int channels_per_point, const int channels_offset, scalar_t *grad_location) {
  CUDA_KERNEL_LOOP(index, n){
    // num_kernels = batch_size * kernel_size * height_out * width_out
    scalar_t val_y = 0;
    scalar_t val_x = 0;
    int w = index % width_out;
    int h = (index / width_out) % height_out;
    int i = (index / width_out / height_out) % kernel_size;   // 0 ~ (kernel_size - 1)
    int b = (index / width_out / height_out) / kernel_size;
    // compute the start and end of the output

    const int c_out = i * channels_per_point;
    const int c_im = i * channels_offset;

    const scalar_t *data_out_ptr = data_out + ((b * channels_out + c_out) * height_out + h) * width_out + w;
    const scalar_t *data_im_ptr = data_im + (b * channels + c_im) * height * width;
    const scalar_t *data_location_ptr = data_location + b * 2 * kernel_size * height_out * width_out;
    scalar_t *grad_location_ptr = grad_location + b * 2 * kernel_size * height_out * width_out;

    const int data_location_h_ptr = (((2 * i) * height_out + h) * width_out + w);
    const int data_location_w_ptr = (((2 * i + 1) * height_out + h) * width_out + w);
    const scalar_t location_h = data_location_ptr[data_location_h_ptr];
    const scalar_t location_w = data_location_ptr[data_location_w_ptr];
    scalar_t inv_h = location_h;
    scalar_t inv_w = location_w;

    const int out_ptr_offset = height_out * width_out;
    const int im_ptr_offset = height * width;
    for (int out_c = 0; out_c < channels_per_point; ++out_c){
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      }
      scalar_t weight_y = 0;
      scalar_t weight_x = 0;
      dmcn_get_both_coordinate_weight(inv_h, inv_w, height, width, data_im_ptr + out_c * im_ptr_offset, width, weight_y, weight_x);
      const scalar_t data_out_val = data_out_ptr[out_c * out_ptr_offset];
      val_y += weight_y * data_out_val;
      val_x += weight_x * data_out_val;
    }
    grad_location_ptr[data_location_h_ptr] = val_y;
    grad_location_ptr[data_location_w_ptr] = val_x;
  }
}



template <typename scalar_t>
__global__ void deformable_concat2im_gpu_kernel(const int n,
                                                const scalar_t *data_out, const scalar_t *data_location,
                                                const int batch_size, const int channels, const int height, const int width,
                                                const int channels_out, const int height_out, const int width_out,
                                                const int kernel_size, const int channels_per_point, const int channels_offset, scalar_t *grad_im) {
  CUDA_KERNEL_LOOP(index, n){
    const int w_out = index % width_out;
    const int h_out = (index / width_out) % height_out;
    const int c = index / width_out / height_out % channels_per_point;
    const int i = (index / width_out / height_out / channels_per_point) % kernel_size;
    const int b = (index / width_out / height_out / channels_per_point) / kernel_size;

    const int c_out = i * channels_per_point + c;
    const int c_im = i * channels_offset + c;

    const scalar_t *data_location_ptr = data_location + b * 2 * kernel_size * height_out * width_out;
    scalar_t *grad_im_ptr = grad_im + (b * channels + c_im) * height * width;

    const int data_location_h_ptr = ((2 * i) * height_out + h_out) * width_out + w_out;
    const int data_location_w_ptr = ((2 * i + 1) * height_out + h_out) * width_out + w_out;
    const scalar_t location_h = data_location_ptr[data_location_h_ptr];
    const scalar_t location_w = data_location_ptr[data_location_w_ptr];

    const int index_out = ((b * channels_out + c_out) * height_out + h_out) * width_out + w_out;
    const scalar_t cur_top_grad = data_out[index_out];

    const int location_h_low = floor(location_h);
    const int location_w_low = floor(location_w);
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dx = 0; dx <= 1; ++dx) {
        const int cur_h = location_h_low + dy;
        const int cur_w = location_w_low + dx;
        if (cur_h >= 0 && cur_h < height && cur_w >= 0 && cur_w < width) {
          // interpolation weight: from the dist between [location_h, location_w], [cur_h, cur_w]
          scalar_t weight = dmcn_get_gradient_weight(location_h, location_w, cur_h, cur_w, height, width);
          atomicAdd(grad_im_ptr + cur_h * width + cur_w, weight * cur_top_grad);
        }
      }
    }
  }
}



void deformable_im2concat_cuda(
    const at::Tensor data_im, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_per_point, const int channels_offset, at::Tensor data_out) {
  // num_axes should be smaller than block size

  const int num_kernels = batch_size * kernel_size * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.type(), "deformable_im2concat_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_location_ = data_location.data<scalar_t>();
        scalar_t *data_out_ = data_out.data<scalar_t>();

        deformable_im2concat_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im_, data_location_,
            batch_size, channels, height_im, width_im,
            channels_out, height_out, width_out,
            kernel_size, channels_per_point, channels_offset, data_out_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("error in deformable_im2concat_cuda: %s\n", cudaGetErrorString(err));
  }
}


void deformable_concat2im_cuda(
    const at::Tensor data_out, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_per_point, const int channels_offset, at::Tensor grad_im) {

  const int num_kernels = batch_size * kernel_size * channels_per_point * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_out.type(), "deformable_concat2im_gpu", ([&] {
        const scalar_t *data_out_ = data_out.data<scalar_t>();
        const scalar_t *data_location_ = data_location.data<scalar_t>();
        scalar_t *grad_im_ = grad_im.data<scalar_t>();

        deformable_concat2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_out_, data_location_,
            batch_size, channels, height_im, width_im,
            channels_out, height_out, width_out,
            kernel_size, channels_per_point, channels_offset, grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("error in deformable_concat2im_cuda: %s\n", cudaGetErrorString(err));
  }
}


void deformable_concat2im_coord_cuda(
    const at::Tensor data_out, const at::Tensor data_im, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_per_point, const int channels_offset, at::Tensor grad_location){
  const int num_kernels = batch_size * kernel_size * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_out.type(), "deformable_concat2im_coord_gpu", ([&] {
        const scalar_t *data_out_ = data_out.data<scalar_t>();
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_location_ = data_location.data<scalar_t>();
        scalar_t *grad_location_ = grad_location.data<scalar_t>();

        deformable_concat2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_out_, data_im_, data_location_,
            batch_size, channels, height_im, width_im,
            channels_out, height_out, width_out,
            kernel_size, channels_per_point, channels_offset, grad_location_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("error in deformable_concat2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}