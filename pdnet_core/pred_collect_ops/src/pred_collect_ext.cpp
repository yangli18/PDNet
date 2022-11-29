// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c
#include <torch/extension.h>

#include <cmath>
#include <vector>

#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK
#endif


void deformable_im2merge_cuda(
    const at::Tensor data_im, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_offset, at::Tensor data_out);

void deformable_merge2im_cuda(
    const at::Tensor data_out, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_offset, at::Tensor grad_im);

void deformable_merge2im_coord_cuda(
    const at::Tensor data_out, const at::Tensor data_im, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_offset, at::Tensor grad_location);


void collect_merge_forward(
    at::Tensor input, at::Tensor bias, at::Tensor location, at::Tensor output,
    const int kernel_size, const int channels_offset,
    const bool with_bias, const bool batch_proc) {
    AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = output.size(1);
    const int height_out = location.size(2);
    const int width_out = location.size(3);

    int batch_step = 1;
    if(batch_proc)
        batch_step = batch;
    const int batch_iter = batch / batch_step;

    // resize output
    input = input.view({batch_iter, batch_step, channels, height, width});
    location = location.view({batch_iter, batch_step, -1, height_out, width_out});
    output = output.view({batch_iter, batch_step, channels_out, height_out, width_out}).zero_();

    for (int b = 0; b < batch_iter; ++b) {
        deformable_im2merge_cuda(
            input[b], location[b],
            batch_step, channels, height, width,
            channels_out, height_out, width_out,
            kernel_size, channels_offset, output[b]);
    }

    output = output.view({batch, channels_out, height_out, width_out});
    if (with_bias) {
        output += bias.view({1, bias.size(0), 1, 1});
    }
}


void collect_merge_backward(
    at::Tensor input, at::Tensor bias, at::Tensor ones, at::Tensor location,
    at::Tensor grad_input, at::Tensor grad_bias,
    at::Tensor grad_location, at::Tensor grad_output,
    const int kernel_size, const int channels_offset,
    const bool with_bias, const bool batch_proc) {
    AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
    grad_output = grad_output.contiguous();

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = grad_output.size(1);
    const int height_out = location.size(2);
    const int width_out = location.size(3);

    ones = at::ones({height_out, width_out}, input.options());

    int batch_step = 1;
    if(batch_proc)
        batch_step = batch;
    const int batch_iter = batch / batch_step;

    input = input.view({batch_iter, batch_step, channels, height, width});
    location = location.view({batch_iter, batch_step, -1, height_out, width_out});

    grad_input = grad_input.view({batch_iter, batch_step, channels, height, width});
    grad_location = grad_location.view({batch_iter, batch_step, -1, height_out, width_out});
    grad_output = grad_output.view({batch_iter, batch_step, -1, height_out, width_out});


    for (int b = 0; b < batch_iter; ++b) {
        // gradient w.r.t. input coordinate data
        deformable_merge2im_coord_cuda(
            grad_output[b], input[b], location[b],
            batch_step, channels, height, width,
            channels_out, height_out, width_out,
            kernel_size, channels_offset, grad_location[b]);

        // gradient w.r.t. input data
        deformable_merge2im_cuda(
            grad_output[b], location[b],
            batch_step, channels, height, width,
            channels_out, height_out, width_out,
            kernel_size, channels_offset, grad_input[b]);

        if (with_bias){
            //grad_bias.copy_(grad_output.sum_to_size({1, 1, channels_out, 1, 1}).view(-1));
            //grad_bias.add_(grad_output.sum_to_size({1, 1, channels_out, 1, 1}).view(-1));
            grad_bias =
                grad_bias
                    .view({-1, 1})
                    .addmm_(grad_output[b].view({-1, height_out * width_out}), ones.view({-1, 1}))
                    .view(-1);
        }
    }
}




void deformable_im2concat_cuda(
    const at::Tensor data_im, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_per_point, const int channels_offset, at::Tensor data_out);

void deformable_concat2im_cuda(
    const at::Tensor data_out, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_per_point, const int channels_offset, at::Tensor grad_im);

void deformable_concat2im_coord_cuda(
    const at::Tensor data_out, const at::Tensor data_im, const at::Tensor data_location,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int channels_out, const int height_out, const int width_out,
    const int kernel_size, const int channels_per_point, const int channels_offset, at::Tensor grad_location);


void collect_concat_forward(
    at::Tensor input, at::Tensor bias, at::Tensor location, at::Tensor output,
    const int kernel_size, const int channels_offset,
    const bool with_bias, const bool batch_proc) {
    AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = output.size(1);
    const int height_out = location.size(2);
    const int width_out = location.size(3);

    int batch_step = 1;
    if(batch_proc)
        batch_step = batch;
    const int batch_iter = batch / batch_step;

    // resize output
    input = input.view({batch_iter, batch_step, channels, height, width});
    location = location.view({batch_iter, batch_step, -1, height_out, width_out});
    output = output.view({batch_iter, batch_step, channels_out, height_out, width_out}).zero_();

    const int channels_per_point = channels_out / kernel_size;
    for (int b = 0; b < batch_iter; ++b) {
        deformable_im2concat_cuda(
            input[b], location[b],
            batch_step, channels, height, width,
            channels_out, height_out, width_out,
            kernel_size, channels_per_point, channels_offset, output[b]);
    }

    output = output.view({batch, channels_out, height_out, width_out});
    if (with_bias) {
        output += bias.view({1, bias.size(0), 1, 1});
    }
}


void collect_concat_backward(
    at::Tensor input, at::Tensor bias, at::Tensor ones, at::Tensor location,
    at::Tensor grad_input, at::Tensor grad_bias,
    at::Tensor grad_location, at::Tensor grad_output,
    const int kernel_size, const int channels_offset,
    const bool with_bias, const bool batch_proc) {
    AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
    grad_output = grad_output.contiguous();

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = grad_output.size(1);
    const int height_out = location.size(2);
    const int width_out = location.size(3);

    ones = at::ones({height_out, width_out}, input.options());

    int batch_step = 1;
    if(batch_proc)
        batch_step = batch;
    const int batch_iter = batch / batch_step;

    input = input.view({batch_iter, batch_step, channels, height, width});
    location = location.view({batch_iter, batch_step, -1, height_out, width_out});

    grad_input = grad_input.view({batch_iter, batch_step, channels, height, width});
    grad_location = grad_location.view({batch_iter, batch_step, -1, height_out, width_out});
    grad_output = grad_output.view({batch_iter, batch_step, -1, height_out, width_out});

    const int channels_per_point = channels_out / kernel_size;
    for (int b = 0; b < batch_iter; ++b) {
        // gradient w.r.t. input coordinate data
        deformable_concat2im_coord_cuda(
            grad_output[b], input[b], location[b],
            batch_step, channels, height, width,
            channels_out, height_out, width_out,
            kernel_size, channels_per_point, channels_offset, grad_location[b]);

        // gradient w.r.t. input data
        deformable_concat2im_cuda(
            grad_output[b], location[b],
            batch_step, channels, height, width,
            channels_out, height_out, width_out,
            kernel_size, channels_per_point, channels_offset, grad_input[b]);

        if (with_bias){
            //grad_bias.copy_(grad_output.sum_to_size({1, 1, channels_out, 1, 1}).view(-1));
            //grad_bias.add_(grad_output.sum_to_size({1, 1, channels_out, 1, 1}).view(-1));
            grad_bias =
                grad_bias
                    .view({-1, 1})
                    .addmm_(grad_output[b].view({-1, height_out * width_out}), ones.view({-1, 1}))
                    .view(-1);
        }
    }
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("collect_merge_forward",
        &collect_merge_forward,
        "collect merge forward (CUDA)");
    m.def("collect_merge_backward",
        &collect_merge_backward,
        "collect merge backward (CUDA)");
    m.def("collect_concat_forward",
        &collect_concat_forward,
        "collect concat forward (CUDA)");
    m.def("collect_concat_backward",
        &collect_concat_backward,
        "collect concat backward (CUDA)");
}