#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>
#include <mpi.h>
#include <iostream>

using namespace tensorflow;
int nsize = -1;

// .Input("to_dummy: float")
REGISTER_OP("TfReduce")
    .Input("to_reduce: float")
    .Input("to_dummy: float")
    .Output("reduced: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
       c->set_output(0, c->input(0));
    return Status::OK();
   });

// #define TS 128

#include "tensorflow/core/framework/op_kernel.h"
class TfReduceOp : public OpKernel{
   public:
      explicit TfReduceOp(OpKernelConstruction *context)
          : OpKernel(context) {}

      void Compute(OpKernelContext *context) override{
         int i, j;
         const Tensor& input_tensor = context->input(0);
         auto input = input_tensor.flat<float>();
         Tensor *output_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
         auto output = output_tensor->flat<float>();
         const int in_size = input.size();
         if(nsize == -1){
             MPI_Comm_size(MPI_COMM_WORLD, &nsize);
         }

         float * outArr = (float*)malloc(sizeof(float) * in_size);

         for(j = 0; j < in_size; ++j){
            outArr[j] = input(j);
         }

         MPI_Allreduce(MPI_IN_PLACE, outArr, in_size, MPI::FLOAT, MPI_SUM, MPI_COMM_WORLD);
         for(j = 0; j < in_size; ++j){
            output(j) = outArr[j];
         }
         free(outArr);
      }
};

REGISTER_KERNEL_BUILDER(Name("TfReduce").Device(DEVICE_CPU), TfReduceOp);
