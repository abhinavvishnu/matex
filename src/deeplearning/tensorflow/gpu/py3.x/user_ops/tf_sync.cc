#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>
#include <mpi.h>
using namespace tensorflow;

/* Artifical operator generated for sync across nodes */
/* Copies the first input to the output. No changes are made */

REGISTER_OP("TfSync")
    .Input("to_sync_new: float")
    .Input("to_sync_old: float")
    .Output("synced: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
       c->set_output(0, c->input(0));
    return Status::OK();
   });

class TfSyncOp : public OpKernel{
   public:
      explicit TfSyncOp(tensorflow::OpKernelConstruction *context)
          : OpKernel(context) {}

      void Compute(tensorflow::OpKernelContext *context) override{
         const tensorflow::Tensor& input_tensor = context->input(0);
         auto input = input_tensor.flat<float>();
         tensorflow::Tensor *output_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
         auto output = output_tensor->flat<float>();
         const int N = input.size();
         MPI_Barrier(MPI_COMM_WORLD);
         for(int i = 0; i < N; ++i){
             output(i) = input(i);
         }
      }
};

REGISTER_KERNEL_BUILDER(Name("TfSync").Device(tensorflow::DEVICE_CPU), TfSyncOp);
