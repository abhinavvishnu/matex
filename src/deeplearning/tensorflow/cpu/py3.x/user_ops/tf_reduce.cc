#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <vector>

using namespace tensorflow;

//#include <iostream>
#include <thread>
#include <mutex>

std::mutex rmtx;

int nsize = -1;

REGISTER_OP("TfReduce")
    .Input("to_reduce: float")
    .Input("to_dummy: float")
    .Output("reduced: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
       c->set_output(0, c->input(0));
    return Status::OK();
   });

#define TS 128

class TfReduceOp : public OpKernel{
   public:
      explicit TfReduceOp(OpKernelConstruction *context) : OpKernel(context) {}
      void Compute(OpKernelContext *context) override{
          int i, j;
         const Tensor& input_tensor = context->input(0);
         //rmtx.lock();
         auto input = input_tensor.flat<float>();
         Tensor *output_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
         auto output = output_tensor->flat<float>();
         
         //output_tensor = (Tensor *)((void *)(&context->input(0)));
         //auto output = output_tensor->flat<float>();

         const int N = input.size();
         if(nsize == -1){
             MPI_Comm_size(MPI_COMM_WORLD, &nsize);
         }
/*         int parts = N / (TS * TS);
         int rems = N % (TS * TS);
         for(i = 0; i < parts; i++)
         {
              memcpy(&(output(i * (TS * TS))), &(input(i * TS * TS)), sizeof(float) * (TS * TS));
              for(j = i * TS * TS; j < (i+1) * (TS * TS); ++j){
                 output(j) /= (float)nsize;
              }
              
         } */
         
         for(j = 0; j < N; ++j){
            output(j) = input(j);
         }
        
          
         MPI_Allreduce(MPI_IN_PLACE, &(output(0)), N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
         //context->set_output(0, input_tensor);
         //rmtx.unlock();
      }
};

REGISTER_KERNEL_BUILDER(Name("TfReduce").Device(DEVICE_CPU), TfReduceOp);
