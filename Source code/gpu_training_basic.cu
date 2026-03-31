#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// image size 28x28
#define width 28
#define height 28
// Number of neurons in the input
#define n_input 784 // 28*28
// Number of neurons in the hidden layer 1 (avtivation function: ReLU)
#define n_hidden1 128
// Number of neurons in the hidden layer 2 (avtivation function: ReLU)
#define n_hidden2 128
// Number of neurons in the output layer (avtivation function: Softmax)
#define n_output 10 // 0-9
// N layers
#define n_layers 4
// Number of training data (60000)

// CẨN THẬN ĐỔI TÊN FILE MODEL_FN

#define nTraining 1000
// Epochs for training (30)
#define epochs 10
// Learning rate for training (0.05)
#define learningRate 0.01
// Training images and labels file name
#define trainImage "mnist/train-images.idx3-ubyte"
#define trainLabel "mnist/train-labels.idx1-ubyte"
// Weights file name sau khi train được
#define model_fn "model/basic-model-neural-network.dat"
// Mẫu test
#define model_fn_w "model/ramdom-model-neural-network.dat"


#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void max_value_in_layerOutput(double **layer_outputs, int *labels, int training_example)
{
    int i;
    double max = layer_outputs[3][0];
    int max_idx = 0;
    for (i = 1; i < n_output; i++)
    {
        if (layer_outputs[3][i] > max)
        {
            max = layer_outputs[3][i];
            max_idx = i;
        }
    }
    printf("Predicted: %d, Actual: %d\n", max_idx, labels[training_example]);
}

void writeWeightsToFile(double *weight_A, double *weight_B, double *weight_C, double *layer_size)
{
    FILE *f = fopen(model_fn, "w");
    if (f == NULL)
    {
        printf("Error: file open\n");
        exit(1);
    }

    for (int i = 0; i < n_input; i++)
    {
        for (int j = 0; j < n_hidden1; j++)
        {
            fprintf(f, "%lf\n", weight_A[i * n_hidden1 + j]);
        }
    }

    for (int i = 0; i < n_hidden1; i++)
    {
        for (int j = 0; j < n_hidden2; j++)
        {
            fprintf(f, "%lf\n", weight_B[i * n_hidden2 + j]);
        }
    }

    for (int i = 0; i < n_hidden2; i++)
    {
        for (int j = 0; j < n_output; j++)
        {
            fprintf(f, "%lf\n", weight_C[i * n_output + j]);
        }
    }

    fclose(f);
}

void shuffle_data(double **images, int *labels)
{
    int i, j;
    srand(time(NULL));
    for (i = 0; i < nTraining; i++)
    {
        j = rand() % nTraining;
        double *temp = images[i];
        images[i] = images[j];
        images[j] = temp;
        int temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
}

void loadWeightForTest(double *weight_A, double *weight_B, double *weight_C)
{
    FILE *f_model = fopen(model_fn_w, "r");
    if (f_model == NULL)
    {
        printf("Error: file open\n");
        exit(1);
    }

    int i, j, k;
    for (int i = 0; i < n_input; i++)
    {
        for (int j = 0; j < n_hidden1; j++)
        {
            fscanf(f_model, "%lf", &weight_A[i * n_hidden1 + j]);
        }
    }

    for (int i = 0; i < n_hidden1; i++)
    {
        for (int j = 0; j < n_hidden2; j++)
        {
            fscanf(f_model, "%lf", &weight_B[i * n_hidden2 + j]);
        }
    }

    for (int i = 0; i < n_hidden2; i++)
    {
        for (int j = 0; j < n_output; j++)
        {
            fscanf(f_model, "%lf", &weight_C[i * n_output + j]);
        }
    }
    fclose(f_model);
}

void initialize_data(double **images, int *labels)
{
    FILE *f_training_images = fopen(trainImage, "rb");
    FILE *f_training_labels = fopen(trainLabel, "rb");
    if (f_training_images == NULL || f_training_labels == NULL)
    {
        printf("Error: file open\n");
        exit(1);
    }

    // With MINST dataset, the first 16 bytes are the header of images file and the first 8 bytes are the header of labels file, ignore them
    fseek(f_training_images, 16, SEEK_SET);
    fseek(f_training_labels, 8, SEEK_SET);

    // Read the training images and labels
    int i, j;
    for (i = 0; i < nTraining; i++)
    {
        // Read the label
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, f_training_labels);
        labels[i] = (int)label;

        // Read the image
        for (j = 0; j < n_input; j++)
        {
            uint8_t pixel;
            fread(&pixel, sizeof(uint8_t), 1, f_training_images);
            images[i][j] = (double)pixel / 255.0;
        }
    }
    // Close the files
    fclose(f_training_images);
    fclose(f_training_labels);
}

void initialize_weights(double *weight_A, double *weight_B, double *weight_C)
{
    srand(time(0));
    // srand(1714103880);
    double *epsilon = (double *)malloc(sizeof(double) * (n_layers - 1));
    epsilon[0] = sqrt(6.0 / (n_input + n_hidden1));
    epsilon[1] = sqrt(6.0 / (n_hidden1 + n_hidden2));
    epsilon[2] = sqrt(6.0 / (n_hidden2 + n_output));

    int i, j;
    for (i = 0; i < n_input; i++)
        for (j = 0; j < n_hidden1; j++)
            weight_A[i * n_hidden1 + j] = -epsilon[0] + ((double)rand() / ((double)RAND_MAX / (2.0 * epsilon[0])));

    for (int i = 0; i < n_hidden1; i++)
        for (int j = 0; j < n_hidden2; j++)
            weight_B[i * n_hidden2 + j] = -epsilon[1] + ((double)rand() / ((double)RAND_MAX / (2.0 * epsilon[1])));

    for (int i = 0; i < n_hidden2; i++)
        for (int j = 0; j < n_output; j++)
            weight_C[i * n_output + j] = -epsilon[2] + ((double)rand() / ((double)RAND_MAX / (2.0 * epsilon[2])));

    free(epsilon);
}

/*====================================================================================*/
// ================================================================================== //
// ================================================================================== //
/*=======================================Kernel=======================================*/

// TODO: 

__global__ void matrix_multiplication_kernel_activationReLU(int m, int n, int k, double *A, double *B, double *C, double *D)
{
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    if (Col < k && Row < m)
    {
        int i = Row * k + Col;
        double pvalue = 0.0;
        for (int j = 0; j < n; j++)
        {
            pvalue += A[Row * n + j] * B[j * k + Col];
        }
        //C[i] = pvalue;
    
        C[i] = pvalue;
        D[i] = (pvalue > 0) ? pvalue : 0;
    }
}

__global__ void matrix_multiplication_kernel(int m, int n, int k, double *A, double *B, double *C)
{
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    if (Col < k && Row < m)
    {
        int i = Row * k + Col;
        double pvalue = 0.0;
        for (int j = 0; j < n; j++)
        {
            pvalue += A[Row * n + j] * B[j * k + Col];
        }
        C[i] = pvalue;
    }
}


__global__ void ReLU(double *input, int n, double *output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        output[i] = max(0.0, input[i]);
    }
}

// __global__ void ReLU(double *input, int n, double *output)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n)
//     {
//         if (input[i] > 0)
//         {
//             output[i] = input[i];
//         }
//         else
//         {
//             output[i] = 0;
//         }
//         //output[i] = (input[i] < 0) ? 0 : input[i];
        
//     }

// }

// void ReLU_CPU(double *input, int n, double *y)
// {
//     for (int i = 0; i < n; i++)
//     {
//         if input[i] < 0 ? output[i]=0 : output[i]=input[i];
//     }
// }

// __global__ void Softmax(double *input, int n, double *output)
// {
//     __shared__ double temp[10];
//     // int row =  threadIdx.x;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n)
//     {
//         temp[i] = input[i];
//         __syncthreads();

//         double sum = 0.0;
//         for (int j = 0; j < n; j++)
//         {
//             sum += exp(temp[j]);
//         }
//         output[i] = exp(input[i]) / sum;
//     }
// }

void Softmax_CPU(double *input, int n, double *output)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += exp(input[i]);
    }
    for (int i = 0; i < n; i++)
    {
        output[i] = exp(input[i]) / sum;
    }
}


__global__ void d_softmax(double *local_gradient, double *layer_outputs, double *expected_output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        local_gradient[i] = layer_outputs[i] - expected_output[i];
    }
}


// TODO:
__global__ void kernel_matrix_transpose(double *input, double *output, int mRows, int kCols)
{
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    if (Col < kCols && Row < mRows)
    {
        // output[Col * mRows + Row] = input[Row * kCols + Col];
        output[Row * kCols + Col] = input[Col * mRows + Row];
    }
}

// TODO:
__global__ void d_relu(int m, int n, int k, double *A, double *B, double *C, double *layer_input)
{
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    if (Col < k && Row < m)
    {
        int i = Row * k + Col;
        double pvalue = 0.0;
        for (int j = 0; j < n; j++)
        {
            pvalue += A[Row * n + j] * B[j * k + Col];
        }


        // if (layer_input[i] > 0)
        //     C[i] = pvalue;
        // else if (layer_input[i] < 0)
        //     C[i] = 0;

        C[i] = (layer_input[i] > 0) ? pvalue : 0;

    }
}


// TODO:
__global__ void update_weight_kernel(int m, int n, int k, double *A, double *B, double *C)
{
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    if (Col < k && Row < m)
    {
        int i = Row * k + Col;
        double pvalue = 0.0;
        for (int j = 0; j < n; j++)
        {
            pvalue += learningRate * A[Row * n + j] * B[j * k + Col];
        }
        C[i] -= pvalue;
    }
}

/*==================================================================================*/
// ================================================================================ //
// ================================================================================ //
/*====================================ANN Method====================================*/

void update_weight(double *layer_outputs, double *local_gradient, double *weight, int left, int right)
{

    // Devive memory for layer_outputs
    double *d_layer_outputs, *d_layer_outputs_transpose;
    size_t size_d_layer_outputs = left * sizeof(double);
    size_t size_d_layer_outputs_transpose = left * sizeof(double);
    CHECK(cudaMalloc(&d_layer_outputs, size_d_layer_outputs));
    CHECK(cudaMalloc(&d_layer_outputs_transpose, size_d_layer_outputs_transpose));
    CHECK(cudaMemcpy(d_layer_outputs, layer_outputs, size_d_layer_outputs, cudaMemcpyHostToDevice));

    int mRows; // number of rows in the matrix A
    int mCols; // number of columns in the matrix A, number of rows in the matrix B
    int nRows; // number of columns in the matrix A, number of rows in the matrix B
    int kCols; // number of columns in the matrix B

    // TODO: Calculate the transpose of layer_outputs
    mRows = left;
    mCols = 1;
    dim3 blockSize_transpose(32, 32);
    dim3 gridSize_transpose((mCols - 1) / blockSize_transpose.x + 1,
                            (mRows - 1) / blockSize_transpose.y + 1);
    kernel_matrix_transpose<<<gridSize_transpose, blockSize_transpose>>>(d_layer_outputs, d_layer_outputs_transpose, mRows, mCols);
    cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // Devive memory for local_gradient
    double *d_local_gradient;
    size_t size_local_gradient = right * sizeof(double);
    CHECK(cudaMalloc(&d_local_gradient, size_local_gradient));
    CHECK(cudaMemcpy(d_local_gradient, local_gradient, size_local_gradient, cudaMemcpyHostToDevice));

    // Devive memory for weight
    double *d_weight;
    size_t size_d_weight = left * right * sizeof(double);
    CHECK(cudaMalloc(&d_weight, size_d_weight));
    CHECK(cudaMemcpy(d_weight, weight, size_d_weight, cudaMemcpyHostToDevice));

    // TODO: Calculate
    /*
        Ta cần d_layer_outputs_transpose: (1 x 128) -> (128 x 1)

        Ta có:
        (128 x 1)               * (1 x 10)          = (128 x 10)
        layer_outputs           * local_gradient    = weight
        A                       * B                 = C
        d_layer_outputs_transpose                   =
    */
    mRows = left;
    mCols = nRows = 1;
    kCols = right;
    dim3 blockSize_calculate_for_update_weight(32, 1);
    dim3 gridSize_calculate_for_update_weight((kCols - 1) / blockSize_calculate_for_update_weight.x + 1,
                                              (mRows - 1) / blockSize_calculate_for_update_weight.y + 1);

    update_weight_kernel<<<gridSize_calculate_for_update_weight, blockSize_calculate_for_update_weight>>>(mRows, mCols, kCols, d_layer_outputs_transpose, d_local_gradient, d_weight);
    cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // TODO: Copy the result back to the host
    CHECK(cudaMemcpy(weight, d_weight, size_d_weight, cudaMemcpyDeviceToHost));

    // TODO: Free device memories
    CHECK(cudaFree(d_layer_outputs));
    CHECK(cudaFree(d_layer_outputs_transpose));
    CHECK(cudaFree(d_local_gradient));
    CHECK(cudaFree(d_weight));
}

void local_gradient_hidden(double *local_gradient_output, double *local_gradient_input, double *weight, double *layer_inputs, int left, int right)
{

    // Devive memory for weight
    double *d_weight, *d_weight_transpose;
    size_t size_weight = left * right * sizeof(double);
    size_t size_weight_transpose = left * right * sizeof(double);
    CHECK(cudaMalloc(&d_weight, size_weight));
    CHECK(cudaMalloc(&d_weight_transpose, size_weight_transpose));
    CHECK(cudaMemcpy(d_weight, weight, size_weight, cudaMemcpyHostToDevice));

    // Devive memory for local_gradient_output, local_gradient_input
    double *d_local_gradient_output, *d_local_gradient_input;
    size_t size_local_gradient_output = left * sizeof(double);
    size_t size_local_gradient_input = right * sizeof(double);
    CHECK(cudaMalloc(&d_local_gradient_output, size_local_gradient_output));
    CHECK(cudaMalloc(&d_local_gradient_input, size_local_gradient_input));
    CHECK(cudaMemcpy(d_local_gradient_input, local_gradient_input, size_local_gradient_input, cudaMemcpyHostToDevice));

    // Devive memory for layer_inputs
    double *d_layer_inputs;
    size_t size_layer_inputs = left * sizeof(double);
    CHECK(cudaMalloc(&d_layer_inputs, size_layer_inputs));
    CHECK(cudaMemcpy(d_layer_inputs, layer_inputs, size_layer_inputs, cudaMemcpyHostToDevice));

    int mRows; // number of rows in the matrix A
    int mCols; // number of columns in the matrix A, number of rows in the matrix B
    int nRows; // number of columns in the matrix A, number of rows in the matrix B
    int kCols; // number of columns in the matrix B

    // TODO: Calculate the transpose of weight
    mRows = right;
    mCols = left;
    dim3 blockSize_transpose(32, 32);
    dim3 gridSize_transpose((mCols - 1) / blockSize_transpose.x + 1,
                            (mRows - 1) / blockSize_transpose.y + 1);

    kernel_matrix_transpose<<<gridSize_transpose, blockSize_transpose>>>(d_weight, d_weight_transpose, mRows, mCols);
    cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // TODO: Calculate the local gradients for the hidden layer 2
    /*
        Ta cần d_weight_transpose: (128 x 10) -> (10 x 128)

        Ta có:
        (1 x 10)                * (10 x 128)        = (1 x 128)
        local_gradient_3        * weight_C          = local_gradient_2
        A                       * B                 = C
        local_gradient_input                        = local_gradient_output

    */

    mRows = 1;
    mCols = nRows = right;
    kCols = left;
    dim3 blockSize_calculate_for_local_gradients(64, 1);
    dim3 gridSize_calculate_for_local_gradients((kCols - 1) / blockSize_calculate_for_local_gradients.x + 1,
                                                (mRows - 1) / blockSize_calculate_for_local_gradients.y + 1);
    d_relu<<<gridSize_calculate_for_local_gradients, blockSize_calculate_for_local_gradients>>>(mRows, mCols, kCols, d_local_gradient_input, d_weight_transpose, d_local_gradient_output, d_layer_inputs);
    cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // TODO: Copy the result back to the host
    CHECK(cudaMemcpy(local_gradient_output, d_local_gradient_output, size_local_gradient_output, cudaMemcpyDeviceToHost));

    // TODO: Free device memories
    CHECK(cudaFree(d_weight));
    CHECK(cudaFree(d_weight_transpose));
    CHECK(cudaFree(d_local_gradient_output));
    CHECK(cudaFree(d_local_gradient_input));
    CHECK(cudaFree(d_layer_inputs));
}

void local_gradient_output(double *local_gradient, double *layer_outputs, double *expected_output, int n)
{
    // Devive memory for local_gradient
    double *d_local_gradient, *d_layer_outputs, *d_expected_output;
    size_t size_local_gradient = n * sizeof(double);
    size_t size_layer_outputs = n * sizeof(double);
    size_t size_expected_output = n * sizeof(double);
    CHECK(cudaMalloc(&d_local_gradient, size_local_gradient));
    CHECK(cudaMalloc(&d_layer_outputs, size_layer_outputs));
    CHECK(cudaMalloc(&d_expected_output, size_expected_output));
    CHECK(cudaMemcpy(d_local_gradient, local_gradient, size_local_gradient, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_layer_outputs, layer_outputs, size_layer_outputs, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_expected_output, expected_output, size_expected_output, cudaMemcpyHostToDevice));

    // TODO: Calculate the local gradients for the output layer
    dim3 blockSize(32);
    dim3 gridSize((n - 1) / blockSize.x + 1);
    d_softmax<<<gridSize, blockSize>>>(d_local_gradient, d_layer_outputs, d_expected_output, n);
    cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // TODO: Copy the result back to the host
    CHECK(cudaMemcpy(local_gradient, d_local_gradient, size_local_gradient, cudaMemcpyDeviceToHost));

    // TODO: Free device memories
    CHECK(cudaFree(d_local_gradient));
    CHECK(cudaFree(d_layer_outputs));
    CHECK(cudaFree(d_expected_output));
}

void forward_pass(double *weight_A, double *weight_B, double *weight_C, int training_example, double **images, double **layer_inputs, double **layer_outputs)
{
    // for (int i = 0; i < n_input; i++)
    // {
    //     layer_inputs[0][i] = layer_outputs[0][i] = images[training_example][i];
    // }

    // Devive memory for weight_A, weight_B, weight_C
    double *d_weight_A, *d_weight_B, *d_weight_C;
    size_t size_weight_A = n_input * n_hidden1 * sizeof(double);
    size_t size_weight_B = n_hidden1 * n_hidden2 * sizeof(double);
    size_t size_weight_C = n_hidden2 * n_output * sizeof(double);
    CHECK(cudaMalloc(&d_weight_A, size_weight_A));
    CHECK(cudaMalloc(&d_weight_B, size_weight_B));
    CHECK(cudaMalloc(&d_weight_C, size_weight_C));
    CHECK(cudaMemcpy(d_weight_A, weight_A, size_weight_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight_B, weight_B, size_weight_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight_C, weight_C, size_weight_C, cudaMemcpyHostToDevice));

    // Devive memory for layer_inputs
    double *d_layer_inputs_1, *d_layer_inputs_2, *d_layer_inputs_3;
    //size_t size_layer_inputs_0 = n_input * sizeof(double);
    size_t size_layer_inputs_1 = n_hidden1 * sizeof(double);
    size_t size_layer_inputs_2 = n_hidden2 * sizeof(double);
    size_t size_layer_inputs_3 = n_output * sizeof(double);
    //CHECK(cudaMalloc(&d_layer_inputs_0, size_layer_inputs_0));
    CHECK(cudaMalloc(&d_layer_inputs_1, size_layer_inputs_1));
    CHECK(cudaMalloc(&d_layer_inputs_2, size_layer_inputs_2));
    CHECK(cudaMalloc(&d_layer_inputs_3, size_layer_inputs_3));
    //CHECK(cudaMemcpy(d_layer_inputs_0, images[training_example], size_layer_inputs_0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_layer_inputs_1, layer_inputs[1], size_layer_inputs_1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_layer_inputs_2, layer_inputs[2], size_layer_inputs_2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_layer_inputs_3, layer_inputs[3], size_layer_inputs_3, cudaMemcpyHostToDevice));

    // Devive memory for layer_outputs
    double *d_layer_outputs_0, *d_layer_outputs_1, *d_layer_outputs_2, *d_layer_outputs_3;
    size_t size_layer_outputs_0 = n_input * sizeof(double);
    size_t size_layer_outputs_1 = n_hidden1 * sizeof(double);
    size_t size_layer_outputs_2 = n_hidden2 * sizeof(double);
    //size_t size_layer_outputs_3 = n_output * sizeof(double);
    CHECK(cudaMalloc(&d_layer_outputs_0, size_layer_outputs_0));
    CHECK(cudaMalloc(&d_layer_outputs_1, size_layer_outputs_1));
    CHECK(cudaMalloc(&d_layer_outputs_2, size_layer_outputs_2));
    //CHECK(cudaMalloc(&d_layer_outputs_3, size_layer_outputs_3));
    CHECK(cudaMemcpy(d_layer_outputs_0, images[training_example], size_layer_outputs_0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_layer_outputs_1, layer_outputs[1], size_layer_outputs_1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_layer_outputs_2, layer_outputs[2], size_layer_outputs_2, cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(d_layer_outputs_3, layer_outputs[3], size_layer_outputs_3, cudaMemcpyHostToDevice));

    int mRows; // number of rows in the matrix A
    int mCols; // number of columns in the matrix A, number of rows in the matrix B
    int nRows; // number of columns in the matrix A, number of rows in the matrix B
    int kCols; // number of columns in the matrix B

    // TODO: Calculate for hidden layer 1 (1 x 128):(rows x columns)
    /*
        Ta có:
        (1 x 784)            * (784 x 128)      = (1 x 128)
        d_layer_outputs_0    * weight_A         = d_layer_inputs_1
        A                    * B                = C

    */
    mRows = 1;
    mCols = nRows = 784;
    kCols = 128;
    dim3 blockSize_calculate_for_hidden1(128, 1);
    dim3 gridSize_calculate_for_hidden1((kCols - 1) / blockSize_calculate_for_hidden1.x + 1,
                                        (mRows - 1) / blockSize_calculate_for_hidden1.y + 1);

    // matrix_multiplication_kernel<<<gridSize_calculate_for_hidden1, blockSize_calculate_for_hidden1>>>(mRows, nRows, kCols, d_layer_outputs_0, d_weight_A, d_layer_inputs_1);
    // cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // dim3 blockSize_calculate_for_hidden2(128);
    // dim3 gridSize_calculate_for_hidden2((kCols - 1) / blockSize_calculate_for_hidden2.x + 1);
    // ReLU<<<gridSize_calculate_for_hidden2, blockSize_calculate_for_hidden2>>>(d_layer_inputs_1, kCols, d_layer_outputs_1);


    matrix_multiplication_kernel_activationReLU<<<gridSize_calculate_for_hidden1, blockSize_calculate_for_hidden1>>>(mRows, nRows, kCols, d_layer_outputs_0, d_weight_A, d_layer_inputs_1, d_layer_outputs_1);
    cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished


    // TODO: Calculate for hidden layer 2 (1 x 128):(rows x columns)
    /*
        Ta có:
        (1 x 128)            * (128 x 128)      = (1 x 128)
        d_layer_outputs_1    * weight_B         = d_layer_inputs_2
        A                    * B                = C

    */
    mRows = 1;
    mCols = nRows = 128;
    kCols = 128;
    dim3 blockSize_calculate_for_hidden3(128, 1);
    dim3 gridSize_calculate_for_hidden3((kCols - 1) / blockSize_calculate_for_hidden3.x + 1,
                                        (mRows - 1) / blockSize_calculate_for_hidden3.y + 1);

    // matrix_multiplication_kernel<<<gridSize_calculate_for_hidden2, blockSize_calculate_for_hidden2>>>(mRows, nRows, kCols, d_layer_outputs_1, d_weight_B, d_layer_inputs_2);
    // cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // dim3 blockSize_calculate_for_hidden4(128);
    // dim3 gridSize_calculate_for_hidden4((kCols - 1) / blockSize_calculate_for_hidden4.x + 1);
    // ReLU<<<gridSize_calculate_for_hidden4, blockSize_calculate_for_hidden4>>>(d_layer_inputs_2, kCols, d_layer_outputs_2);


    matrix_multiplication_kernel_activationReLU<<<gridSize_calculate_for_hidden3, blockSize_calculate_for_hidden3>>>(mRows, nRows, kCols, d_layer_outputs_1, d_weight_B, d_layer_inputs_2, d_layer_outputs_2);
    cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // TODO: Calculate for output layer (1 x 10):(rows x columns)
    /*
        Ta có:
        (1 x 128)            * (128 x 10)      = (1 x 10)
        d_layer_outputs_2    * weight_C         = d_layer_inputs_3
        A                    * B                = C

    */
    mRows = 1;
    mCols = nRows = 128;
    kCols = 10;
    dim3 blockSize_calculate_for_output(10, 1);
    dim3 gridSize_calculate_for_output((kCols - 1) / blockSize_calculate_for_output.x + 1,
                                       (mRows - 1) / blockSize_calculate_for_output.y + 1);

    matrix_multiplication_kernel<<<gridSize_calculate_for_output, blockSize_calculate_for_output>>>(mRows, nRows, kCols, d_layer_outputs_2, d_weight_C, d_layer_inputs_3);
    cudaDeviceSynchronize(); // Synchronize to make sure that the kernel is finished

    // dim3 blockSize_calculate_for_output2(6);
    // dim3 gridSize_calculate_for_output2((kCols - 1) / blockSize_calculate_for_output2.x + 1);
    // Softmax<<<gridSize_calculate_for_output2, blockSize_calculate_for_output2>>>(d_layer_inputs_3, kCols, d_layer_outputs_3);





    // TODO: Copy the result back to the host
    CHECK(cudaMemcpy(layer_inputs[1], d_layer_inputs_1, size_layer_inputs_1, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(layer_inputs[2], d_layer_inputs_2, size_layer_inputs_2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(layer_inputs[3], d_layer_inputs_3, size_layer_inputs_3, cudaMemcpyDeviceToHost));

    Softmax_CPU(layer_inputs[3], n_output,layer_outputs[3]);

    CHECK(cudaMemcpy(layer_outputs[0], d_layer_outputs_0, size_layer_outputs_0, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(layer_outputs[1], d_layer_outputs_1, size_layer_outputs_1, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(layer_outputs[2], d_layer_outputs_2, size_layer_outputs_2, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(layer_outputs[3], d_layer_outputs_3, size_layer_outputs_3, cudaMemcpyDeviceToHost));

    // TODO: Free device memories
    CHECK(cudaFree(d_weight_A));
    CHECK(cudaFree(d_weight_B));
    CHECK(cudaFree(d_weight_C));

    //CHECK(cudaFree(d_layer_inputs_0));
    CHECK(cudaFree(d_layer_inputs_1));
    CHECK(cudaFree(d_layer_inputs_2));
    CHECK(cudaFree(d_layer_inputs_3));

    CHECK(cudaFree(d_layer_outputs_0));
    CHECK(cudaFree(d_layer_outputs_1));
    CHECK(cudaFree(d_layer_outputs_2));
    //CHECK(cudaFree(d_layer_outputs_3));
}

void back_propagation(int training_example, int *labels, double *weight_A, double *weight_B, double *weight_C, double **layer_inputs, double **layer_outputs)
{
    double *local_gradient_1;
    double *local_gradient_2;
    double *local_gradient_3;
    local_gradient_1 = (double *)calloc(n_hidden1, sizeof(double));
    local_gradient_2 = (double *)calloc(n_hidden2, sizeof(double));
    local_gradient_3 = (double *)calloc(n_output, sizeof(double));

    double *expected_output = (double *)calloc(n_output, sizeof(double));
    for (int i = 0; i < n_output; i++)
    {
        if (i == labels[training_example])
            expected_output[i] = 1.0;
        else
            expected_output[i] = 0.0;
    }

    // TODO: Calculate the local gradients 3 for the output layer
    //local_gradient_output(local_gradient_3, layer_outputs[3], expected_output, n_output);

    for (int i = 0; i < n_output; i++)
    {
        local_gradient_3[i] = layer_outputs[3][i] - expected_output[i];
    }


    // TODO: Calculate the local gradients 2 for the hidden layer 2
    local_gradient_hidden(local_gradient_2, local_gradient_3, weight_C, layer_inputs[2], n_hidden2, n_output);

    // TODO: Calculate the local gradients 1 for the hidden layer 1
    local_gradient_hidden(local_gradient_1, local_gradient_2, weight_B, layer_inputs[1], n_hidden1, n_hidden2);

    // TODO: Update the weights C
    update_weight(layer_outputs[2], local_gradient_3, weight_C, n_hidden2, n_output);

    // TODO: Update the weights B
    update_weight(layer_outputs[1], local_gradient_2, weight_B, n_hidden1, n_hidden2);

    // TODO: Update the weights A
    update_weight(layer_outputs[0], local_gradient_1, weight_A, n_input, n_hidden1);

    

    // Free memory
    free(local_gradient_1);
    free(local_gradient_2);
    free(local_gradient_3);
    free(expected_output);
}

void mlp_trainer(double *weight_A, double *weight_B, double *weight_C, double **images, int *labels, double *layer_size)
{
    double **layer_inputs = (double **)calloc(n_layers, sizeof(double *));
    for (int i = 0; i < n_layers; i++)
    {
        layer_inputs[i] = (double *)calloc(layer_size[i], sizeof(double));
    }
    double **layer_outputs = (double **)calloc(n_layers, sizeof(double *));
    for (int i = 0; i < n_layers; i++)
    {
        layer_outputs[i] = (double *)calloc(layer_size[i], sizeof(double));
    }
    GpuTimer timer;
    timer.Start();
    for (int epoch_idx = 0; epoch_idx < epochs; epoch_idx++)
    {
        //shuffle_data(images, labels);
        int training_example;
        for (int sample_idx = 0; sample_idx < nTraining; sample_idx++)
        {
            training_example = sample_idx;
            printf("Epoch: %d, Training example: %d, ", epoch_idx, training_example);
            /*
                training_example: Chỉ số của ảnh trong tập training
                weights_A: 784 x 128 (Rows x Columns)
                weights_B: 128 x 128 (Rows x Columns)
                weights_C: 128 x 10 (Rows x Columns)
                images: 1 x 784 (Rows x Columns)
                layer_inputs & layer_outputs: Chứa giá trị của các neuron trong mạng
            */
            forward_pass(weight_A, weight_B, weight_C, training_example, images, layer_inputs, layer_outputs);

            // In ra layer_outputs[3] để kiểm tra
            // printf("\n Layer_outputs[3]: ");
            // for (int i = 0; i < 10; i++)
            // {
            //     printf("%lf ", layer_outputs[3][i]);
            // }
            // printf("\n");

            /*
                training_example: Chỉ số nhãn của ảnh trong tập training
                weights_A: 784 x 128 (Rows x Columns)
                weights_B: 128 x 128 (Rows x Columns)
                weights_C: 128 x 10 (Rows x Columns)
                layer_inputs & layer_outputs: Chứa giá trị của các neuron trong mạng
            */
            back_propagation(training_example, labels, weight_A, weight_B, weight_C, layer_inputs, layer_outputs);

            // if (sample_idx % 100 == 0)
            // {
            //     writeWeightsToFile(weight_A, weight_B, weight_C , layer_size);
            // }

            max_value_in_layerOutput(layer_outputs, labels, training_example);
        }
    }
    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time : %f ms\n",time);

    writeWeightsToFile(weight_A, weight_B, weight_C, layer_size);

    printf("\nComplete the training process\n");

    // ================================Free memory================================
    for (int i = 0; i < n_layers; i++)
    {
        free(layer_inputs[i]);
        free(layer_outputs[i]);
    }
    free(layer_inputs);
    free(layer_outputs);
}

int main(int argc, char **argv)
{
    double *layer_size = (double *)calloc(n_layers, sizeof(double));
    layer_size[0] = n_input;
    layer_size[1] = n_hidden1;
    layer_size[2] = n_hidden2;
    layer_size[3] = n_output;

    double *weight_A; // 784 x 128
    double *weight_B; // 128 x 128
    double *weight_C; // 128 x 10
    weight_A = (double *)calloc(n_input * n_hidden1, sizeof(double));
    weight_B = (double *)calloc(n_hidden1 * n_hidden2, sizeof(double));
    weight_C = (double *)calloc(n_hidden2 * n_output, sizeof(double));


    // TODO: Đây là load bộ dữ liệu Weight, mà CPU và GPU đều sử dụng chung (test)
    loadWeightForTest(weight_A, weight_B, weight_C);
    // TODO: Đây là hàm khởi tạo trọng số ngẫu nhiên (thực tế)
    //initialize_weights(weight_A, weight_B, weight_C);

    int i, j;
    double **images = (double **)calloc(nTraining, sizeof(double *));
    for (i = 0; i < nTraining; i++)
    {
        images[i] = (double *)calloc(n_input, sizeof(double));
    }
    int *labels = (int *)calloc(nTraining, sizeof(int));
    initialize_data(images, labels);

    mlp_trainer(weight_A, weight_B, weight_C, images, labels, layer_size);

    // Free memory
    for (i = 0; i < nTraining; i++)
    {
        free(images[i]);
    }
    free(images);
    free(labels);
    free(weight_A);
    free(weight_B);
    free(weight_C);
    free(layer_size);

    return 0;
}