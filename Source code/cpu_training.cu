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
#define model_fn "model/cpu-model-neural-network.dat"
// Mẫu test
#define model_fn_w "model/ramdom-model-neural-network.dat"

void initialize_weights(double ***weight, double *layer_sizes)
{
    srand(time(0));
    double *epsilon = (double *)malloc(sizeof(double) * (n_layers - 1));
    epsilon[0] = sqrt(6.0 / (n_input + n_hidden1));
    epsilon[1] = sqrt(6.0 / (n_hidden1 + n_hidden2));
    epsilon[2] = sqrt(6.0 / (n_hidden2 + n_output));

    // Random initialization between [-epsilon[i], epsilon[i]] for weight[i]
    int idx, row, column;
    // n_layers = 4
    for (idx = 0; idx < n_layers - 1; idx++)
        for (row = 0; row < layer_sizes[idx]; row++)
            for (column = 0; column < layer_sizes[idx + 1]; column++)
                weight[idx][row][column] = -epsilon[idx] + ((double)rand() / ((double)RAND_MAX / (2.0 * epsilon[idx])));
    // weight[idx][row][column] = (double)rand() / RAND_MAX * sqrt(2.0 / layer_sizes[idx]) * 0.5;

    // Free the memory allocated in Heap for epsilon array
    free(epsilon);
}

void loadWeightForTest(double ***weight, double *layer_size)
{
    FILE *f_model = fopen(model_fn_w, "r");
    if (f_model == NULL)
    {
        printf("Error: file open\n");
        exit(1);
    }

    int i, j, k;
    for (i = 0; i < n_layers - 1; i++)
    {
        switch (i)
        {
        case 0:
            for (j = 0; j < n_input; j++)
            {
                for (k = 0; k < n_hidden1; k++)
                {
                    fscanf(f_model, "%lf", &weight[i][j][k]);
                }
            }
            break;
        case 1:
            for (j = 0; j < n_hidden1; j++)
            {
                for (k = 0; k < n_hidden2; k++)
                {
                    fscanf(f_model, "%lf", &weight[i][j][k]);
                }
            }
            break;
        case 2:
            for (j = 0; j < n_hidden2; j++)
            {
                for (k = 0; k < n_output; k++)
                {
                    fscanf(f_model, "%lf", &weight[i][j][k]);
                }
            }
            break;
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

void ReLU(double *x, int n, double *y)
{
    for (int i = 0; i < n; i++)
    {
        if (x[i] < 0)
        {
            x[i] = 0;
        }
        y[i] = x[i];
    }
}

void Softmax(double *x, int n, double *y)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += exp(x[i]);
    }
    for (int i = 0; i < n; i++)
    {
        y[i] = exp(x[i]) / sum;
    }
}

void d_relu(int layer_size, double *layer_input, double *layer_output, double *layer_derivative)
{
    int i;
    for (i = 0; i < layer_size; i++)
    {
        if (layer_input[i] > 0)
            layer_derivative[i] = 1;
        else if (layer_input[i] < 0)
            layer_derivative[i] = 0;
        // else                           // derivative does not exist
        //     layer_derivative[i] = 0.5; // giving arbitrary value
    }
}

void forward_pass(int training_example, double *layer_size, double **layer_inputs, double **layer_outputs, double **images, double ***weight)
{
    int i, j, k;
    for (i = 0; i < n_input; i++)
        layer_inputs[0][i] = layer_outputs[0][i] = images[training_example][i];

    for (j = 0; j < layer_size[1]; j++)
    {
        layer_inputs[1][j] = 0.0;
        for (k = 0; k < layer_size[0]; k++)
            layer_inputs[1][j] += weight[0][k][j] * layer_outputs[0][k];
    }

    ReLU(layer_inputs[1], layer_size[1], layer_outputs[1]);

    for (j = 0; j < layer_size[2]; j++)
    {
        layer_inputs[2][j] = 0.0;
        for (k = 0; k < layer_size[1]; k++)
            layer_inputs[2][j] += weight[1][k][j] * layer_outputs[1][k];
    }
    ReLU(layer_inputs[2], layer_size[2], layer_outputs[2]);

    for (j = 0; j < layer_size[3]; j++)
    {
        layer_inputs[3][j] = 0.0;
        for (k = 0; k < layer_size[2]; k++)
            layer_inputs[3][j] += weight[2][k][j] * layer_outputs[2][k];
    }

    Softmax(layer_inputs[3], layer_size[3], layer_outputs[3]);
}

void back_propagation(int training_example, double *layer_size, double **layer_inputs, double **layer_outputs, int *labels, double ***weight)
{
    int i, j, k;
    double *expected_output = (double *)calloc(n_output, sizeof(double));
    for (int i = 0; i < n_output; i++)
    {
        if (i == labels[training_example])
            expected_output[i] = 1.0;
        else
            expected_output[i] = 0.0;
    }

    double ***weight_correction = (double ***)calloc(n_layers - 1, sizeof(double **));
    for (i = 0; i < n_layers - 1; i++)
        weight_correction[i] = (double **)calloc(layer_size[i], sizeof(double *));

    for (i = 0; i < n_layers - 1; i++)
        for (j = 0; j < layer_size[i]; j++)
            weight_correction[i][j] = (double *)calloc(layer_size[i + 1], sizeof(double));

    double **local_gradient = (double **)calloc(n_layers, sizeof(double *));
    for (i = 0; i < n_layers; i++)
        local_gradient[i] = (double *)calloc(layer_size[i], sizeof(double));

    //===============
    double **layer_derivatives = (double **)calloc(n_layers, sizeof(double *));
    for (i = 0; i < n_layers; i++)
        layer_derivatives[i] = (double *)calloc(layer_size[i], sizeof(double));

    double *error_output = (double *)calloc(n_output, sizeof(double));
    for (i = 0; i < n_output; i++)
        error_output[i] = layer_outputs[3][i] - expected_output[i];

    for (i = 0; i < n_output; i++)
        local_gradient[3][i] = error_output[i];

    //===============

    d_relu(layer_size[2], layer_inputs[2], layer_outputs[2], layer_derivatives[2]);

    for (i = 0; i < layer_size[2]; i++)
    {
        double error = 0.0;
        for (j = 0; j < layer_size[3]; j++)
            error += local_gradient[3][j] * weight[2][i][j];

        local_gradient[2][i] = error * layer_derivatives[2][i];
    }

    //===============

    d_relu(layer_size[1], layer_inputs[1], layer_outputs[1], layer_derivatives[1]);

    for (i = 0; i < layer_size[1]; i++)
    {
        double error = 0.0;
        for (j = 0; j < layer_size[2]; j++)
            error += local_gradient[2][j] * weight[1][i][j];

        local_gradient[1][i] = error * layer_derivatives[1][i];
    }

    for (i = 0; i < n_output; i++)
        for (j = 0; j < layer_size[2]; j++)
            weight_correction[2][j][i] = (learningRate)*local_gradient[3][i] * layer_outputs[2][j];

    for (j = 0; j < layer_size[2]; j++)
        for (k = 0; k < layer_size[1]; k++)
            weight_correction[1][k][j] = (learningRate)*local_gradient[2][j] * layer_outputs[1][k];

    for (j = 0; j < layer_size[1]; j++)
        for (k = 0; k < layer_size[0]; k++)
            weight_correction[0][k][j] = (learningRate)*local_gradient[1][j] * layer_outputs[0][k];

    //===============

    for (i = 0; i < n_hidden2; i++)
    {
        for (j = 0; j < n_output; j++)
        {
            weight[2][i][j] -= weight_correction[2][i][j];
        }
    }

    for (i = 0; i < n_hidden1; i++)
    {
        for (j = 0; j < n_hidden2; j++)
        {
            weight[1][i][j] -= weight_correction[1][i][j];
        }
    }

    for (i = 0; i < n_input; i++)
    {
        for (j = 0; j < n_hidden1; j++)
        {
            weight[0][i][j] -= weight_correction[0][i][j];
        }
    }




    // ======================================Test====================================
    // ======================================Test====================================
    // ======================================Test====================================

    // // In ra giá trị của local_gradient[3][i]
    // printf("\n Local_gradient_3: ");
    // for (i = 0; i < n_output; i++)
    // {
    //     printf("%lf ", local_gradient[3][i]);
    // }
    // printf("\n");

    // // In ra giá trị của local_gradient[2][i], cứ mỗi 32 giá trị in ra xuống dòng
    // printf("\n Local_gradient_2: ");
    // for (i = 0; i < n_hidden2; i++)
    // {
    //     printf("%lf ", local_gradient[2][i]);
    //     if ((i + 1) % 32 == 0)
    //         printf("\n");
    // }
    // printf("\n");

    // // In ra giá trị của local_gradient[1][i], cứ mỗi 32 giá trị in ra xuống dòng
    // printf("\n Local_gradient_1: ");
    // for (i = 0; i < n_hidden1; i++)
    // {
    //     printf("%lf ", local_gradient[1][i]);
    //     if ((i + 1) % 32 == 0)
    //         printf("\n");
    // }
    // printf("\n");

    // //In ra giá trị của weight2[i][j], cứ mỗi 10 giá trị in ra xuống dòng
    // printf("\n Weight2: ");
    // for (i = 0; i < n_hidden2; i++)
    // {
    //     for (j = 0; j < n_output; j++)
    //     {
    //         printf("%lf ", weight[2][i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // // In ra giá trị của weight1[i][j], cứ mỗi 10 giá trị in ra xuống dòng
    // printf("\n Weight1: ");
    // for (i = 0; i < n_hidden1; i++)
    // {
    //     for (j = 0; j < n_hidden2; j++)
    //     {
    //         printf("%lf ", weight[1][i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // // In ra giá trị của weight0[i][j], cứ mỗi 10 giá trị in ra xuống dòng
    // printf("\n Weight0: ");
    // for (i = 0; i < n_input; i++)
    // {
    //     for (j = 0; j < n_hidden1; j++)
    //     {
    //         printf("%lf ", weight[0][i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    

    // Free the memory allocated in Heap
    for (i = 0; i < n_layers - 1; i++)
    {
        for (j = 0; j < layer_size[i]; j++)
            free(weight_correction[i][j]);
        free(weight_correction[i]);
    }
    free(weight_correction);

    for (i = 0; i < n_layers; i++)
        free(local_gradient[i]);

    for (i = 0; i < n_layers; i++)
        free(layer_derivatives[i]);

    free(layer_derivatives);

    free(error_output);

    free(expected_output);
}

void writeWeightsToFile(double ***weight, double *layer_size)
{
    FILE *f = fopen(model_fn, "w");
    if (f == NULL)
    {
        printf("Error: file open\n");
        exit(1);
    }

    int i, j, k;
    for (i = 0; i < n_layers - 1; i++)
    {
        for (j = 0; j < layer_size[i]; j++)
        {
            for (k = 0; k < layer_size[i + 1]; k++)
            {
                fprintf(f, "%lf\n", weight[i][j][k]);
            }
        }
    }

    fclose(f);
}

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

void mlp_trainer(double **images, int *labels, double ***weight, double *layer_size)
{
    double **layer_inputs = (double **)calloc(n_layers, sizeof(double *));
    for (int i = 0; i < n_layers; i++)
        layer_inputs[i] = (double *)calloc(layer_size[i], sizeof(double));

    double **layer_outputs = (double **)calloc(n_layers, sizeof(double *));
    for (int i = 0; i < n_layers; i++)
        layer_outputs[i] = (double *)calloc(layer_size[i], sizeof(double));

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    for (int epoch_idx = 0; epoch_idx < epochs; epoch_idx++)
    {
        int training_example;
        for (int sample_idx = 0; sample_idx < nTraining; sample_idx++)
        {
            training_example = sample_idx;
            printf("Epoch: %d, Training example: %d, ", epoch_idx, training_example);

            forward_pass(training_example, layer_size, layer_inputs, layer_outputs, images, weight);

            // In ra layer_outputs[3][i]
            // printf("\n Layer_outputs_3: ");
            // for (int i = 0; i < n_output; i++)
            // {
            //     printf("%lf ", layer_outputs[3][i]);
            // }
            // printf("\n");

            back_propagation(training_example, layer_size, layer_inputs, layer_outputs, labels, weight);

            max_value_in_layerOutput(layer_outputs, labels, training_example);
        }
    }
    end = clock();

    cpu_time_used = ((double) (end - start)) / (CLOCKS_PER_SEC / 1000);
    printf("Thời gian CPU đã sử dụng: %f mili giây\n", cpu_time_used);
    writeWeightsToFile(weight, layer_size);
    printf("Complete the training process\n");

    // Free the memory allocated in Heap
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

    int i, j;
    double ***weight = (double ***)calloc(n_layers - 1, sizeof(double **));
    for (i = 0; i < n_layers - 1; i++)
        weight[i] = (double **)calloc(layer_size[i], sizeof(double *));

    for (i = 0; i < n_layers - 1; i++)
        for (j = 0; j < layer_size[i]; j++)
            weight[i][j] = (double *)calloc(layer_size[i + 1], sizeof(double));

    double **images = (double **)calloc(nTraining, sizeof(double *));
    for (i = 0; i < nTraining; i++)
        images[i] = (double *)calloc(layer_size[0], sizeof(double));

    int *labels = (int *)calloc(nTraining, sizeof(int));

    initialize_data(images, labels);
    // TODO: Đây là load bộ dữ liệu Weight, mà CPU và GPU đều sử dụng chung (test)
    loadWeightForTest(weight, layer_size);
    // TODO: Đây là hàm khởi tạo trọng số ngẫu nhiên (thực tế)
    //initialize_weights(weight, layer_size);

    mlp_trainer(images, labels, weight, layer_size);

    // Free the memory allocated in Heap
    for (i = 0; i < n_layers - 1; i++)
    {
        for (j = 0; j < layer_size[i]; j++)
            free(weight[i][j]);
        free(weight[i]);
    }
    free(weight);

    for (i = 0; i < nTraining; i++)
        free(images[i]);

    free(images);

    free(labels);

    free(layer_size);

    return 0;
}