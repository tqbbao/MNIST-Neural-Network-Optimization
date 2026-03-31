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
// Number of testing data (10000)
#define nTesting 10000

// Training images and labels file name
// Training images and labels file name
#define testImage "mnist/t10k-images.idx3-ubyte"
#define testLabel "mnist/t10k-labels.idx1-ubyte"
// Weights file name
#define model_fn "model/gpu-model-neural-network.dat"

void initialize_data(double **images, int *labels)
{
    FILE *f_testing_images = fopen(testImage, "rb");
    FILE *f_testing_labels = fopen(testLabel, "rb");
    if (f_testing_images == NULL || f_testing_labels == NULL)
    {
        printf("Error: file open\n");
        exit(1);
    }

    // With MINST dataset, the first 16 bytes are the header of images file and the first 8 bytes are the header of labels file, ignore them
    fseek(f_testing_images, 16, SEEK_SET);
    fseek(f_testing_labels, 8, SEEK_SET);

    // Read the training images and labels
    int i, j;
    for (i = 0; i < nTesting; i++)
    {
        // Read the label
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, f_testing_labels);
        labels[i] = (int)label;

        // Read the image
        for (j = 0; j < n_input; j++)
        {
            uint8_t pixel;
            fread(&pixel, sizeof(uint8_t), 1, f_testing_images);
            images[i][j] = (double)pixel / 255.0;
        }
    }
    // Close the files
    fclose(f_testing_images);
    fclose(f_testing_labels);
}

// Load model trained from the file
void loadModel(double ***weight, double *layer_size)
{
    FILE *f_model = fopen(model_fn, "r");
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

void print_image(double **image, int *labels, int idx)
{
    printf("+--LABEL--+: %d\n", (int)labels[idx]);
    int i, j;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            (image[idx][i * width + j] > 0.5 ? printf("1") : printf("0"));
            // printf("%.1f ", image[idx][i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Calculate the ReLU
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

// Calculate the Softmax
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

    double **images = (double **)calloc(nTesting, sizeof(double *));
    for (i = 0; i < nTesting; i++)
        images[i] = (double *)calloc(layer_size[0], sizeof(double));

    int *labels = (int *)calloc(nTesting, sizeof(int));

    double **layer_inputs = (double **)calloc(n_layers, sizeof(double *));
    for (i = 0; i < n_layers; i++)
        layer_inputs[i] = (double *)calloc(layer_size[i], sizeof(double));
    
    double **layer_outputs = (double **)calloc(n_layers, sizeof(double *));
    for (i = 0; i < n_layers; i++)
        layer_outputs[i] = (double *)calloc(layer_size[i], sizeof(double));

    double **final_output = (double **)calloc(nTesting, sizeof(double *));
    for (i = 0; i < nTesting; i++)
        final_output[i] = (double *)calloc(n_output, sizeof(double));

    initialize_data(images, labels);
    // print_image(images, labels, 0);
    // print_image(images, labels, 1);

    loadModel(weight, layer_size);

    printf("Classifying test examples...\n");
    int test_example;
    for (test_example = 0; test_example < nTesting; test_example++)
    {
        printf("Classifying test example %d of %d\r", test_example + 1, nTesting);
        int i, j, k;

        for (i = 0; i < n_input; i++)
        {
            layer_inputs[0][i] = layer_outputs[0][i] = images[test_example][i];
        }

        for (j = 0; j < layer_size[1]; j++)
        {
            layer_inputs[1][j] = 0.0;
            for (k = 0; k < layer_size[0]; k++)
            {
                layer_inputs[1][j] += weight[0][k][j] * layer_outputs[0][k];
            }
        }
        ReLU(layer_inputs[1], n_hidden1, layer_outputs[1]);

        for (j = 0; j < layer_size[2]; j++)
        {
            layer_inputs[2][j] = 0.0;
            for (k = 0; k < layer_size[1]; k++)
            {
                layer_inputs[2][j] += weight[1][k][j] * layer_outputs[1][k];
            }
        }
        ReLU(layer_inputs[2], n_hidden2, layer_outputs[2]);

        for (j = 0; j < layer_size[3]; j++)
        {
            layer_inputs[3][j] = 0.0;
            for (k = 0; k < layer_size[2]; k++)
            {
                layer_inputs[3][j] += weight[2][k][j] * layer_outputs[2][k];
            }
        }
        Softmax(layer_inputs[3], n_output, layer_outputs[3]);

        for (i = 0; i < n_output; i++)
            final_output[test_example][i] = layer_outputs[3][i];
    }
    printf("\n");
    // Find the output class for each test example
    for (test_example = 0; test_example < nTesting; test_example++)
    {
        double max = -1;
        int max_class;
        for (i = 0; i < n_output; i++)
        {
            if (final_output[test_example][i] > max)
            {
                max = final_output[test_example][i];
                max_class = i;
            }
        }
        final_output[test_example][0] = max_class;
    }

    int **confusion_matrix = (int **)calloc(n_output, sizeof(int *));
    for (i = 0; i < n_output; i++)
        confusion_matrix[i] = (int *)calloc(n_output, sizeof(int));

    // Fill the confusion matrix
    int actual_class, predicted_class;
    for (test_example = 0; test_example < nTesting; test_example++)
    {
        actual_class = labels[test_example];
        predicted_class = final_output[test_example][0];

        ++confusion_matrix[actual_class][predicted_class];
    }

    // Print the confusion matrix
    printf("\n");
    printf("\n");
    printf("\n");
    printf("\t\tPre0\tPre1\tPre2\tPre3\tPre4\tPre5\tPre6\tPre7\tPre8\tPre9\n");
    printf("\n------------------------------------------------------------------------------------------------\n");

    for (actual_class = 0; actual_class < n_output; actual_class++)
    {
        printf("Label %d | \t", actual_class);
        for (predicted_class = 0; predicted_class < n_output; predicted_class++)
            printf("%d\t", confusion_matrix[actual_class][predicted_class]);
        printf("\n");
    }

    // Find the accuracy
    double accuracy = 0.0;
    for (i = 0; i < n_output; i++)
        accuracy += confusion_matrix[i][i];
    accuracy /= nTesting;

    // Print the accuracy
    printf("\n=====> Accuracy: %.2lf\n\n", accuracy * 100);

    // Free the memory allocated in heap
    for (i = 0; i < n_output; i++)
        free(confusion_matrix[i]);
    free(confusion_matrix);

    for (i = 0; i < nTesting; i++)
        free(final_output[i]);
    free(final_output);

    for (i = 0; i < n_layers; i++)
        free(layer_inputs[i]);
    free(layer_inputs);

    for (i = 0; i < n_layers; i++)
        free(layer_outputs[i]);
    free(layer_outputs);

    for (i = 0; i < nTesting; i++)
        free(images[i]);
    free(images);

    free(labels);

    for (i = 0; i < n_layers - 1; i++)
    {
        for (j = 0; j < layer_size[i]; j++)
        {
            free(weight[i][j]);
        }
        free(weight[i]);
    }
    free(weight);

    free(layer_size);

    return 0;
}