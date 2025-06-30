#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/gd.h"
#include "../include/dataset.h"
#include "../include/model.h"

int main() {
    Dataset* data = create_sample_dataset();  // Must have multi-class labels (e.g., 0, 1, 2)
    set_dataset(data);

    int k = 3;              // Number of classes
    int d = data->d;        // Feature dimension
    int max_iters = 1000;
    double tol = 1e-6;
    double lr = 0.1;

    // Allocate weight matrix W[k][d] and gradient
    double** W = (double**)malloc(k * sizeof(double*));
    double** grad = (double**)malloc(k * sizeof(double*));
    for (int c = 0; c < k; c++) {
        W[c] = (double*)calloc(d, sizeof(double));
        grad[c] = (double*)calloc(d, sizeof(double));
    }

    for (int iter = 1; iter <= max_iters; iter++) {
        softmax_grad(W, grad, k, d);

        double change = 0.0;
        for (int c = 0; c < k; c++) {
            for (int j = 0; j < d; j++) {
                double delta = lr * grad[c][j];
                W[c][j] -= delta;
                change += fabs(delta);
            }
        }

        double loss = softmax_loss(W, k, d);
        printf("Iter %3d | loss = %.6f | change = %.6f\n", iter, loss, change);
        if (change < tol) break;
    }

    printf("Final Weights:\n");
    for (int c = 0; c < k; c++) {
        printf("Class %d: ", c);
        for (int j = 0; j < d; j++) {
            printf("%.4f ", W[c][j]);
        }
        printf("\n");
    }

    free_dataset(data);
    for (int c = 0; c < k; c++) {
        free(W[c]);
        free(grad[c]);
    }
    free(W);
    free(grad);
    return 0;
}
