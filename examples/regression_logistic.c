#include <stdio.h>
#include <stdlib.h>
#include "../include/gd.h"
#include "../include/model.h"
#include "../include/dataset.h"

int main() {
    Dataset* data = create_sample_dataset();
    set_dataset(data);

    int dim = data->d;
    double* weights = (double*)calloc(dim, sizeof(double));

    double lr = 0.1;
    int max_iters = 1000;
    double tol = 1e-6;

    printf("Training Logistic Regression with Gradient Descent...\n");
    gradient_descent_multi(logistic_loss, logistic_grad, weights, dim, lr, max_iters, tol);

    printf("Trained weights:\n");
    for (int i = 0; i < dim; i++) {
        printf("  w[%d] = %.6f\n", i, weights[i]);
    }

    printf("Predictions:\n");
    for (int i = 0; i < data->n; i++) {
        double z = 0.0;
        for (int j = 0; j < dim; j++) {
            z += data->X[i][j] * weights[j];
        }
        double pred = 1.0 / (1.0 + exp(-z));
        printf("  Sample %d: Pred = %.4f | Label = %.0f\n", i, pred, data->y[i]);
    }

    free(weights);
    free_dataset(data);
    return 0;
}
