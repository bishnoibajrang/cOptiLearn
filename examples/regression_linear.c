#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/gd.h"
#include "../include/model.h"
#include "../include/dataset.h"


void run_optimizer(const char* name, void (*optimizer)(FuncPtrND, GradPtrND, double*, int, double, double, int, double), double lr, double param, Dataset* data, int dim, int max_iters, double tol) {
    double* weights = (double*)calloc(dim, sizeof(double));
    set_dataset(data);

    printf("\n--- %s ---\n", name);
    optimizer(mse_loss, mse_grad, weights, dim, lr, param, max_iters, tol);
    printf("Final Weights [%s]:", name);
    for (int i = 0; i < dim; i++) printf(" %.6f", weights[i]);
    printf("\n");

    free(weights);
}

void run_adam(Dataset* data, int dim, double lr, int max_iters, double tol) {
    double* weights = (double*)calloc(dim, sizeof(double));
    set_dataset(data);

    printf("\n--- Adam ---\n");
    gradient_descent_adam(mse_loss, mse_grad, weights, dim, lr, 0.9, 0.999, 1e-8, max_iters, tol);
    printf("Final Weights [Adam]:");
    for (int i = 0; i < dim; i++) printf(" %.6f", weights[i]);
    printf("\n");

    free(weights);
}

int main() {
    Dataset* data = create_sample_dataset();
    int dim = data->d;
    int max_iters = 1000;
    double tol = 1e-6;
    double lr = 0.1;

    run_optimizer("Momentum",
        (void*)gradient_descent_momentum, lr, 0.9, data, dim, max_iters, tol);

    run_optimizer("Nesterov",
        (void*)gradient_descent_nesterov, lr, 0.9, data, dim, max_iters, tol);

    run_adam(data, dim, lr, max_iters, tol);

    free_dataset(data);
    return 0;
}
