#include <stdio.h>
#include <math.h>
#include "../include/gd.h"

double func2d(double* x, int dim) {
    return pow(x[0] - 1, 2) + pow(x[1] + 2, 2);
}

void grad2d(double* x, double* grad_out, int dim) {
    grad_out[0] = 2 * (x[0] - 1);
    grad_out[1] = 2 * (x[1] + 2);
}

int main() {
    int dim = 2;
    double x[2] = {0.0, 0.0};

    double lr = 0.1;
    double momentum = 0.9;
    int max_iters = 1000;
    double tol = 1e-6;

    printf("Training with Nesterov Accelerated Gradient...\n");
    gradient_descent_nesterov(func2d, grad2d, x, dim, lr, momentum, max_iters, tol);

    printf("Minimum found at x = [%.6f, %.6f]\n", x[0], x[1]);
    return 0;
}
