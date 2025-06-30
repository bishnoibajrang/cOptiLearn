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
    double x[2] = {0.0, 0.0};  // initial guess
    double learning_rate = 0.1;
    int max_iters = 100;
    double tol = 1e-6;

    gradient_descent_multi(func2d, grad2d, x, dim, learning_rate, max_iters, tol);

    printf("Minimum found at x = [%.6f, %.6f]\n", x[0], x[1]);
    return 0;
}
