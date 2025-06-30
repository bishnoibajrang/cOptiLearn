#include <stdio.h>
#include <math.h>
#include "../include/gd.h"


/*

Adagrad accumulates all past g²:
Gₜ = Gₜ₋₁ + gₜ²

RMSProp replaces this with an exponential moving average:
Gₜ = β * Gₜ₋₁ + (1 - β) * gₜ²
xₜ = xₜ₋₁ - α * gₜ / (sqrt(Gₜ) + ε)
Where:
α is learning rate
β is decay rate for the moving average (usually 0.9)
ε avoids division by 0 (typically 1e-8)

*/

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

    double lr = 0.01;
    double beta = 0.9;
    double epsilon = 1e-8;
    int max_iters = 1000;
    double tol = 1e-6;

    printf("Training with RMSProp Optimizer...\n");
    gradient_descent_rmsprop(func2d, grad2d, x, dim, lr, beta, epsilon, max_iters, tol);

    printf("Minimum found at x = [%.6f, %.6f]\n", x[0], x[1]);
    return 0;
}
