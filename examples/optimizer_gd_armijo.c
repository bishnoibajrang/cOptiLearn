#include <stdio.h>
#include <math.h>
#include "../include/gd.h"


/*

Armijo Line Search:
f(x - α∇f(x)) ≤ f(x) - c * α * ||∇f(x)||²
Where,
f is the cost function
∇f(x) is the gradient
c is a small constant (typically 1e-4)
α starts from α0 (like 1.0) and is shrunk by factor β (like 0.5) until the condition is met.

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
    double x[2] = {0.0, 0.0};  // initial guess
    double alpha_init = 1.0;   // Start with alpha = 1.0
    double beta = 0.5;         // Shrink factor
    double c = 1e-4;           // Armijo constant
    int max_iters = 100;
    double tol = 1e-6;

    gradient_descent_armijo(func2d, grad2d, x, dim, alpha_init, beta, c, max_iters, tol);

    printf("Minimum found at x = [%.6f, %.6f]\n", x[0], x[1]);
    return 0;
}