#include <stdio.h>
#include <math.h>
#include "../include/gd.h"

// Function: f(x) = (x - 3)^2
double f(double x) {
    return (x - 3) * (x - 3);
}

// Derivative: f'(x) = 2*(x - 3)
double df(double x) {
    return 2 * (x - 3);
}

int main() {
    double x0 = 0.0;  // initial guess
    double learning_rate = 0.1;
    int max_iters = 100;
    double tol = 1e-6;

    gradient_descent(f, df, &x0, learning_rate, max_iters, tol);

    printf("Minimum found at x = %.6f\n", x0);
    return 0;
}
