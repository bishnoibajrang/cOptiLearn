#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../include/gd.h"

void gradient_descent(FuncPtr f, GradPtr grad, double* x0, double lr, int max_iters, double tol) {
    int i;
    for (i = 0; i < max_iters; i++) {
        double g = grad(*x0);
        double prev_x = *x0;
        *x0 = *x0 - lr * g;

        double diff = fabs(*x0 - prev_x);
        printf("Iter %3d | x = %.6f | f(x) = %.6f | grad = %.6f\n", i+1, *x0, f(*x0), g);

        if (diff < tol) {
            printf("Converged in %d iterations.\n", i+1);
            break;
        }
    }
    if (i == max_iters) {
        printf("Stopped after %d iterations (didn't converge).\n", max_iters);
    }
}


void gradient_descent_multi(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, int max_iters, double tol) {
    double* g = (double*)malloc(dim * sizeof(double));
    int i;

    for (i = 0; i < max_iters; i++) {
        grad(x, g, dim);

        double prev_sum = 0.0, new_sum = 0.0;
        for (int j = 0; j < dim; j++) {
            prev_sum += x[j] * x[j];
            x[j] = x[j] - lr * g[j];
            new_sum += x[j] * x[j];
        }

        printf("Iter %3d | f(x) = %.6f | grad_norm = %.6f\n", i + 1, f(x, dim), sqrt(new_sum));

        if (fabs(new_sum - prev_sum) < tol) {
            printf("Converged in %d iterations.\n", i + 1);
            break;
        }
    }

    if (i == max_iters) {
        printf("Did not converge within %d iterations.\n", max_iters);
    }

    free(g);
}

//  Gradient Descent with Armijo Line Search 

double norm_squared(double* v, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

void gradient_descent_armijo(FuncPtrND f, GradPtrND grad, double* x, int dim, double alpha_init, double beta, double c, int max_iters, double tol) {
    double* g = (double*)malloc(dim * sizeof(double));
    double* x_new = (double*)malloc(dim * sizeof(double));
    int i;

    for (i = 0; i < max_iters; i++) {
        grad(x, g, dim);
        double grad_norm2 = norm_squared(g, dim);
        double fx = f(x, dim);

        double alpha = alpha_init;
        while (1) {
            for (int j = 0; j < dim; j++) {
                x_new[j] = x[j] - alpha * g[j];
            }

            double fx_new = f(x_new, dim);
            if (fx_new <= fx - c * alpha * grad_norm2) {
                break;
            }
            alpha *= beta;
            if (alpha < 1e-10) break; // Prevent getting stuck
        }

        double diff = 0.0;
        for (int j = 0; j < dim; j++) {
            diff += fabs(x[j] - x_new[j]);
            x[j] = x_new[j];
        }

        printf("Iter %3d | f(x) = %.6f | alpha = %.6f\n", i + 1, f(x, dim), alpha);

        if (diff < tol) {
            printf("Converged in %d iterations.\n", i + 1);
            break;
        }
    }

    if (i == max_iters) {
        printf("Did not converge within %d iterations.\n", max_iters);
    }

    free(g);
    free(x_new);
}


// Momentum-based Gradient Descent 

void gradient_descent_momentum(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double gamma, int max_iters, double tol){
    double* g = (double*)malloc(dim * sizeof(double));
    double* v = (double*)calloc(dim, sizeof(double)); // velocity
    int i;

    for (i = 0; i < max_iters; i++) {
        grad(x, g, dim);

        double change = 0.0;
        for (int j = 0; j < dim; j++) {
            v[j] = gamma * v[j] - lr * g[j];  // update velocity
            x[j] += v[j];                     // apply velocity
            change += fabs(v[j]);
        }

        printf("Iter %3d | f(x) = %.6f | velocity_norm = %.6f\n", i + 1, f(x, dim), sqrt(norm_squared(v, dim)));

        if (change < tol) {
            printf("Converged in %d iterations.\n", i + 1);
            break;
        }
    }

    if (i == max_iters) {
        printf("Did not converge within %d iterations.\n", max_iters);
    }

    free(g);
    free(v);
}


// Adam Optimizer

void gradient_descent_adam(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double beta1, double beta2, double epsilon, int max_iters, double tol) {
    double* m = (double*)calloc(dim, sizeof(double)); // 1st moment
    double* v = (double*)calloc(dim, sizeof(double)); // 2nd moment
    double* g = (double*)malloc(dim * sizeof(double)); // gradient

    for (int t = 1; t <= max_iters; t++) {
        grad(x, g, dim);

        double change = 0.0;
        for (int i = 0; i < dim; i++) {
            // Update biased first and second moment estimates
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];

            // Bias-corrected estimates
            double m_hat = m[i] / (1 - pow(beta1, t));
            double v_hat = v[i] / (1 - pow(beta2, t));

            // Update parameter
            double delta = lr * m_hat / (sqrt(v_hat) + epsilon);
            x[i] -= delta;
            change += fabs(delta);
        }

        printf("Iter %3d | f(x) = %.6f | change = %.6f\n", t, f(x, dim), change);

        if (change < tol) {
            printf("Converged in %d iterations.\n", t);
            break;
        }
    }

    free(m);
    free(v);
    free(g);
}


//  Adagrad GD
void gradient_descent_adagrad(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double epsilon, int max_iters, double tol) {
    double* g = (double*)malloc(dim * sizeof(double));
    double* G = (double*)calloc(dim, sizeof(double)); // accumulated gradient^2

    for (int t = 1; t <= max_iters; t++) {
        grad(x, g, dim);

        double change = 0.0;
        for (int i = 0; i < dim; i++) {
            G[i] += g[i] * g[i];
            double adjusted_lr = lr / (sqrt(G[i]) + epsilon);
            double delta = adjusted_lr * g[i];
            x[i] -= delta;
            change += fabs(delta);
        }

        printf("Iter %3d | f(x) = %.6f | change = %.6f\n", t, f(x, dim), change);

        if (change < tol) {
            printf("Converged in %d iterations.\n", t);
            break;
        }
    }

    free(g);
    free(G);
}

// RMSProp
void gradient_descent_rmsprop(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double beta, double epsilon, int max_iters, double tol) {
    double* g = (double*)malloc(dim * sizeof(double));     // current grad
    double* G = (double*)calloc(dim, sizeof(double));      // moving average of g²

    for (int t = 1; t <= max_iters; t++) {
        grad(x, g, dim);

        double change = 0.0;
        for (int i = 0; i < dim; i++) {
            G[i] = beta * G[i] + (1 - beta) * g[i] * g[i]; // EMA of g²
            double adjusted_lr = lr / (sqrt(G[i]) + epsilon);
            double delta = adjusted_lr * g[i];
            x[i] -= delta;
            change += fabs(delta);
        }

        printf("Iter %3d | f(x) = %.6f | change = %.6f\n", t, f(x, dim), change);

        if (change < tol) {
            printf("Converged in %d iterations.\n", t);
            break;
        }
    }

    free(g);
    free(G);
}


//  Nesterov Accelerated Gradient (NAG)
void gradient_descent_nesterov(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double gamma, int max_iters, double tol) {
    double* v = (double*)calloc(dim, sizeof(double)); // velocity
    double* g = (double*)malloc(dim * sizeof(double));
    double* x_lookahead = (double*)malloc(dim * sizeof(double));

    for (int t = 1; t <= max_iters; t++) {
        // x_lookahead = x + gamma * v
        for (int i = 0; i < dim; i++) {
            x_lookahead[i] = x[i] + gamma * v[i];
        }

        // gradient at lookahead point
        grad(x_lookahead, g, dim);

        double change = 0.0;
        for (int i = 0; i < dim; i++) {
            v[i] = gamma * v[i] - lr * g[i];
            x[i] += v[i];
            change += fabs(v[i]);
        }

        printf("Iter %3d | f(x) = %.6f | change = %.6f\n", t, f(x, dim), change);

        if (change < tol) {
            printf("Converged in %d iterations.\n", t);
            break;
        }
    }

    free(v);
    free(g);
    free(x_lookahead);
}
