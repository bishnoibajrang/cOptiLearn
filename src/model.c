#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../include/model.h"
#include "../include/dataset.h"


// Global dataset pointer
static Dataset* g_data = NULL;

void set_dataset(Dataset* data) {
    g_data = data;
}

// y_pred = Xw
double mse_loss(double* weights, int dim) {
    if (!g_data) return -1;

    double loss = 0.0;
    for (int i = 0; i < g_data->n; i++) {
        double y_pred = 0.0;
        for (int j = 0; j < dim; j++) {
            y_pred += g_data->X[i][j] * weights[j];
        }
        double error = y_pred - g_data->y[i];
        loss += error * error;
    }
    return loss / g_data->n;
}

void mse_grad(double* weights, double* grad_out, int dim) {
    if (!g_data) return;

    for (int j = 0; j < dim; j++) grad_out[j] = 0.0;

    for (int i = 0; i < g_data->n; i++) {
        double y_pred = 0.0;
        for (int j = 0; j < dim; j++) {
            y_pred += g_data->X[i][j] * weights[j];
        }
        double error = y_pred - g_data->y[i];
        for (int j = 0; j < dim; j++) {
            grad_out[j] += 2 * error * g_data->X[i][j];
        }
    }

    for (int j = 0; j < dim; j++) {
        grad_out[j] /= g_data->n;
    }
}



// Logistic Loss + Gradient
static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double logistic_loss(double* weights, int dim) {
    if (!g_data) return -1;

    double loss = 0.0;
    for (int i = 0; i < g_data->n; i++) {
        double z = 0.0;
        for (int j = 0; j < dim; j++) {
            z += g_data->X[i][j] * weights[j];
        }
        double pred = sigmoid(z);
        double y = g_data->y[i];
        loss += -y * log(pred + 1e-8) - (1 - y) * log(1 - pred + 1e-8);
    }
    return loss / g_data->n;
}

void logistic_grad(double* weights, double* grad_out, int dim) {
    if (!g_data) return;

    for (int j = 0; j < dim; j++) grad_out[j] = 0.0;

    for (int i = 0; i < g_data->n; i++) {
        double z = 0.0;
        for (int j = 0; j < dim; j++) {
            z += g_data->X[i][j] * weights[j];
        }
        double pred = sigmoid(z);
        double error = pred - g_data->y[i];

        for (int j = 0; j < dim; j++) {
            grad_out[j] += error * g_data->X[i][j];
        }
    }

    for (int j = 0; j < dim; j++) {
        grad_out[j] /= g_data->n;
    }
}


// SoftmaxHelper
void compute_softmax(double* z, double* softmax_out, int k) {
    double max_z = z[0];
    for (int i = 1; i < k; i++) if (z[i] > max_z) max_z = z[i];

    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        softmax_out[i] = exp(z[i] - max_z);
        sum += softmax_out[i];
    }

    for (int i = 0; i < k; i++) softmax_out[i] /= sum;
}

double softmax_loss(double** W, int k, int d) {
    if (!g_data) return -1;

    double loss = 0.0;
    double* z = (double*)malloc(k * sizeof(double));
    double* prob = (double*)malloc(k * sizeof(double));

    for (int i = 0; i < g_data->n; i++) {
        // Compute z_c = W_c ⋅ x
        for (int c = 0; c < k; c++) {
            z[c] = 0.0;
            for (int j = 0; j < d; j++) {
                z[c] += W[c][j] * g_data->X[i][j];
            }
        }

        compute_softmax(z, prob, k);
        int y = (int)g_data->y[i];
        loss += -log(prob[y] + 1e-8);
    }

    free(z);
    free(prob);
    return loss / g_data->n;
}


void softmax_grad(double** W, double** grad_out, int k, int d) {
    if (!g_data) return;

    for (int c = 0; c < k; c++)
        for (int j = 0; j < d; j++)
            grad_out[c][j] = 0.0;

    double* z = (double*)malloc(k * sizeof(double));
    double* prob = (double*)malloc(k * sizeof(double));

    for (int i = 0; i < g_data->n; i++) {
        // z = W ⋅ x
        for (int c = 0; c < k; c++) {
            z[c] = 0.0;
            for (int j = 0; j < d; j++) {
                z[c] += W[c][j] * g_data->X[i][j];
            }
        }

        compute_softmax(z, prob, k);
        int y = (int)g_data->y[i];

        for (int c = 0; c < k; c++) {
            double error = prob[c] - (c == y ? 1.0 : 0.0);
            for (int j = 0; j < d; j++) {
                grad_out[c][j] += error * g_data->X[i][j];
            }
        }
    }

    for (int c = 0; c < k; c++)
        for (int j = 0; j < d; j++)
            grad_out[c][j] /= g_data->n;

    free(z);
    free(prob);
}


void train_logistic(Dataset* data, double* w, double lr, int max_iter) {
    int n = data->n;
    int d = data->d;
    double* grad = malloc(d * sizeof(double));

    for (int iter = 0; iter < max_iter; iter++) {
        // Reset gradient
        for (int j = 0; j < d; j++) grad[j] = 0;

        // Compute gradient
        for (int i = 0; i < n; i++) {
            double z = 0;
            for (int j = 0; j < d; j++) z += w[j] * data->X[i][j];
            double pred = sigmoid(z);
            double error = pred - data->y[i];
            for (int j = 0; j < d; j++) grad[j] += error * data->X[i][j];
        }

        // Update weights
        for (int j = 0; j < d; j++) w[j] -= lr * grad[j] / n;
    }

    free(grad);
}

double predict_sample(double* w, double* x, int d) {
    double z = 0;
    for (int i = 0; i < d; i++) z += w[i] * x[i];
    return sigmoid(z);
}