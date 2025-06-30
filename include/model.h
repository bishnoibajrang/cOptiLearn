#ifndef MODEL_H
#define MODEL_H


#include "dataset.h"

void train_logistic(Dataset* data, double* weights, double lr, int max_iter);
double predict_sample(double* w, double* x, int d);

// Mean Squared Error: loss
double mse_loss(double* weights, int dim);

// MSE Gradient: ∇loss
void mse_grad(double* weights, double* grad_out, int dim);

// Logistic Regression
double logistic_loss(double* weights, int dim);
void logistic_grad(double* weights, double* grad_out, int dim);

//  Softmax function
double softmax_loss(double** W, int num_classes, int dim);
void softmax_grad(double** W, double** grad_out, int num_classes, int dim);



#endif
