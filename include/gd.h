#ifndef GD_H
#define GD_H

typedef double (*FuncPtr)(double);
typedef double (*GradPtr)(double);

typedef double (*FuncPtrND)(double* x, int dim);
typedef void (*GradPtrND)(double* x, double* grad_out, int dim);

// 1D
void gradient_descent(FuncPtr f, GradPtr grad, double* x0, double lr, int max_iters, double tol);

// nD
void gradient_descent_multi(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, int max_iters, double tol);

// Armijo Line Search
void gradient_descent_armijo(FuncPtrND f, GradPtrND grad, double* x, int dim, double alpha_init, double beta, double c, int max_iters, double tol);

// Momentum-based Gradient Descent
void gradient_descent_momentum(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double momentum, int max_iters, double tol);

//  Adam Optimizer
void gradient_descent_adam(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double beta1, double beta2, double epsilon, int max_iters, double tol);


// Adagrad GD
void gradient_descent_adagrad(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double epsilon, int max_iters, double tol);

// RMSProp
void gradient_descent_rmsprop(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double beta, double epsilon, int max_iters, double tol);

//  Nesterov Accelerated Gradient (NAG)
void gradient_descent_nesterov(FuncPtrND f, GradPtrND grad, double* x, int dim, double lr, double momentum, int max_iters, double tol);


#endif
