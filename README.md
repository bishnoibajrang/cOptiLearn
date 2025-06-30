# 🔧 First-Order Optimization Algorithms in C

An efficient and modular C-based library for implementing first-order optimization techniques widely used in machine learning and numerical optimization. From gradient descent variants like Momentum, Nesterov Accelerated Gradient (NAG), Adam, Adagrad, and RMSProp to training models like linear and logistic regression — this project builds everything from scratch without external libraries.

🧠 Ideal for educational, research, and performance-critical applications that need full control over optimization logic in pure C.

## 🚀 Key Features
- ✅ Scalar and multi-dimensional gradient descent  
- ✅ Adaptive learning algorithms (Adam, Adagrad, RMSProp)  
- ✅ Line search using Armijo rule  
- ✅ Logistic & Linear Regression using all optimizers  
- ✅ Clean modular design using headers and source separation  
- ✅ Easy benchmarking and comparison across optimizers  
- ✅ Works on real datasets (e.g., Iris CSV)  

## 📂 Project Structure

```
.
├── include/          # Header files (declarations)
├── src/              # Source files (core logic)
├── examples/         # Usage examples, regressions, optimizer tests
├── data/             # CSV datasets (e.g., iris.csv)
├── Makefile          # Build automation
└── README.md         # You're here!
```

## 🧱 Requirements

- 🖥️ GCC compiler (or compatible)
- 🔧 Make (optional, for automation)
- 🧼 No external libraries — pure C

## ⚙️ Build & Run

### 🔨 Build All

```bash
make
```

### ▶️ Run Examples

```bash
make run_regression_logistic
make run_optimizer_adam
make run_gd_scalar_1d
```

Or run manually:

```bash
./run_regression_logistic
```

## 📌 Optimizers Included

| Optimizer             | File                         | Description                              |
|-----------------------|------------------------------|------------------------------------------|
| Gradient Descent      | gd_scalar_1d.c, gd_multidim.c| Basic optimization on scalar and vector functions |
| Armijo Line Search    | optimizer_gd_armijo.c        | Adaptive step size with Armijo backtracking |
| Momentum              | optimizer_momentum.c         | Velocity-based accelerated descent       |
| Nesterov (NAG)        | optimizer_nesterov.c         | Look-ahead gradient update               |
| Adagrad               | optimizer_adagrad.c          | Adaptive learning rate per parameter     |
| RMSProp               | optimizer_rmsprop.c          | Smoothed gradient-based learning rate    |
| Adam                  | optimizer_adam.c             | Combines Momentum + RMSProp              |

## 🧮 Machine Learning Models

| Model                | File                    | Highlights                                |
|----------------------|-------------------------|--------------------------------------------|
| Linear Regression    | regression_linear.c     | Train with any optimizer                   |
| Logistic Regression  | regression_logistic.c   | Binary classification using sigmoid        |
| Softmax Regression   | regression_softmax.c    | Multiclass classification                  |
| Iris Dataset Classifier | regression_iris.c    | Train/test split with Iris CSV (binary)    |

## 📊 Example Output

```bash
./run_regression_linear
```

```yaml
--- Momentum Optimizer ---
Iter 1000 | Loss: 0.00023
Weights: [0.9987, 1.0021]

--- Adam Optimizer ---
Iter 1000 | Loss: 0.00012
Weights: [1.0001, 0.9998]
```

## 🧪 Real Dataset Support

- Load CSV files like iris.csv for model training and testing.
- Supports classification labels like "Setosa", "Versicolor"
- Includes train-test split and normalization
- Accurate label parsing and memory management

## 🧭 Roadmap

- [x] First-order optimizers in modular C
- [x] Linear & logistic regression from scratch
- [x] Softmax classifier with real CSV data
- [ ] Batch & stochastic gradient variants
- [ ] Export results as CSV
- [ ] Integration with plotting tools (Python / Gnuplot)
- [ ] Second-order optimizers (Newton, BFGS, etc.)

## 👤 Author

**Bishnoi Bajrang**  
Passionate about low-level performance, numerical optimization, and building machine learning from the ground up.

🔗 GitHub   📧 bishnoibajrang502@gmail.com

## 📜 License

This project is licensed under the MIT License — free to use, modify, and distribute.

## 🤝 Contributions Welcome!

PRs, issues, and forks are welcome. Whether it's adding a new optimizer or improving existing logic — contributions make this project better.

## 🏷️ SEO Tags

*C Gradient Descent · Momentum in C · Optimization Algorithms in C · Machine Learning in C · Adam Optimizer in C · Logistic Regression Pure C · RMSProp Implementation C · Armijo Line Search C · Iris Dataset C Project · ML from Scratch C*
=======
# cOptiLearn
Optimization Algorithms Implemented from Scratch in C
>>>>>>> e4870c443bbdc6bbfeb8d6f7224fd60475bb8367
