#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/gd.h"
#include "../include/model.h"
#include "../include/dataset.h"

int main() {
    Dataset* data = load_csv("data/iris.csv", 4);
    if (!data) {
        fprintf(stderr, "Failed to load dataset\n");
        return 1;
    }

    Dataset *train = NULL, *test = NULL;
    train_test_split(data, &train, &test, 0.2);  // 80-20 split
    free_dataset(data); // No longer needed

    double* weights = calloc(train->d, sizeof(double));
    if (!weights) {
        fprintf(stderr, "Failed to allocate weights\n");
        return 1;
    }

    train_logistic(train, weights, 0.1, 1000);

    // Evaluate on test set
    int correct = 0;
    for (int i = 0; i < test->n; i++) {
        double z = 0;
        for (int j = 0; j < test->d; j++)
            z += weights[j] * test->X[i][j];

        double pred_prob = 1.0 / (1.0 + exp(-z));
        int pred = (pred_prob >= 0.5) ? 1 : 0;
        int label = (int)test->y[i];

        if (i < 10)  // Print first 10 predictions
            printf("Sample %d: pred = %.4f | class %d | label = %d\n", i, pred_prob, pred, label);

        if (pred == label)
            correct++;
    }

    printf("\n✅ Test Accuracy: %.2f%% (%d/%d)\n", 100.0 * correct / test->n, correct, test->n);

    free(weights);
    free_dataset(train);
    free_dataset(test);

    return 0;
}