#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include "../include/dataset.h"



Dataset* load_csv(const char* filename, int features) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        perror("File error");
        return NULL;
    }

    int cap = 100;
    double** X = malloc(cap * sizeof(double*));
    double* y = malloc(cap * sizeof(double));
    int n = 0;
    char line[1024];

    while (fgets(line, sizeof(line), f)) {
        if (n >= cap) {
            cap *= 2;
            X = realloc(X, cap * sizeof(double*));
            y = realloc(y, cap * sizeof(double));
        }

        X[n] = malloc((features + 1) * sizeof(double));
        char* token = strtok(line, ",");
        for (int i = 0; i < features; i++) {
            X[n][i + 1] = atof(token); // +1 to leave room for bias
            token = strtok(NULL, ",");
        }

        
        // Read label
        if (token && strstr(token, "Setosa")) {
            // printf("%s\n",token);
            y[n] = 0;}
        else if (token && strstr(token, "Versicolor")){ 
            // printf("%s\n",token);
            y[n] = 1;}
        else {
            free(X[n]);
            continue; // Skip Virginica
        }

        X[n][0] = 1.0; // bias
        n++;
    }

    fclose(f);

    Dataset* data = malloc(sizeof(Dataset));
    data->X = X;
    data->y = y;
    data->n = n;
    data->d = features + 1;
    return data;
}


void normalize_features(Dataset* data) {
    for (int j = 1; j < data->d; j++) {
        double mean = 0, std = 0;
        for (int i = 0; i < data->n; i++) mean += data->X[i][j];
        mean /= data->n;
        for (int i = 0; i < data->n; i++) std += pow(data->X[i][j] - mean, 2);
        std = sqrt(std / data->n);

        for (int i = 0; i < data->n; i++)
            data->X[i][j] = (data->X[i][j] - mean) / (std + 1e-8);
    }
}

void free_dataset(Dataset* data) {
    if (!data) return;
    for (int i = 0; i < data->n; i++) free(data->X[i]);
    free(data->X);
    free(data->y);
    free(data);
}


// Hardcoded simple dataset (for demo)
Dataset* create_sample_dataset() {
    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    data->n = 8;
    data->d = 2; // bias + 1 feature

    data->X = (double**)malloc(data->n * sizeof(double*));
    data->y = (double*)malloc(data->n * sizeof(double));

    double raw_X[8][2] = {
        {1.0, 1.0}, {1.0, 2.0}, {1.0, 1.5}, {1.0, 0.5},
        {1.0, 4.0}, {1.0, 5.0}, {1.0, 4.5}, {1.0, 6.0}
    };

    double raw_y[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    for (int i = 0; i < data->n; i++) {
        data->X[i] = (double*)malloc(data->d * sizeof(double));
        for (int j = 0; j < data->d; j++) {
            data->X[i][j] = raw_X[i][j];
        }
        data->y[i] = raw_y[i];
    }

    return data;
}

/*
// Dataset with bias
Dataset* create_sample_dataset() {
    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    data->n = 12;
    data->d = 3; // bias + x1 + x2

    data->X = (double**)malloc(data->n * sizeof(double*));
    data->y = (double*)malloc(data->n * sizeof(double));

    // Format: {bias, x1, x2}
    double raw_X[12][3] = {
        {1.0, 1.0, 1.0},  // Class 0
        {1.0, 1.2, 0.8},
        {1.0, 0.8, 1.1},
        {1.0, 1.5, 1.5},
        {1.0, 0.5, 0.7},
        {1.0, 1.1, 1.3},
        {1.0, 4.8, 5.2},  // Class 1
        {1.0, 5.0, 5.0},
        {1.0, 5.2, 4.9},
        {1.0, 4.7, 5.3},
        {1.0, 5.3, 4.8},
        {1.0, 4.9, 5.1}
    };

    double raw_y[12] = {
        0, 0, 0, 0, 0, 0,  // Class 0
        1, 1, 1, 1, 1, 1   // Class 1
    };

    for (int i = 0; i < data->n; i++) {
        data->X[i] = (double*)malloc(data->d * sizeof(double));
        for (int j = 0; j < data->d; j++) {
            data->X[i][j] = raw_X[i][j];
        }
        data->y[i] = raw_y[i];
    }

    return data;
}
*/

void train_test_split(Dataset* full, Dataset** train_out, Dataset** test_out, double test_ratio) {
    int total = full->n;
    int test_size = (int)(total * test_ratio);
    int train_size = total - test_size;

    // Randomly shuffle indices
    int* indices = malloc(total * sizeof(int));
    for (int i = 0; i < total; i++) indices[i] = i;
    srand(time(NULL));
    for (int i = total - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    // Allocate train
    Dataset* train = malloc(sizeof(Dataset));
    train->n = train_size;
    train->d = full->d;
    train->X = malloc(train_size * sizeof(double*));
    train->y = malloc(train_size * sizeof(double));
    for (int i = 0; i < train_size; i++) {
        int idx = indices[i];
        train->X[i] = malloc(full->d * sizeof(double));
        for (int j = 0; j < full->d; j++) train->X[i][j] = full->X[idx][j];
        train->y[i] = full->y[idx];
    }

    // Allocate test
    Dataset* test = malloc(sizeof(Dataset));
    test->n = test_size;
    test->d = full->d;
    test->X = malloc(test_size * sizeof(double*));
    test->y = malloc(test_size * sizeof(double));
    for (int i = 0; i < test_size; i++) {
        int idx = indices[i + train_size];
        test->X[i] = malloc(full->d * sizeof(double));
        for (int j = 0; j < full->d; j++) test->X[i][j] = full->X[idx][j];
        test->y[i] = full->y[idx];
    }

    free(indices);
    *train_out = train;
    *test_out = test;
}
