#ifndef DATASET_H
#define DATASET_H

typedef struct {
    int n;      // number of samples
    int d;      // number of features (+1 for bias if added)
    double** X; // features
    double* y;  // target (for classification: 0, 1, ..., k-1)
} Dataset;

Dataset* load_csv_dataset(const char* filename, int feature_count, int has_header, int classification);
void free_dataset(Dataset* data);
void normalize_features(Dataset* data);
void add_bias_column(Dataset* data);  // x[0] = 1.0 style

Dataset* create_sample_dataset();
void set_dataset(Dataset* data);
void train_test_split(Dataset* full, Dataset** train, Dataset** test, double test_ratio);


#endif
