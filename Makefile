CC = gcc
CFLAGS = -Wall -Iinclude

SRC = src/gd.c src/model.c src/dataset.c
HEADERS = include/gd.h include/model.h include/dataset.h

EXAMPLES = \
    gd_scalar_1d \
    gd_multidim \
    optimizer_gd_armijo \
    optimizer_momentum \
    optimizer_adam \
    optimizer_adagrad \
    optimizer_rmsprop \
    optimizer_nesterov \
    regression_linear \
    regression_logistic \
    regression_softmax \
    regression_iris


.PHONY: all clean

all: $(EXAMPLES:%=run_%)

run_%: examples/%.c $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	-del /Q *.o *.exe 2>nul || exit 0
