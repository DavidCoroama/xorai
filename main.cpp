#include <xorai/network.h>

/* Uncomment the following line to train the model. */
//#define TRAIN_MODEL_EXAMPLE

/* Uncomment the following line to use the model. */
//#define USE_MODEL_EXAMPLE

void train_model(Dataset<f64>& inputs, Dataset<f64>& targets)
{
    /* Create a new neural network with the following architecture:
     *   - 2 input layers
     *   - 3 hidden layers
     *   - 1 output layer
     * The network is configured with a learning rate of 0.5
     * and uses 64-bit floating-point precision. */
    Network<f64> network((U64Array){2, 3, 1}, 0.5);

    /* Train the model with the given inputs and targets. */
    network.train(inputs, targets, 1000);

    /* Save the model to a file named `model.xorai` using the
    highest precision available for 64-bit floating-point
    representation for each weight, bias, and data object. */
    network.save("model.xorai", UseMaxPrecision(64));
}

void use_model()
{
    /* Load the model from the file `model.xorai`. */
    Network<f64> network("model.xorai");

    /* Test the model with the given inputs. */
    Matrix<f64>* result = network.test(1.0, 1.0);

    /* Display the result. */
    Matrix<f64>::display(result);

    /* Free the memory allocated for the result. */
    delete(result);
}

int main()
{
#ifdef TRAIN_MODEL_EXAMPLE
    /* Create the Inputs and Targets Datasets. */
    Dataset<f64> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    Dataset<f64> targets = {
        {1.0},
        {0.0},
        {0.0},
        {1.0}
    };

    train_model(inputs, targets);
#endif

#ifdef USE_MODEL_EXAMPLE
    use_model();
#endif

    return 0;
}