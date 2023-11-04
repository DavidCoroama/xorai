# XorAI
A basic neural network written in C++ that can calculate the expected output of an xor between two numbers.

## Requirements
To work with this project, you'll need to have [JsonCpp](https://github.com/open-source-parsers/jsoncpp) installed for creating and loading model files.
If you are using Ubuntu, Debian, or Kali Linux, you can install JsonCpp using the following command: `sudo apt-get install libjsoncpp-dev`

If you intend to use 128-bit floating-point precision, ensure that your computer supports the `__float128` type 
and that you have the `quadmath` library installed. Also, make sure to uncomment the `__F128_SUPPORT__` flag in the `xorai/types.h` file. 
By default, it is disabled.

## Training a Model
``` C++
#include <xorai/network.h>

int main() {
    /* Create the Inputs and Targets Datasets. */
    Dataset<f64> inputs = {
        {0.0, 0.0},     // 1: 0 XOR 0
        {0.0, 1.0},     // 2: 0 XOR 1
        {1.0, 0.0},     // 3: 1 XOR 0
        {1.0, 1.0}      // 4: 1 XOR 1
    };

    Dataset<f64> targets = {
        {1.0},          // 1: Expect 1
        {0.0},          // 2: Expect 0
        {0.0},          // 3: Expect 0
        {1.0}           // 4: Expect 1
    };

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
```

## Using a Model
``` C++
#include <xorai/network.h>

int main() {
    /* Load the model from the file `model.xorai`. */
    Network<f64> network("model.xorai");

    /* Test the model with the given inputs. */
    Matrix<f64>* result = network.test(1.0, 1.0);

    /* Display the result. */
    Matrix<f64>::display(result);

    /* Free the memory allocated for the result. */
    delete(result);
}
```

## Things to Note
The accuracy of the Neural Network is influenced by several key factors, 
including the learning rate, the number of hidden layers, 
and the total number of training epochs.

The training speed of the model is determined 
by the bit size of the floating-point numbers used, 
the number of epochs, and the total number of hidden layers.

With the default settings provided, you can anticipate the model producing 
results around `0.5372382`. For increased accuracy, consider adjusting the 
bit size to 64-bits, increasing the total number of hidden layers to around 
`9999` for faster results or `99999` for higher precision, 
and setting the total epochs to `1000`.

These adjustments should lead to a result closer to `0.9782378`.
Although not perfect, it is a significant improvement over the default settings.

### Credits
This project is, at its core, a `C++` translation of the Neural Network implementation originally created by [codemoonsxyz](https://github.com/codemoonsxyz/neural-net-rs) in `Rust`.