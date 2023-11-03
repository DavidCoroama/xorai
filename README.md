# XorAI
A basic neural network written in C++ that can calculate the expected output of an xor between two numbers.

## Requirements
To work with this project, you'll need to have [JsonCpp](https://github.com/open-source-parsers/jsoncpp) installed for creating and loading model files.
If you are using Ubuntu, Debian, or Kali Linux, you can install JsonCpp using the following command: `sudo apt-get install libjsoncpp-dev`

## Training a Model
``` C++
#include <xorai/network.h>

int main() {
    /* Create the Inputs and Targets Datasets. */
    Dataset inputs = {
        {0.0, 0.0},     // 1: 0 XOR 0
        {0.0, 1.0},     // 2: 0 XOR 1
        {1.0, 0.0},     // 3: 1 XOR 0
        {1.0, 1.0}      // 4: 1 XOR 1
    };

    Dataset targets = {
        {1.0},          // 1: Expect 1
        {0.0},          // 2: Expect 0
        {0.0},          // 3: Expect 0
        {1.0}           // 4: Expect 1
    };

    /* Create a new network instance with 
        - 2 input layers 
        - 3 hidden layers 
        - 1 output layer
       using a learning rate of 0.5. */
    Network network((U64Array){2, 3, 1}, 0.5);

    /* Train the model with the given inputs and targets. */
    network.train(inputs, targets, 1000);

    /* Save the model to a file called `model.xorai` using 64 bit 
       floating point precision for each weight, biases and data. */
    network.save("model.xorai", 64);

    /* [NOTE]
     * The accuracy of the Neural Network is determined by 
     * the learning rate, hidden layer count, and total epochs.
     */
}
```

## Using a Model
``` C++
#include <xorai/network.h>

int main() {
    /* Load the model from the file `model.xorai`. */
    Network network("model.xorai");

    /* Test the model with the given inputs. */
    Matrix* result = network.test(1.0, 1.0);

    /* Display the result. */
    Matrix::display(result);

    /* Free the memory allocated for the result. */
    delete(result);
}
```

### Credits
This project is, at its core, a `C++` translation of the Neural Network implementation originally created by [codemoonsxyz](https://github.com/codemoonsxyz/neural-net-rs) in `Rust`.