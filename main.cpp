#include <xorai/network.h>

/* Uncomment the following line to train the model. */
//#define TRAIN_MODEL_EXAMPLE

/* Uncomment the following line to use the model. */
//#define USE_MODEL_EXAMPLE

void train_model(Dataset& inputs, Dataset& targets)
{
	/* Create a new network instance with 
		- 2 input layers 
		- 3 hidden layers 
		- 1 output layer
	   Also using a learning rate of 0.5. */
	Network network((U64Array){2, 3, 1}, 0.5);

	/* Train the model with the given inputs and targets. */
    network.train(inputs, targets, 1000);

	/* Save the model to a file called `model.xorai` using 64 bit 
	   floating point precision for each weight, biases and data. */
    network.save("model.xorai", 64);
}

void use_model()
{
	/* Load the model from the file `model.xorai`. */
	Network network("model.xorai");

	/* Test the model with the given inputs. */
	Matrix* result = network.test(1.0, 1.0);

	/* Display the result. */
	Matrix::display(result);

	/* Free the memory allocated for the result. */
	delete(result);
}

int main()
{
    Dataset inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    Dataset targets = {
		{1.0}, 
		{0.0}, 
		{0.0}, 
		{1.0}
	};

    #if defined(TRAIN_MODEL_EXAMPLE)
		train_model(inputs, targets);
	#endif
	
	#if defined(USE_MODEL_EXAMPLE)
		use_model();
	#endif

    return 0;
}