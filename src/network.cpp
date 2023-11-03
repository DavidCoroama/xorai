#include <xorai/activation.h>
#include <xorai/network.h>
#include <cassert>

Network::Network(const U64Array& layers, f64 learning_rate)
{
    this->data = {};
    this->biases = {};
    this->weights = {};

    for(u64 i = 0; i < layers.size() - 1; i++)
    {
        this->weights.push_back(Matrix::random(layers[i + 1], layers[i]));
        this->biases.push_back(Matrix::random(layers[i + 1], 1));
    }

    this->layers = layers;
    this->learning_rate = learning_rate;
}

Network::Network(std::string filename, f64 learning_rate)
{

    ModelViewer viewer(std::move(filename));
    Model* model = viewer.load();

    this->data = model->data;
    this->biases = model->biases;
    this->weights = model->weights;

    this->layers = model->layers;
    this->learning_rate = learning_rate;

    delete(model);
}

Network::~Network()
{
    auto d = BASIC_UNARY_DELETE;

    this->data.map(d);
    this->biases.map(d);
    this->weights.map(d);
}

Matrix* Network::feed_forward(Matrix* inputs)
{
    assert(this->layers[0] == inputs->data.size());

    this->data.map_if(!this->data.empty(), BASIC_UNARY_DELETE);

    Matrix* current = inputs;
    this->data = {current->clone()};

    for(u64 i = 0; i < this->layers.size() - 1; i++)
    {
        current = this->weights[i]->clone()
                ->dot(current)
                ->add(this->biases[i])
                ->map(sigmoid);

        this->data.push_back(current);
    }

    return current;
}

void Network::back_propagate(Matrix* inputs, Matrix* targets)
{
    Matrix* errors = targets->clone()->sub(inputs);
    Matrix* gradients = inputs->clone()->map(derivative);

    auto scale = [this]BASIC_UNARY(x, x * this->learning_rate);

    for(u64 i = this->layers.size() - 1; i--;)
    {
        gradients->mul(errors)->map(scale);

        this->weights[i]->add(gradients->clone(true)->dot(this->data[i]->clone(true)->transpose()));
        this->biases[i]->add(gradients->ref());

        errors = this->weights[i]->clone()->transpose()->dot(errors->ref());
        gradients = this->data[i]->clone()->map(derivative);
    }

    delete(errors);
    delete(gradients);
}

void Network::train(Dataset& inputs, Dataset& targets, u64 epochs)
{
    Matrix *input, *target, *output;
    u64 i, j;

    for(i = 1; i < epochs + 1; i++)
    {
#if defined(DEBUG) && !defined(NO_DEBUG)
        if((epochs < 100) || (i % (epochs / 100) == 0))
            std::cout << "Epoch " << i << " of " << epochs << "\n";
#endif
        for(j = 0; j < inputs.size(); j++)
        {
            input = Matrix::from(inputs[j].clone());
            target = Matrix::from(targets[j].clone());
            output = feed_forward(input);

            back_propagate(output, target);

            delete(input);
            delete(target);
        }
    }
}

Matrix* Network::test(f64 a, f64 b)
{
    /* Save the network's current data state */
    MatrixArray netdata = this->data.clone();
    this->data = {};

    Matrix* input = Matrix::from({a, b});
    Matrix* output = feed_forward(input)->clone();

    this->data.map_if(!this->data.empty(), BASIC_UNARY_DELETE);

    /* Revert back to the networks old data state */
    this->data = netdata;

    delete(input);
    return output;
}

void Network::save(std::string filename, i64 float_precision) const
{
    ModelViewer viewer(std::move(filename), float_precision);
    Json::Value model;

    model["d"] = viewer.jsonify(this->data);
    model["b"] = viewer.jsonify(this->biases);
    model["w"] = viewer.jsonify(this->weights);
    model["l"] = viewer.jsonify(this->layers);

    viewer.write(model);
}