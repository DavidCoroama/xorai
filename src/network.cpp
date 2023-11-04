#include <xorai/activation.h>
#include <xorai/network.h>
#include <cassert>

#define matrix_t Matrix<Float>

template<typename Float>
Network<Float>::Network(const U64Array& layers, Float learning_rate)
{
    assert_float_type();

    this->data = {};
    this->biases = {};
    this->weights = {};

    for(u64 i = 0; i < layers.size() - 1; i++)
    {
        this->weights.push_back(matrix_t::random(layers[i + 1], layers[i]));
        this->biases.push_back(matrix_t::random(layers[i + 1], 1));
    }

    this->layers = layers;
    this->learning_rate = learning_rate;
}

template<typename Float>
Network<Float>::Network(std::string filename, Float learning_rate)
{
    assert_float_type();

    ModelViewer<Float> viewer(std::move(filename));
    Model<Float>* model = viewer.load();

    this->data = model->data;
    this->biases = model->biases;
    this->weights = model->weights;

    this->layers = model->layers;
    this->learning_rate = learning_rate;

    delete(model);
}

template<typename Float>
Network<Float>::~Network()
{
    auto d = BASIC_UNARY_DELETE;

    this->data.map(d);
    this->biases.map(d);
    this->weights.map(d);
}

template<typename Float>
matrix_t* Network<Float>::feed_forward(matrix_t* inputs)
{
    assert(this->layers[0] == inputs->data.size());

    this->data.map_if(!this->data.empty(), BASIC_UNARY_DELETE);

    matrix_t* current = inputs;
    this->data = {current->clone()};

    for(u64 i = 0; i < this->layers.size() - 1; i++)
    {
        current = this->weights[i]->clone()
                ->dot(current)
                ->add(this->biases[i])
                ->map(sigmoid<Float>);

        this->data.push_back(current);
    }

    return current;
}

template<typename Float>
void Network<Float>::back_propagate(matrix_t* inputs, matrix_t* targets)
{
    matrix_t* errors = targets->clone()->sub(inputs);
    matrix_t* gradients = inputs->clone()->map(derivative<Float>);

    auto scale = [this]BASIC_UNARY(x, x * this->learning_rate);

    for(u64 i = this->layers.size() - 1; i--;)
    {
        gradients->mul(errors)->map(scale);

        this->weights[i]->add(gradients->clone(true)->dot(this->data[i]->clone(true)->transpose()));
        this->biases[i]->add(gradients->ref());

        errors = this->weights[i]->clone()->transpose()->dot(errors->ref());
        gradients = this->data[i]->clone()->map(derivative<Float>);
    }

    delete(errors);
    delete(gradients);
}

template<typename Float>
void Network<Float>::train(Dataset<Float>& inputs, Dataset<Float>& targets, u64 epochs)
{
    matrix_t *input, *target, *output;
    u64 i, j;

    for(i = 1; i < epochs + 1; i++)
    {
#if defined(DEBUG) && !defined(NO_DEBUG)
        if((epochs < 100) || (i % (epochs / 100) == 0))
            std::cout << "Epoch " << i << " of " << epochs << "\n";
#endif
        for(j = 0; j < inputs.size(); j++)
        {
            input = matrix_t::from(inputs[j].clone());
            target = matrix_t::from(targets[j].clone());
            output = feed_forward(input);

            back_propagate(output, target);

            delete(input);
            delete(target);
        }
    }
}

template<typename Float>
matrix_t* Network<Float>::test(Float a, Float b)
{
    /* Save the network's current data state */
    MatrixArray<Float> netdata = this->data.clone();
    this->data = {};

    matrix_t* input = matrix_t::from({a, b});
    matrix_t* output = feed_forward(input)->clone();

    this->data.map_if(!this->data.empty(), BASIC_UNARY_DELETE);

    /* Revert back to the networks old data state */
    this->data = netdata;

    delete(input);
    return output;
}

template<typename Float>
void Network<Float>::save(std::string filename, i8 float_precision) const
{
    ModelViewer<Float> viewer(std::move(filename), float_precision);
    Json::Value model;

    model["d"] = viewer.jsonify(this->data);
    model["b"] = viewer.jsonify(this->biases);
    model["w"] = viewer.jsonify(this->weights);
    model["l"] = viewer.jsonify(this->layers);

    viewer.write(model);
}

template<typename Float>
void Network<Float>::assert_float_type()
{
    if constexpr (!is_float_type<Float>)
    {
        std::cout << "[C++ Network]: Network<Float> requires Float to be a floating point type.\n";
        std::cout << "\tSupported float types include: [f32 (float), f64 (double), and f128 (long double || __float128)]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

INSTANTIATE_CLASS_FLOATS(Network)