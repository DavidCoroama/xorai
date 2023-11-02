#include <xorai/model.h>
#include <algorithm>
#include <iomanip>

ModelViewer::ModelViewer(std::string filename, i64 float_precision)
    : filename(std::move(filename)), float_precision(float_precision)
{
    this->filestream.open(this->filename, std::ios::in | std::ios::out);
    this->writer = ModelViewer::create_stream_writer();
}

ModelViewer::~ModelViewer()
{
    this->filestream.close();
    delete(this->writer);
}

Model* ModelViewer::load()
{
    Json::CharReaderBuilder builder = ModelViewer::create_reader_builder();
    JSONCPP_STRING errors;

    if(!Json::parseFromStream(builder, this->filestream, &this->root, &errors))
    {
        std::cout << "[C++ ModelViewer]: Failed to load model file: `" << this->filename << '`' << std::endl;
        std::cout << errors << std::endl;
        exit(EXIT_FAILURE);
    }
    else if(!check_root_members())
    {
        std::cout << "[C++ ModelViewer]: Cannot load a malformed model file: `" << this->filename << "`" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto model = new Model;

    model->layers  = parse<U64Array>(this->root["l"]);
    model->data    = parse<MatrixArray>(this->root["d"]);
    model->biases  = parse<MatrixArray>(this->root["b"]);
    model->weights = parse<MatrixArray>(this->root["w"]);

    return model;
}

Matrix* ModelViewer::load(const Json::Value& matrix)
{
    const Json::Value& _rows = matrix["r"];
    const Json::Value& _cols = matrix["c"];
    const Json::Value& _data = matrix["d"];

    u64 rows = _rows.asUInt64();
    u64 cols = _cols.asUInt64();

    F64Array data = F64Array::with_capacity((u64)_data.size());

    std::transform(
        _data.begin(),
        _data.end(),
        std::back_inserter(data),
        []BASIC_UNARY(n, std::stold(n.asString()))
    );

    return new Matrix(rows, cols, data);
}

Json::Value ModelViewer::jsonify(const MatrixArray& matrixArray) const
{
    Json::Value output(Json::arrayValue);

    for(Matrix* m : matrixArray)
        output.append(jsonify(m));

    return output;
}

Json::Value ModelViewer::jsonify(const U64Array& u64Array)
{
    Json::Value output(Json::arrayValue);

    for(const u64& n : u64Array)
        output.append(Json::UInt64(n));

    return output;
}

Json::Value ModelViewer::jsonify(Matrix* matrix) const
{
    Json::Value object(Json::objectValue);
    Json::Value data(Json::arrayValue);

    for(const f64& v : matrix->data)
        data.append(jsonify(v));

    object["r"] = Json::UInt64(matrix->rows);
    object["c"] = Json::UInt64(matrix->cols);
    object["d"] = data;

    return object;
}

std::string ModelViewer::jsonify(f64 n) const
{
    std::stringstream s;
    s << std::fixed << std::setprecision((int)this->float_precision) << n;
    return s.str();
}

template<typename T>
T ModelViewer::parse(const Json::Value& j)
{
    using ModifierReturnType = std::conditional_t<std::is_same_v<T, MatrixArray>, Matrix*, Json::UInt64>;
    std::function<ModifierReturnType(const Json::Value&)> modifier;

    if constexpr(std::is_same_v<T, U64Array>)
        modifier = []BASIC_UNARY(n, n.asUInt64());
    else if constexpr(std::is_same_v<T, MatrixArray>)
        modifier = []BASIC_UNARY(m, ModelViewer::load(m));
    else
    {
        std::cout << "[C++ ModelViewer]: Unable to parse `Json::Value` with the given template type.\n";
        std::cout << "\tSupported Types: [`U64Array`, `MatrixArray`];" << std::endl;
        exit(EXIT_FAILURE);
    }

    T output = T::with_capacity(j.size());
    std::transform(j.begin(), j.end(), std::back_inserter(output), modifier);
    return output;
}

void ModelViewer::write(const Json::Value& json)
{
    writer->write(json, &this->filestream);
}

bool ModelViewer::check_root_members()
{
    bool flags[4] = {false, false, false, false};

    for(const auto& id : this->root.getMemberNames()) {
        switch(id[0])
        {
            case 'l': flags[0] = true; break;
            case 'd': flags[1] = true; break;
            case 'b': flags[2] = true; break;
            case 'w': flags[3] = true; break;
            default: return false;
        }
    }

    return flags[0] && flags[1] && flags[2] && flags[3];
}

Json::StreamWriter* ModelViewer::create_stream_writer()
{
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "";
    return builder.newStreamWriter();
}

Json::CharReaderBuilder ModelViewer::create_reader_builder()
{
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    return builder;
}