#include <xorai/model.h>
#include <algorithm>
#include <iomanip>

#define matrix_t Matrix<Float>

template<typename Float>
ModelViewer<Float>::ModelViewer(std::string filename, i8 float_precision)
    : filename(std::move(filename)), float_precision(float_precision)
{
    this->filestream = create_file_stream(false);
    this->writer = ModelViewer<Float>::create_stream_writer();
}

template<typename Float>
ModelViewer<Float>::~ModelViewer()
{
    this->filestream.close();
    delete(this->writer);
}

template<typename Float>
Model<Float>* ModelViewer<Float>::load()
{
    Json::CharReaderBuilder builder = ModelViewer<Float>::create_reader_builder();
    JSONCPP_STRING errors;

    if(!this->filestream)
    {
        std::cout << "[C++ ModelViewer]: Failed to load model file: `" << this->filename << "`\n";
        std::cout << "[!] File does not exist." << std::endl;
        exit(EXIT_FAILURE);
    }
    else if(!Json::parseFromStream(builder, this->filestream, &this->root, &errors))
    {
        std::cout << "[C++ ModelViewer]: Failed to parse model file: `" << this->filename << "`\n";
        std::cout << errors << std::endl;
        exit(EXIT_FAILURE);
    }
    else if(!check_root_members())
    {
        std::cout << "[C++ ModelViewer]: Cannot load a malformed model file: `" << this->filename << "`" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto model = new Model<Float>;

    model->layers  = parse<U64Array>(this->root["l"]);
    model->data    = parse<MatrixArray_t>(this->root["d"]);
    model->biases  = parse<MatrixArray_t>(this->root["b"]);
    model->weights = parse<MatrixArray_t>(this->root["w"]);

    return model;
}

template<typename Float>
matrix_t* ModelViewer<Float>::load(const Json::Value& matrix)
{
    const Json::Value& _rows = matrix["r"];
    const Json::Value& _cols = matrix["c"];
    const Json::Value& _data = matrix["d"];

    u64 rows = _rows.asUInt64();
    u64 cols = _cols.asUInt64();

    cvector<Float> data = cvector<Float>::with_capacity((u64)_data.size());

    std::transform(
        _data.begin(),
        _data.end(),
        std::back_inserter(data),
        []BASIC_UNARY(number, ModelViewer<Float>::string_to_float(number.asString()))
    );

    return new matrix_t(rows, cols, data);
}

template<typename Float>
Json::Value ModelViewer<Float>::jsonify(const MatrixArray_t& matrixArray) const
{
    Json::Value output(Json::arrayValue);

    for(matrix_t* matrix : matrixArray)
        output.append(jsonify(matrix));

    return output;
}

template<typename Float>
Json::Value ModelViewer<Float>::jsonify(const U64Array& u64Array)
{
    Json::Value output(Json::arrayValue);

    for(const u64& number : u64Array)
        output.append(Json::UInt64(number));

    return output;
}

template<typename Float>
Json::Value ModelViewer<Float>::jsonify(matrix_t* matrix) const
{
    Json::Value object(Json::objectValue);
    Json::Value data(Json::arrayValue);

    for(const Float& v : matrix->data)
        data.append(jsonify(v));

    object["r"] = Json::UInt64(matrix->rows);
    object["c"] = Json::UInt64(matrix->cols);
    object["d"] = data;

    return object;
}

template<typename Float>
std::string ModelViewer<Float>::jsonify(Float number) const {
#ifdef __F128_SUPPORT__
    if constexpr (std::is_same_v<Float, f128>)
    {
        std::string output = ModelViewer<Float>::f128_to_string(number, this->float_precision);
        return !output.empty() ? output : ModelViewer<Float>::f64_to_string((f64)number);
    }
    else
    {
#endif
    std::stringstream s;
    (s << std::fixed << std::setprecision((int)this->float_precision) << number);
    return s.str();
#ifdef __F128_SUPPORT__
    }
#endif
}

template<typename Float>
template<typename T>
T ModelViewer<Float>::parse(const Json::Value& jsonValue)
{
    using ModifierReturnType = std::conditional_t<std::is_same_v<T, MatrixArray_t>, matrix_t*, Json::UInt64>;
    std::function<ModifierReturnType(const Json::Value&)> modifier;

    if constexpr(std::is_same_v<T, U64Array>)
        modifier = []BASIC_UNARY(json_value, json_value.asUInt64());
    else if constexpr(std::is_same_v<T, MatrixArray_t>)
        modifier = []BASIC_UNARY(json_value, ModelViewer<Float>::load(json_value));
    else
    {
        std::cout << "[C++ ModelViewer]: Unable to parse `Json::Value` with the given template type.\n";
        std::cout << "\tSupported Types: [`U64Array`, `MatrixArray`];" << std::endl;
        exit(EXIT_FAILURE);
    }

    T output = T::with_capacity(jsonValue.size());
    std::transform(jsonValue.begin(), jsonValue.end(), std::back_inserter(output), modifier);
    return output;
}

template<typename Float>
void ModelViewer<Float>::write(const Json::Value& json)
{
    if(!this->filestream)
        this->filestream = create_file_stream();

    writer->write(json, &this->filestream);
}

template<typename Float>
bool ModelViewer<Float>::check_root_members()
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

template<typename Float>
std::fstream ModelViewer<Float>::create_file_stream(bool truncate)
{
    std::fstream filestream(this->filename, std::ios::in | std::ios::out);

    if(!filestream && truncate)
        return std::fstream(this->filename, std::ios::in | std::ios::out | std::ios::trunc);        

    return filestream;
}

#ifdef __F128_SUPPORT__
extern "C" {
    #include <quadmath.h>
}

template<typename Float>
std::string ModelViewer<Float>::f64_to_string(f64 number)
{
    std::stringstream s;
    (s << std::fixed << std::setprecision(UseMaxPrecision(64)) << number);
    return s.str();
}

template<typename Float>
std::string ModelViewer<Float>::f128_to_string(f128 number, i8 precision)
{
    std::stringstream p;
    p << "%." << (int)precision << "Qg";

    char buffer[128];
    int result = quadmath_snprintf(buffer, sizeof(buffer), p.str().c_str(), number);

    return result >= 0 ? std::move(std::string(buffer)) : "";
}

template<typename Float>
f128 ModelViewer<Float>::string_to_f128(const std::string& floatString)
{
    return strtoflt128(floatString.c_str(), nullptr);
}
#endif

template<typename Float>
Float ModelViewer<Float>::string_to_float(const std::string& number)
{
    if constexpr (std::is_same_v<Float, f32>)
        return std::stof(number);
    else if constexpr (std::is_same_v<Float, f64>)
        return std::stod(number);
    else if constexpr (std::is_same_v<Float, f128>)
#ifdef __F128_SUPPORT__
        return ModelViewer<Float>::string_to_f128(number);
#else
        return std::stold(number);
#endif

    std::cout << "[C++ ModelViewer]: Failed to convert float to " << typeid(Float).name() << std::endl;
    exit(EXIT_FAILURE);
}

template<typename Float>
Json::StreamWriter* ModelViewer<Float>::create_stream_writer()
{
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "";
    return builder.newStreamWriter();
}

template<typename Float>
Json::CharReaderBuilder ModelViewer<Float>::create_reader_builder()
{
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    return builder;
}

INSTANTIATE_CLASS_FLOATS(ModelViewer)