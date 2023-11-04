#pragma once
#ifndef XORAI_MODEL_H
#define XORAI_MODEL_H

#include <jsoncpp/json/writer.h>
#include <jsoncpp/json/reader.h>
#include <xorai/matrix.h>
#include <fstream>

template<typename Float>
struct Model {
    U64Array layers;
    MatrixArray<Float> data;
    MatrixArray<Float> biases;
    MatrixArray<Float> weights;
};

template<typename Float>
class ModelViewer
{
private:
    using matrix_t = Matrix<Float>;
    using MatrixArray_t = MatrixArray<Float>;

public:
    explicit ModelViewer(std::string, i8 = 8);
    ~ModelViewer();

    Model<Float>* load();
    static matrix_t* load(const Json::Value&);

    static Json::Value jsonify(const U64Array&);
    Json::Value jsonify(const MatrixArray_t&) const;
    Json::Value jsonify(matrix_t*) const;
    std::string jsonify(Float) const;

    template<typename T>
    static T parse(const Json::Value&);
    void write(const Json::Value&);

    const std::string filename;
    const i8 float_precision;

private:
    bool check_root_members();
    std::fstream create_file_stream(bool = true);

#ifdef __F128_SUPPORT__
    static std::string f64_to_string(f64);
    static std::string f128_to_string(f128, i8);
    static f128 string_to_f128(const std::string&);
#endif

    static Float string_to_float(const std::string&);
    static Json::StreamWriter* create_stream_writer();
    static Json::CharReaderBuilder create_reader_builder();

    Json::StreamWriter* writer;
    std::fstream filestream;
    Json::Value root;
};

#endif //XORAI_MODEL_H