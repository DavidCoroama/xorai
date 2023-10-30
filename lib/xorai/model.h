#pragma once
#ifndef XORAI_MODEL_H
#define XORAI_MODEL_H

#include <jsoncpp/json/writer.h>
#include <jsoncpp/json/reader.h>
#include <xorai/matrix.h>
#include <fstream>
#include <string>

typedef struct __Model {
    U64Array layers;
    MatrixArray data;
    MatrixArray biases;
    MatrixArray weights;
} Model;

class ModelViewer
{
public:
    explicit ModelViewer(std::string , i64 = 7);
    ~ModelViewer();

    Model* load();
    static Matrix* load(const Json::Value&);

    static Json::Value jsonify(const U64Array&);
    Json::Value jsonify(const MatrixArray&) const;
    Json::Value jsonify(Matrix*) const;
    std::string jsonify(f64) const;

    template<typename T>
    static T parse(const Json::Value&);
    void write(const Json::Value&);

    const std::string filename;
    const i64 float_precision;

private:
    bool check();
    static Json::StreamWriter* create_stream_writer();
    static Json::CharReaderBuilder create_reader_builder();

    Json::StreamWriter* writer;
    std::fstream filestream;
    Json::Value root;
};

#endif //XORAI_MODEL_H