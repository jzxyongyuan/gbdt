/**
* @file gbdt_data.h
* @author jiangzhengxiang
* @date 2016/05/26 12:59:57
* @brief 
*  
**/
#ifndef INC_GBDT_DATA_T
#define INC_GBDT_DATA_T

#include <boost/lexical_cast.hpp>
#include "gbdt_include.h"
#include "common_func.h"

namespace gbdt
{
typedef std::vector<GBDT_TUPLE_T *> GBDTDataVector;

class GBDTData
{
public:
    GBDTData() {}

    ~GBDTData() {}

    int init(bool load_initial_guess, bool ignore_weight, int feature_size) {
        _m_load_initial_guess = load_initial_guess;
        _m_ignore_weight      = ignore_weight;
        _m_feature_size       = feature_size;
        return 0;
    }

    bool empty() {
        return _m_data.empty();
    }

    int sort(int index) {
        std::sort(_m_data.begin(), _m_data.end(), TupleCompare(index));
        return 0;
    }

    int clear() {
        _m_data.clear();
        return 0;
    }

    int random() {
        std::random_shuffle(_m_data.begin(), _m_data.end());
        return 0;
    }

    int size() {
        return (int)_m_data.size();
    }

    int push_back(GBDT_TUPLE_T* tuple) {
        _m_data.push_back(tuple);
        return 0;
    }

    GBDT_TUPLE_T* operator[] (const int i) {
        return _m_data[i];
    }

    int parse_feature(const char* feat_str, int& idx, GBDTValue& feature);

    int split_line(const char* line, std::vector<std::string> &tokens);

    int load_line(const char* line);

    int load_file(const char* path);

private:
    int  _m_feature_size;
    bool _m_ignore_weight;
    bool _m_load_initial_guess;
    std::vector<GBDT_TUPLE_T* > _m_data;
};

}
#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

