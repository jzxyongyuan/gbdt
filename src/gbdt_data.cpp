/**
* @file gbdt_data.cpp
* @author jiangzhengxiang
* @date 2016/05/26 13:01:36
* @brief 
*  
**/

#include "gbdt_data.h"
namespace gbdt
{
int GBDTData::parse_feature(const char* feat_str, int& idx, GBDTValue& feature) {
    char* p = NULL;
    char tmp[GBDT_MAX_LINE_LEN] = {0};
    snprintf(tmp, GBDT_MAX_LINE_LEN, feat_str);
    p       = strtok(tmp, GBDT_FEAT_DELIM); 
    idx     = boost::lexical_cast<int>(p);
    p      += strlen(p) + 1;
    feature = boost::lexical_cast<GBDTValue>(p);
    return 0;
}

int GBDTData::split_line(const char* line, std::vector<std::string> &tokens) {
    if (line == NULL || line[0] == 0) {
        return 1;
    }
    char* p = NULL;
    char tmp[GBDT_MAX_LINE_LEN] = {0};
    snprintf(tmp, GBDT_MAX_LINE_LEN, line);
    p = strtok(tmp, GBDT_ITEM_DELIM);
    while (p) {
        tokens.push_back(p);
        p += strlen(p) + 1;
        p = strtok(p, GBDT_ITEM_DELIM);
    }
    return 0;
}

int GBDTData::load_line(const char* line) {
    if (line == NULL || line[0] == 0) {
        return 1;
    }
    int ret = 0;
    GBDT_TUPLE_T* result = new GBDT_TUPLE_T();
    result -> feature = new GBDTValue[_m_feature_size];
    for (int i = 0; i < _m_feature_size; i++) {
        result -> feature[i] = GBDT_UNKNOWN_VALUE;
    }

    std::vector<std::string> tokens;
    ret = split_line(line, tokens);
    if (ret != 0) {
        return 1;
    }
    if (_m_load_initial_guess && tokens.size() < 3) {
        return 1;
    } else if (!_m_load_initial_guess && tokens.size() < 2) {
        return 1;
    }
    result -> label  = boost::lexical_cast<GBDTValue>(tokens[0]);
    if (_m_ignore_weight) {
        result -> weight = 1;
    } else {
        result -> weight = boost::lexical_cast<GBDTValue>(tokens[1]);
    }
    for (int i = 2; i < (int)tokens.size(); i++) {
        int idx           = 0;
        GBDTValue feature = 0;
        parse_feature(tokens[i].c_str(), idx, feature);
        result -> feature[idx] = feature;
    }
    _m_data.push_back(result);
    return 0;
}

int GBDTData::load_file(const char* path) {
    std::ifstream stream(path);
    if (!stream) {
        LOG(WARNING) << "fail to open " << path;
        return 1;
    }
    _m_data.clear();

    char* local_buffer = new char[GBDT_MAX_BUFF_LEN];
    stream.rdbuf()->pubsetbuf(local_buffer, GBDT_MAX_BUFF_LEN);

    std::string line;
    while(std::getline(stream, line)) {
        load_line(line.c_str());
    }
    
    DLOG(INFO) << "load file finished, size: " << _m_data.size();
    delete[] local_buffer;

    return 0;
}

}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

