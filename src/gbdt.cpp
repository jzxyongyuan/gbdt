/**
* @file gbdt.cpp
* @author jiangzhengxiang
* @date 2016/05/23 19:56:56
* @brief 
*  
**/

#include "gbdt.h"

namespace gbdt
{
int GBDT::init(const char* path, const char* file) {
    if (path == NULL || file == NULL || path[0] == 0 || file[0] == 0) {
        return 1;
    }
    int ret = 0;
    char conf_file[GBDT_MAX_PATH_LEN] = {0};
    std::string str_tmp;

    snprintf(conf_file, GBDT_MAX_PATH_LEN, "%s/%s", path, file);
    conf_file[GBDT_MAX_PATH_LEN - 1] = '\0';

    cfgiml::Configure config;
    ret = config.load(conf_file);
    if (ret != 0) {
        LOG(WARNING) << "load configure file " << conf_file << " failed";
        return 1;
    }

    try {
        _m_thread_num = config["THREAD_NUM"].to_int32();
    } catch (...) {
        LOG(WARNING) << "fail to read THREAD_NUM, use default value " << GBDT_DEFAULT_THREAD_NUM;
        _m_thread_num = GBDT_DEFAULT_THREAD_NUM;
    }
    try {
        _m_iterations = config["ITERATION"].to_int32();
    } catch (...) {
        LOG(WARNING) << "fail to read ITERATION, use default value " << GBDT_DEFAULT_ITERATIONS;
        _m_iterations = GBDT_DEFAULT_ITERATIONS;
    }
    try {
        _m_feature_size = config["FEATURE_SIZE"].to_int32();
    } catch (...) {
        LOG(WARNING) << "fail to read FEATURE_SIZE";
        return 1;
    }
    try {
        _m_shrinkage = config["SHRINKAGE"].to_double();
    } catch (...) {
        LOG(WARNING) << "fail to read SHRINKAGE, use default value " << GBDT_DEFAULT_SHRINKAGE;
        _m_shrinkage = GBDT_DEFAULT_SHRINKAGE;
    }
    try {
        _m_sample_ratio = config["SAMPLE_RATIO"].to_double();
    } catch (...) {
        LOG(WARNING) << "fail to read SAMPLE_RATIO, use default value " 
                << GBDT_DEFAULT_SAMPLE_RATIO;
        _m_sample_ratio = GBDT_DEFAULT_SAMPLE_RATIO;
    }
    try {
        str_tmp = config["LOSS_TYPE"].to_string();
        if (str_tmp == "LOG_LIKELIHOOD") {
            _m_loss_type = GBDT_LOG_LIKELIHOOD;
        } else if (str_tmp == "SQUARED_ERROR") {
            _m_loss_type = GBDT_SQUARED_ERROR;
        } else {
            LOG(WARNING) << "wrong value of LOSS_TYPE " << str_tmp;
            return 1;
        }
    } catch (...) {
        LOG(WARNING) << "fail to read LOSS_TYPE, use default value GBDT_SQUARED_ERROR";
        _m_loss_type = GBDT_SQUARED_ERROR;
    }
    try {
        str_tmp = config["INITIAL_GUESS"].to_string();
        _m_load_initial_guess = str_tmp == "TRUE" ? true : false;
    } catch (...) {
        LOG(WARNING) << "fail to read INITIAL_GUESS, use default value false";
        _m_load_initial_guess = GBDT_DEFAULT_INIT_GUESS;
    }
    try {
        str_tmp = config["IGNORE_WEIGHT"].to_string();
        _m_ignore_weight = str_tmp == "TRUE" ? true : false;
    } catch (...) {
        LOG(WARNING) << "fail to read IGNORE_WEIGHT, use default value false";
        _m_ignore_weight = GBDT_DEFAULT_IGNORE_WEIGHT;
    }
   
    return 0;
}

int GBDT::init_fit() {
    int len  = _m_train_data.size();
    double s = 0;
    double c = 0;
    for (int idx = 0; idx < len; idx++) {
        s += _m_train_data[idx]->label * _m_train_data[idx]->weight;
        c += _m_train_data[idx]->weight;
    }

    double v = s / c;

    if (_m_loss_type == GBDT_SQUARED_ERROR) {
        _m_bias = static_cast<GBDTValue>(v);
    } else if (_m_loss_type == GBDT_LOG_LIKELIHOOD) {
        _m_bias = static_cast<GBDTValue>(std::log((1 + v) / (1 - v)) / 2.0);
    }
    DLOG(INFO) << "v " << v << " s " << s << " c " << c << " bias " << _m_bias << " loss " << _m_loss_type;

    _m_trees = new RegressionTree[_m_iterations];

    for (int i = 0; i < _m_iterations; i++) {
        _m_trees[i].init(_m_feature_size, _m_loss_type);
    }
    _m_multi_thread = new pthread_t[_m_thread_num];
    _m_multi_data   = new GBDT_PREDICT_T[_m_thread_num];
    for (int i = 0; i < _m_thread_num; i++) {
        _m_multi_data[i].gbdt            = this;
        _m_multi_data[i].thread_id       = i;
        _m_multi_data[i].thread_num      = _m_thread_num;
        _m_multi_data[i].max_sample_num  = _m_sample_ratio * _m_train_data.size();
    }

    return 0;
}

void* multi_predict(void* arg) {
    GBDT_PREDICT_T* data = (GBDT_PREDICT_T*)arg;
    int        thread_id = data -> thread_id;
    int       thread_num = data -> thread_num;
    int      sample_size = data -> max_sample_num;
    int     max_tree_num = data -> max_tree_num;
    GBDT*           gbdt = data -> gbdt;
    GBDTValue      value = 0;
    
    for (int i = 0; i < sample_size; i++) {
        if ((i - thread_id) % thread_num != 0) {
            continue;
        }
        gbdt -> predict(i, max_tree_num, value);
        
        gbdt -> update_target(i, value);
    }
    return NULL;
}

int GBDT::update_target(int idx, GBDTValue value) {
    if (idx < 0 || idx >= _m_train_data.size()) {
        return 1;
    }
    if (_m_loss_type == GBDT_SQUARED_ERROR) {
        _m_train_data[idx] -> target = _m_train_data[idx]->label - value;
    } else if (_m_loss_type == GBDT_LOG_LIKELIHOOD) {
        _m_train_data[idx] -> target = 
                logit_loss_gradient(_m_train_data[idx]->label, value);
    }
    DLOG(INFO) << "idx: " << idx << " label " << _m_train_data[idx]->label 
            << " target: " << _m_train_data[idx]->target << " p " << value;
    return 0;
}

int GBDT::predict(int idx, int max_tree_num, GBDTValue& p) {
    return predict(_m_train_data[idx], max_tree_num, p);
}

int GBDT::predict(GBDT_TUPLE_T* tuple, int max_tree_num, GBDTValue &p) {
    p = _m_bias;
    if (_m_load_initial_guess) {
        p = tuple -> initial_guess;
    }
    DLOG(INFO) << "bias: " << p;
    for (int i = 0; i < max_tree_num; i++) {
        GBDTValue tmp = 0;
        _m_trees[i].predict(tuple, tmp);
        DLOG(INFO) << "i: " << i << " " << tmp;
        p += _m_shrinkage * tmp;
    }
    return 0;
}

int GBDT::fit() {
    int samples = _m_train_data.size();
    if (_m_sample_ratio < 1) {
        samples = samples * _m_sample_ratio;
    }

    init_fit();

    for (int iter = 0; iter < _m_iterations; iter++) {
        LOG_EVERY_N(INFO, 10) << "iteration: " << iter;

        if (_m_sample_ratio < 1) {
            _m_train_data.random();
        }
        if (_m_thread_num > 1 && iter > 100) {
            for (int i = 0; i < _m_thread_num; i++) {
                _m_multi_data[i].max_tree_num = iter;
                pthread_create(&_m_multi_thread[i], NULL, multi_predict, (void*)(&_m_multi_data[i]));
            }
            for (int i = 0; i < _m_thread_num; i++) {
                pthread_join(_m_multi_thread[i], NULL);
            }
        } else {
            for (int s_idx = 0; s_idx < samples; s_idx++) {
                GBDTValue p;
                predict(_m_train_data[s_idx], iter, p);
                
                if (_m_loss_type == GBDT_SQUARED_ERROR) {
                    _m_train_data[s_idx]->target = _m_train_data[s_idx]->label - p;
                } else if (_m_loss_type == GBDT_LOG_LIKELIHOOD) {
                    _m_train_data[s_idx]->target = 
                            logit_loss_gradient(_m_train_data[s_idx]->label, p);
                }
                DLOG(INFO) << "idx: " << s_idx << " label " << _m_train_data[s_idx]->label 
                        << " target: " << _m_train_data[s_idx]->target << " p " << p;
            }
        }
        _m_trees[iter].fit(&_m_train_data, samples);
    }
    // Calculate gain
    GBDTValue gain[_m_feature_size];

    for (int i = 0; i < _m_feature_size; i++) {
        gain[i] = 0.0;
    }

    for (int i = 0; i < _m_iterations; i++) {
        GBDTValue *g;
        _m_trees[i].get_gain(g);
        for (int j = 0; j < _m_feature_size; j++) {
            gain[j] += g[j];
        }
    }

    for (int i = 0; i < _m_feature_size; i++) {
        DLOG(INFO) << "gain feat[" << i << "]: " << gain[i];
    }
    return 0;
}

int GBDT::test(const char* test_file) {
    int ret         = 0;
    int samples     = 0;
    int correct_cnt = 0;
    ret = _m_test_data.init(_m_load_initial_guess, _m_ignore_weight, _m_feature_size);
    if (ret != 0) {
        return 1;
    }

    ret = _m_test_data.load_file(test_file);
    if (ret != 0) {
        return 1;
    }
   
    samples = _m_test_data.size();
    for (int s_idx = 0; s_idx < samples; s_idx++) {
        GBDTValue p;
        predict(_m_test_data[s_idx], _m_iterations, p);
        GBDTValue tmp = abs(_m_test_data[s_idx]->label - p);
        LOG(INFO) << "idx: " << s_idx << " predict " << p << " label " << 
                _m_test_data[s_idx]->label << " " << (tmp < 0.5 ? "TRUE" : "FALSE");
        correct_cnt += tmp < 0.5 ? 1 : 0;
    }
    LOG(INFO) << "correct rate: " << correct_cnt * 100.0 / samples;
    
    return 0;
}

int GBDT::train(const char* train_file) {
    int ret = 0;
    
    ret = _m_train_data.init(_m_load_initial_guess, _m_ignore_weight, _m_feature_size);
    if (ret != 0) {
        return 1;
    }

    ret = _m_train_data.load_file(train_file);
    if (ret != 0) {
        return 1;
    }

    ret = fit();
    if (ret != 0) {
        return 1;
    }
    return 0;
}

int GBDT::save(const char* model_file) {
    FILE* fp = fopen(model_file, "wb");
    if (fp == NULL) {
        return 1;
    }
    fwrite(&_m_bias,       sizeof(GBDTValue), 1, fp);
    fwrite(&_m_shrinkage,  sizeof(GBDTValue), 1, fp);
    fwrite(&_m_iterations, sizeof(int),       1, fp);
    fwrite(&_m_loss_type,  sizeof(int),       1, fp);
    for (int i = 0; i < _m_iterations; i++) {
        _m_trees[i].save(fp);
    }
    fclose(fp);
    DLOG(INFO) << "bias: " << _m_bias << " shrinkage: " << _m_shrinkage << 
            " tree_cnt: " << _m_iterations << " loss: " << _m_loss_type;
    LOG(INFO) << "save model success";
    return 0;
}

int GBDT::load(const char* model_file) {
    FILE* fp = fopen(model_file, "rb");
    if (fp == NULL) {
        return 1;
    }
    fread(&_m_bias,       sizeof(GBDTValue), 1, fp);
    fread(&_m_shrinkage,  sizeof(GBDTValue), 1, fp);
    fread(&_m_iterations, sizeof(int),       1, fp);
    fread(&_m_loss_type,  sizeof(int),       1, fp);
    DLOG(INFO) << "bias: " << _m_bias << " shrinkage: " << _m_shrinkage << 
            " tree_cnt: " << _m_iterations << " loss: " << _m_loss_type;
    
    _m_trees = new RegressionTree[_m_iterations];
    for (int i = 0; i < _m_iterations; i++) {
        _m_trees[i].init(_m_feature_size, _m_loss_type);
        _m_trees[i].load(fp);
    }
    LOG(INFO) << "load model success";
    fclose(fp);
    return 0;
}
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

