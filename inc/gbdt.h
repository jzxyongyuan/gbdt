/**
* @file gbdt.h
* @author jiangzhengxiang
* @date 2016/05/23 19:54:18
* @brief 
*  
**/

#include "gbdt_include.h"
#include "gbdt_data.h"
#include "regression_tree.h"
#include "common_func.h"

#ifndef INC_GBDT_H
#define INC_GBDT_H
namespace gbdt
{

class GBDT;

typedef struct _GBDT_PREDICT_T_ {
    int        thread_id;
    int       thread_num;
    int     max_tree_num;
    int   max_sample_num;
    GBDT*           gbdt;
} GBDT_PREDICT_T; 

class GBDT
{

public:
    GBDT() {
        _m_trees        = NULL;
        _m_multi_data   = NULL;
        _m_multi_thread = NULL;
    }
    
    ~GBDT() {
        if (_m_multi_data != NULL) {
            delete[] _m_multi_data;
        }
        if (_m_trees != NULL) {
            delete[] _m_trees;
        }
        if (_m_multi_thread != NULL) {
            delete[] _m_multi_thread;
        }
    }
    
    int init(const char* conf_path, const char* conf_file);

    int test(const char* test_file);

    int train(const char* train_file);

    int save(const char* model_file);

    int load(const char* model_file);

    GBDTValue get_shrinkage() {
        return _m_shrinkage;
    }

    RegressionTree* get_tree(int idx) {
        if (idx >= _m_iterations) {
            return NULL;
        }
        return &_m_trees[idx];
    }

    int update_target(int idx, GBDTValue value);
    
    int predict(int idx, int max_tree_num, GBDTValue &p);

private:
    int               _m_thread_num;
    int               _m_iterations;
    int               _m_feature_size;
    bool              _m_ignore_weight;
    bool              _m_load_initial_guess;
    double            _m_sample_ratio;
    GBDTData          _m_test_data;
    GBDTData          _m_train_data;
    _GBDT_PREDICT_T_* _m_multi_data;
    pthread_t*        _m_multi_thread;

    int               _m_loss_type;
    GBDTValue         _m_shrinkage;
    GBDTValue         _m_bias;
    RegressionTree*   _m_trees;

    int init_fit();

    int fit();

    int predict(GBDT_TUPLE_T* tuple, int max_tree_num, GBDTValue &p);

    friend void* multi_predict(void *arg);
};

}

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

