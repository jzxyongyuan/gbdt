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

class GBDT
{

public:
    GBDT() {}
    
    ~GBDT() {}
    
    int init(const char* conf_path, const char* conf_file);

    int test(const char* test_file);

    int train(const char* train_file);

    int save(const char* model_file);

    int load(const char* model_file);
private:
    int             _m_iterations;
    int             _m_feature_size;
    bool            _m_ignore_weight;
    bool            _m_load_initial_guess;
    double          _m_sample_ratio;
    GBDTData        _m_test_data;
    GBDTData        _m_train_data;

    int             _m_loss_type;
    GBDTValue       _m_shrinkage;
    GBDTValue       _m_bias;
    RegressionTree* _m_trees;

    int init_fit();

    int fit();

    int predict(GBDT_TUPLE_T* tuple, int max_tree_num, GBDTValue &p);
};

}

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

