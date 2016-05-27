/**
* @file regression_tree.h
* @author jiangzhengxiang
* @date 2016/05/25 17:56:08
* @brief 
*  
**/

#ifndef _INC_REGRESSION_TREE_H_
#define _INC_REGRESSION_TREE_H_

#include "gbdt_include.h"
#include "gbdt_data.h"
#include "common_func.h"

namespace gbdt {
class RegressionNode {
public:
    int       index;
    int       feat_index;
    bool      leaf;
    GBDTValue value;
    GBDTValue predict;
    RegressionNode** child;
    
    RegressionNode(int idx);
    
    ~RegressionNode() {
        if (child != NULL) {
            delete child[0];
            delete child[1];
            delete child[2];
            delete[] child;
        }
    }

    int save(FILE* fp);

    int load(FILE* fp);

    int load(std::vector<RegressionNode* >* node_vec);
private:
};

class RegressionTree {
public:
    RegressionTree() {
        _m_node_cnt             = 0;
        _m_max_depth            = GBDT_MAX_DEPTH;
        _m_min_leaf_size        = GBDT_MIN_LEAF_SIZE;
        _m_feature_sample_ratio = GBDT_DEFAULT_FEAT_RATIO;
    }
    
    ~RegressionTree() {
        if (_m_feature_costs != NULL) {
            delete[] _m_feature_costs;
        }
        if (_m_gain != NULL) {
            delete[] _m_gain;
        }
        if (_m_root != NULL) {
            delete _m_root;
        }
    }

    int init(int feature_size, int loss_type) {
        _m_feature_size  = feature_size;
        _m_loss_type     = loss_type;
        _m_feature_costs = new GBDTValue[_m_feature_size];
        for (int i = 0; i < _m_feature_size; i++) {
            _m_feature_costs[i] = 1;
        }
        return 0;
    }
    
    int fit(GBDTData* train_data, int sample_size);

    int predict(GBDT_TUPLE_T* tuple, GBDTValue& res);

    int get_gain(GBDTValue*& gain) {
        gain = _m_gain;
        return 0;
    }

    int save(FILE* fp);

    int load(FILE* fp);

private:
    int             _m_node_cnt;
    int             _m_loss_type;
    int             _m_max_depth;
    int             _m_feature_size;
    int             _m_min_leaf_size;
    bool            _m_enable_feature_tunning;

    GBDTValue       _m_feature_sample_ratio;
    GBDTValue*      _m_feature_costs;
    GBDTValue*      _m_gain;
    RegressionNode* _m_root;

    bool same(GBDTData* train_data, int sample_size);
    
    GBDTValue average(GBDTData* train_data, int sample_size);
    
    GBDTValue logit_opt_value(GBDTData* train_data, int sample_size);

    int get_impurity(GBDTData* train_data, int sample_size, int index, GBDTValue& value, 
            GBDTValue& impurity, GBDTValue& gain);

    int find_split(GBDTData* train_data, int sample_size, int& index, GBDTValue& value, 
            GBDTValue& gain);
    
    int split_data(GBDTData* train_data, GBDTData* data_split, int sample_size, 
            int index, GBDTValue value); 
    
    int fit(GBDTData* train_data, int sample_size, RegressionNode* p_node, int depth);
    
    int predict(RegressionNode* node, GBDT_TUPLE_T* tuple, GBDTValue& res);
};

}
#endif

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

