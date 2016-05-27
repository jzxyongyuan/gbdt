/**
* @file regression_tree.cpp
* @author jiangzhengxiang
* @date 2016/05/26 10:58:37
* @brief 
*  
**/

#include "regression_tree.h"

namespace gbdt
{
RegressionNode::RegressionNode(int idx) {
    index      = idx;
    leaf       = false;
    child      = new RegressionNode*[3];
    feat_index = -1;
    bzero(child, sizeof(RegressionNode*) * 3);
}

int RegressionNode::save(FILE* fp) {
    int tmp = -1;
    fwrite(&index,      sizeof(int),       1, fp);
    fwrite(&feat_index, sizeof(int),       1, fp);
    fwrite(&leaf,       sizeof(bool),      1, fp);
    fwrite(&value,      sizeof(GBDTValue), 1, fp);
    fwrite(&predict,    sizeof(GBDTValue), 1, fp);
    DLOG(INFO) << index << " " << feat_index << " " << leaf << " " << 
            value << " " << predict;
    for (int i = 0; i < 3; i++) {
        if (child[i] == NULL) {
            fwrite(&tmp,              sizeof(int), 1, fp);
            DLOG(INFO) << index << " " << tmp;
        } else {
            fwrite(&child[i] -> index, sizeof(int), 1, fp);
            DLOG(INFO) << index << " " << child[i] -> index;
        }
    }
    for (int i = 0; i < 3; i++) {
        if (child[i] != NULL) {
            child[i] -> save(fp);
        }
    }
    return 0;
}

int RegressionNode::load(FILE* fp) {
    int tmp = -1;
    fread(&index,      sizeof(int),       1, fp);
    fread(&feat_index, sizeof(int),       1, fp);
    fread(&leaf,       sizeof(bool),      1, fp);
    fread(&value,      sizeof(GBDTValue), 1, fp);
    fread(&predict,    sizeof(GBDTValue), 1, fp);
    DLOG(INFO) << index << " " << feat_index << " " << leaf << " " << 
            value << " " << predict;
    for (int i = 0; i < 3; i++) {
        fread(&tmp,     sizeof(int), 1, fp);
        child[i] = (RegressionNode*)(long)tmp;
        DLOG(INFO) << index << " " << tmp;
    }
    return 0;
}

int RegressionNode::load(std::vector<RegressionNode* >* node_vec) {
    int tmp = -1;
    for (int i = 0; i < 3; i++) {
        tmp = (int)(long)child[i];
        if (tmp == -1) {
            child[i] = NULL;
        } else {
            child[i] = NULL;
            for (int j = 1; j < (int)node_vec -> size(); j++) {
                if ((*node_vec)[j] -> index == tmp) {
                    child[i] = (*node_vec)[j];
                    child[i] -> load(node_vec);
                }
            }
            if (child[i] == NULL) {
                return 1;
            }
        }
    }
    return 0;
}

bool RegressionTree::same(GBDTData* train_data, int sample_size) {
    if (sample_size <= 1) {
        return true;
    }

    GBDTValue target = (*train_data)[0] -> target;
    for (int i = 1; i < sample_size; i++) {
        if (!equal(target, (*train_data)[i]->target)) {
            return false;
        }
    }
    return true;
}

GBDTValue RegressionTree::average(GBDTData* train_data, int sample_size) {
    GBDTValue s = 0;
    GBDTValue c = 0;
    for (int i = 0; i < sample_size; i++) {
        s += (*train_data)[i]->target * (*train_data)[i]->weight;
        c += (*train_data)[i]->weight;
    }
    if (c == 0) {
        return static_cast<GBDTValue>(0);
    } else {
        return static_cast<GBDTValue> (s / c);
    }
}

GBDTValue RegressionTree::logit_opt_value(GBDTData* train_data, int sample_size) {
    GBDTValue s = 0;
    GBDTValue c = 0;
    GBDTValue y = 0;
    for (int i = 0; i < sample_size; i++) {
        s += (*train_data)[i]->target * (*train_data)[i]->weight;
        y = abs((*train_data)[i]->target);
        c += y *(2 - y) * (*train_data)[i]->weight;
    }

    if (c == 0) {
        return static_cast<GBDTValue>(0);
    } else {
        return static_cast<GBDTValue> (s / c);
    }
}

int RegressionTree::get_impurity(GBDTData* train_data, int sample_size, int feat_index, 
        GBDTValue& value, GBDTValue& impurity, GBDTValue& gain) {  
    gain     = 0;
    value    = GBDT_UNKNOWN_VALUE;
    impurity = GBDT_MAX_VALUE;

    train_data -> sort(feat_index);
    
    int unknown  = 0;
    GBDTValue s  = 0;
    GBDTValue ss = 0;
    GBDTValue c  = 0;

    for (int j = 0; j < sample_size && 
            (*train_data)[j] -> feature[feat_index] == GBDT_UNKNOWN_VALUE; j++) {
        s  += (*train_data)[j] -> target * (*train_data)[j] -> weight;
        ss += squared((*train_data)[j] -> target) * (*train_data)[j] -> weight;
        c  += (*train_data)[j] -> weight;
        unknown++;
    }

    if (unknown == sample_size) {
        return 1;
    }

    double fitness0 = c > 1 ? (ss - s * s / c) : 0;
    if (fitness0 < 0) {
        fitness0 = 0;
    }

    s = 0;
    ss = 0;
    c = 0;
    for (int j = unknown; j < sample_size; ++j) {
        s  += (*train_data)[j] -> target * (*train_data)[j] -> weight;
        ss += squared((*train_data)[j] -> target) * (*train_data)[j] -> weight;
        c  += (*train_data)[j] -> weight;
    }

    double fitness00 = c > 1 ? (ss - s * s / c) : 0;

    double ls = 0, lss = 0, lc = 0;
    double rs = s, rss = ss, rc = c;
    double fitness1 = 0, fitness2 = 0;
    for (int j = unknown; j < sample_size - 1; j++) {
        s  = (*train_data)[j] -> target * (*train_data)[j] -> weight;
        ss = squared((*train_data)[j] -> target) * (*train_data)[j] -> weight;
        c  = (*train_data)[j] -> weight;

        ls  += s;
        lss += ss;
        lc  += c;

        rs  -= s;
        rss -= ss;
        rc  -= c;

        GBDTValue f1 = (*train_data)[j]     -> feature[feat_index];
        GBDTValue f2 = (*train_data)[j + 1] -> feature[feat_index];
        if (equal(f1, f2)) {
            continue;
        }

        fitness1 = lc > 1 ? (lss - ls * ls / lc) : 0;
        if (fitness1 < 0) {
            fitness1 = 0;
        }

        fitness2 = rc > 1 ? (rss - rs * rs / rc) : 0;
        if (fitness2 < 0) {
            fitness2 = 0;
        }

        double fitness = fitness0 + fitness1 + fitness2;

        if (_m_feature_costs && _m_enable_feature_tunning) {
            fitness *= _m_feature_costs[feat_index];
        }

        if (impurity > fitness) {
            impurity = fitness;
            value = (f1 + f2) / 2;
            gain = fitness00 - fitness1 - fitness2;
        }
    }

    return impurity != GBDT_MAX_VALUE ? 0 : 1;
}

int RegressionTree::find_split(GBDTData* train_data, int sample_size, int& feat_index, 
        GBDTValue& value, GBDTValue& gain) {
    int ret = 0;
    GBDTValue best_fitness = GBDT_MAX_VALUE;

    std::vector<int> feature_vec;
    for (int i = 0; i < _m_feature_size; ++i) {
        feature_vec.push_back(i);
    }

    int feature_size = _m_feature_size;
    if (_m_feature_sample_ratio < 1) {
        feature_size = static_cast<int>(_m_feature_size * _m_feature_sample_ratio);
        std::random_shuffle(feature_vec.begin(), feature_vec.end());
    }

    DLOG(INFO) << "feature_size " << feature_size;

    for (int feat = 0; feat < feature_size; feat++) {
        int idx = feature_vec[feat];
        GBDTValue gain_t;
        GBDTValue value_t;
        GBDTValue impurity;
        ret = get_impurity(train_data, sample_size, idx, value_t, impurity, gain_t);
        if (ret != 0) {
            continue;
        }
        // Choose feature with smallest impurity to split.  If there's
        // no unknown value, it's equivalent to choose feature with
        // largest gain
        if (best_fitness > impurity) {
            feat_index   = idx;
            value        = value_t;
            gain         = gain_t;
            best_fitness = impurity;
        }
    }
    DLOG(INFO) << " best_fitness " << best_fitness << " feat_index " << feat_index << 
            " value " << value << " gain " << gain;

    return best_fitness != GBDT_MAX_VALUE ? 0 : 1;
}

int RegressionTree::split_data(GBDTData* train_data, GBDTData* data_split, int sample_size, int feat_index, GBDTValue value) {
    for (int i = 0; i < sample_size; ++i) {
        if ((*train_data)[i] -> feature[feat_index] == GBDT_UNKNOWN_VALUE) {
            data_split[2].push_back((*train_data)[i]);
        } else if ((*train_data)[i] -> feature[feat_index] < value) {
            data_split[0].push_back((*train_data)[i]);
        } else {
            data_split[1].push_back((*train_data)[i]);
        }
    }
    return 0;
}

int RegressionTree::fit(GBDTData* train_data, int sample_size) {
    _m_root = new RegressionNode(_m_node_cnt++);
    _m_gain = new GBDTValue[_m_feature_size];
    for (int i = 0; i < _m_feature_size; i++) {
        _m_gain[i] = 0.0;
    }
    fit(train_data, sample_size, _m_root, 0);
    return 0;
}

int RegressionTree::fit(GBDTData* train_data, int sample_size, RegressionNode* p_node, 
        int depth) {
    int ret = 0;
    if (_m_loss_type == GBDT_SQUARED_ERROR) {
        p_node -> predict = average(train_data, sample_size);
    } else if (_m_loss_type == GBDT_LOG_LIKELIHOOD) {
        p_node -> predict = logit_opt_value(train_data, sample_size);
    } else {
        return 1;
    }

    DLOG(INFO) << "predict: " << p_node -> predict;

    if (depth == _m_max_depth || same(train_data, sample_size) || sample_size <= _m_min_leaf_size) {
        p_node -> leaf = true;
        return 0;
    }
    
    GBDTValue gain = 0.0;
    ret = find_split(train_data, sample_size, p_node -> feat_index, p_node -> value, gain);
    if (ret != 0) {
        p_node -> leaf = true;
        return 0;
    }

    GBDTData data_split[3];
    split_data(train_data, data_split, sample_size, p_node -> feat_index, p_node -> value);
    if (data_split[0].empty() || data_split[1].empty()) {
        p_node -> leaf = true;
        return 0;
    }

    if (_m_gain[p_node -> feat_index] < gain) {
        _m_gain[p_node -> feat_index] = gain;
    }
    if (_m_feature_costs && _m_enable_feature_tunning) {
        _m_feature_costs[p_node -> feat_index] += 1.0e-4;
    }
    p_node -> child[0] = new RegressionNode(_m_node_cnt++);
    p_node -> child[1] = new RegressionNode(_m_node_cnt++);

    fit(&data_split[0], data_split[0].size(), p_node -> child[0], depth + 1);
    fit(&data_split[1], data_split[1].size(), p_node -> child[1], depth + 1);

    if (!data_split[2].empty()) {
        p_node -> child[2] = new RegressionNode(_m_node_cnt++);
        fit(&data_split[2], data_split[2].size(), p_node -> child[2], depth + 1);
    }
    return 0;
}

int RegressionTree::predict(GBDT_TUPLE_T* tuple, GBDTValue& res) {
    return predict(_m_root, tuple, res);
}

int RegressionTree::predict(RegressionNode* node, GBDT_TUPLE_T* tuple, GBDTValue& res) {
    if (tuple == NULL) {
        res = 0;
        return 1;
    }  
    if (node -> leaf) {
        DLOG(INFO) << "index: " << node -> index << " feat_index: " << node -> feat_index << 
                " predict: " << node -> predict;
        res = node -> predict;
        return 0;
    }
    if (tuple -> feature[node -> feat_index] == GBDT_UNKNOWN_VALUE) {
        if (node -> child[2]) {
            return predict(node -> child[2], tuple, res);
        } else {
            DLOG(INFO) << "index: " << node -> index << " feat_index: " << node -> feat_index << 
                    " predict: " << node -> predict;
            res = node -> predict;
            return 0;
        }
    } else if (tuple -> feature[node -> feat_index] < node -> value) {
        return predict(node -> child[0], tuple, res);
    } else {
        return predict(node -> child[1], tuple, res);
    }

    return 0;
}

int RegressionTree::save(FILE* fp) {
    if (fp == NULL) {
        return 1;
    }
    fwrite(&_m_node_cnt, sizeof(int), 1, fp);
    _m_root -> save(fp);
    return 0;
}

int RegressionTree::load(FILE* fp) {
    if (fp == NULL) {
        return 1;
    }
    std::vector<RegressionNode* > node_vec;
    fread(&_m_node_cnt, sizeof(int), 1, fp);
    for (int i = 0; i < _m_node_cnt; i++) {
        RegressionNode* tmp = new RegressionNode(i);
        tmp -> load(fp);
        node_vec.push_back(tmp);
    }
    _m_root = node_vec[0];
    _m_root -> load(&node_vec);

    node_vec.clear();
    return 0;
}

}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

