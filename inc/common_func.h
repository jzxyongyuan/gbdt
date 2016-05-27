/**
* @file common_func.h
* @author jiangzhengxiang
* @date 2016/05/25 19:08:32
* @brief 
*  
**/

#ifndef _INC_COMMON_H
#define _INC_COMMON_H

#include "gbdt_include.h"

namespace gbdt
{

template <typename T>
T squared(const T &v) {
  return v * v;
}

template <typename T>
T abs(const T &v) {
  return v >= 0? v : -v;
}

struct TupleCompare {
    TupleCompare(int i): index(i) {}

    bool operator () (const GBDT_TUPLE_T *t1, const GBDT_TUPLE_T *t2) {
        return t1 -> feature[index] < t2 -> feature[index];
    }

    int index;
};

bool equal(GBDTValue num1, GBDTValue num2);

GBDTValue logit_loss_gradient(GBDTValue x, GBDTValue y);

}

#endif

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

