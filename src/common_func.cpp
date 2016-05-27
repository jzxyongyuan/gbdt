/**
* @file common_func.cpp
* @author jiangzhengxiang
* @date 2016/05/26 09:36:14
* @brief 
*  
**/

#include "common_func.h"

namespace gbdt
{

bool equal(GBDTValue num1, GBDTValue num2) {
    GBDTValue diff = fabs(num1 - num2);
    if (diff < 10e-5) {
        return true;
    }
    return false;
}

GBDTValue logit_loss_gradient(GBDTValue x, GBDTValue y) {
    return 2.0 * x / (1 + std::exp(2.0 * x * y));
}
}


/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

