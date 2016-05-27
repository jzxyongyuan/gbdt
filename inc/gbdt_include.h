/**
* @file gbdt_include.h
* @author jiangzhengxiang
* @date 2016/05/23 18:14:38
* @brief 
*  
**/

#include "Configure.h"
#include "glog/logging.h"
#include <cmath>
#include <limits>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>

#ifndef INC_GBDT_INCLUDE_H
#define INC_GBDT_INCLUDE_H
namespace gbdt
{

typedef double GBDTValue;
const int    GBDT_MAX_DEPTH                = 2;
const int    GBDT_MAX_LINE_LEN             = 1024;
const int    GBDT_MAX_PATH_LEN             = 1024;
const int    GBDT_MAX_BUFF_LEN             = 512 * 1024 * 1024;
const int    GBDT_SQUARED_ERROR            = 1;
const int    GBDT_MIN_LEAF_SIZE            = 1;
const int    GBDT_LOG_LIKELIHOOD           = 2;
const int    GBDT_DEFAULT_ITERATIONS       = 10000;
const bool   GBDT_DEFAULT_INIT_GUESS       = false;
const bool   GBDT_DEFAULT_IGNORE_WEIGHT    = false;
const double GBDT_DEFAULT_SHRINKAGE        = 0.1;
const double GBDT_DEFAULT_FEAT_RATIO       = 1.0;
const double GBDT_DEFAULT_SAMPLE_RATIO     = 1.0;
const GBDTValue GBDT_MAX_VALUE             = std::numeric_limits<GBDTValue>::max();
const GBDTValue GBDT_MIN_VALUE             = std::numeric_limits<GBDTValue>::min();
const GBDTValue GBDT_UNKNOWN_VALUE         = GBDT_MIN_VALUE;

const char* const GBDT_ITEM_DELIM = " ";
const char* const GBDT_FEAT_DELIM = ":";
 
typedef struct _gbdt_conf_t
{
    char conf_file[GBDT_MAX_PATH_LEN];
    char conf_path[GBDT_MAX_PATH_LEN];

    char input_file[GBDT_MAX_PATH_LEN];
    char model_file[GBDT_MAX_PATH_LEN];
}gbdt_conf_t;

typedef struct _GBDT_TUPLE_T
{
    GBDTValue*  feature;
    GBDTValue   label;
    GBDTValue   target;
    GBDTValue   weight;
    GBDTValue   initial_guess;
} GBDT_TUPLE_T;

}
#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

