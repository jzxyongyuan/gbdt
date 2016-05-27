/**
* @file test_main.cpp
* @author jiangzhengxiang
* @date 2016/05/23 17:58:46
* @brief 
*  
**/

#include "gbdt_include.h"
#include "gbdt.h"

gbdt::gbdt_conf_t g_conf;

int init_log(const char* proc_name) {
    FLAGS_log_dir = "log";     // 设置日志文件保存目录,这个目录必须是已经存在的,否则不能生成日志文件.

    google::InitGoogleLogging(proc_name);// 设置日志文件名中的"文件名"字段.

    DLOG(INFO) << "Init Log Success";
    return 0;
}

void show_usage(const char* cmd) {
    printf("usage: %s -d conf -f file\n", cmd);
    return;
}

int load_conf(const char* path, const char* file) {
    int ret = 0;
    char conf_file[gbdt::GBDT_MAX_PATH_LEN];

    snprintf(conf_file, gbdt::GBDT_MAX_PATH_LEN, "%s/%s", path, file);
    conf_file[gbdt::GBDT_MAX_PATH_LEN - 1] = '\0';

    cfgiml::Configure config;
    ret = config.load(conf_file);
    if (ret != 0) {
        LOG(WARNING)<<"load configure file "<<conf_file<<" failed";
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    char c = 0;
    bzero(&g_conf, sizeof(g_conf));
    ret = init_log(argv[0]);
    while ((c = getopt(argc, argv, "f:d:i:m:hv")) != -1) {
        switch (c) {
            case 'd':
                snprintf(g_conf.conf_path, gbdt::GBDT_MAX_PATH_LEN, "%s", optarg);
                g_conf.conf_path[gbdt::GBDT_MAX_PATH_LEN-1] = '\0';
                break;
            case 'f':
                snprintf(g_conf.conf_file, gbdt::GBDT_MAX_PATH_LEN, "%s", optarg);
                g_conf.conf_file[gbdt::GBDT_MAX_PATH_LEN-1] = '\0';
                break;
            case 'i':
                snprintf(g_conf.input_file, gbdt::GBDT_MAX_PATH_LEN, "%s", optarg);
                g_conf.input_file[gbdt::GBDT_MAX_PATH_LEN-1] = '\0';
                break;
            case 'm':
                snprintf(g_conf.model_file, gbdt::GBDT_MAX_PATH_LEN, "%s", optarg);
                g_conf.model_file[gbdt::GBDT_MAX_PATH_LEN-1] = '\0';
                break;
            case 'h':
            case 'v':
            case '?':
                show_usage(argv[0]);
                return 1;
        }
    }

    if (g_conf.conf_path[0] == 0 || g_conf.conf_file[0] == 0 || g_conf.input_file[0] == 0 || 
                g_conf.model_file[0] == 0) {
        show_usage(argv[0]);
        return 1;
    }

    ret = load_conf(g_conf.conf_path, g_conf.conf_file);
    if (ret != 0) {
        return ret;
    }
    
    gbdt::GBDT gbdt;
    ret = gbdt.init(g_conf.conf_path, g_conf.conf_file);
    if (ret != 0) {
        LOG(WARNING) << "fail to init gbdt";
        return ret;
    }

    ret = gbdt.load(g_conf.model_file);
    if (ret != 0) {
        LOG(WARNING) << "fail to load model from " << g_conf.model_file;
        return ret;
    }

    ret = gbdt.test(g_conf.input_file);
    if (ret != 0) {
        LOG(WARNING) << "fail to read gbdt using file " << g_conf.input_file;
        return ret;
    }
    return 0;
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
