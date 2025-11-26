#ifndef UTILS_H
#define UTILS_H

typedef struct {
    char device[64];
    char vehicle_model[256];
    char plate_model[256];
    char ocr_model[256];
    float threshold;
} AppConfig;

int load_config(const char* path, AppConfig* config);

#endif