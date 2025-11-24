#include "include/utils.h"
#include <ctype.h>

int config_parse(AppConfig* config, const char* config_path) {
    FILE* file = fopen(config_path, "r");
    if (!file) {
        fprintf(stderr, "无法打开配置文件: %s\n", config_path);
        return -1;
    }
    
    char line[256];
    char section[64] = "";
    
    // 设置默认值
    strcpy(config->camera_device, "/dev/video0");
    config->camera_width = 640;
    config->camera_height = 480;
    strcpy(config->vehicle_model, "models/vehicle_detector.onnx");
    strcpy(config->plate_model, "models/plate_detector.onnx");
    strcpy(config->ocr_model, "models/ppocr_rec.onnx");
    config->vehicle_threshold = 0.5f;
    config->plate_threshold = 0.5f;
    config->ocr_threshold = 0.5f;
    
    while (fgets(line, sizeof(line), file)) {
        // 移除换行符和空白字符
        char* ptr = line;
        while (*ptr && isspace(*ptr)) ptr++;
        char* end = ptr + strlen(ptr) - 1;
        while (end > ptr && isspace(*end)) *end-- = '\0';
        
        // 跳过空行和注释
        if (*ptr == '\0' || *ptr == '#' || *ptr == ';') continue;
        
        // 处理段标题
        if (*ptr == '[' && *end == ']') {
            *end = '\0';
            strncpy(section, ptr + 1, sizeof(section) - 1);
            continue;
        }
        
        // 处理键值对
        char* equal = strchr(ptr, '=');
        if (equal) {
            *equal = '\0';
            char* key = ptr;
            char* value = equal + 1;
            
            // 移除键值周围的空白
            while (*key && isspace(*key)) key++;
            char* key_end = key + strlen(key) - 1;
            while (key_end > key && isspace(*key_end)) *key_end-- = '\0';
            
            while (*value && isspace(*value)) value++;
            
            // 根据段和键设置配置值
            if (strcmp(section, "Camera") == 0) {
                if (strcmp(key, "device") == 0) strcpy(config->camera_device, value);
                else if (strcmp(key, "width") == 0) config->camera_width = atoi(value);
                else if (strcmp(key, "height") == 0) config->camera_height = atoi(value);
            }
            else if (strcmp(section, "Models") == 0) {
                if (strcmp(key, "vehicle_model") == 0) strcpy(config->vehicle_model, value);
                else if (strcmp(key, "plate_model") == 0) strcpy(config->plate_model, value);
                else if (strcmp(key, "ocr_model") == 0) strcpy(config->ocr_model, value);
            }
            else if (strcmp(section, "Thresholds") == 0) {
                if (strcmp(key, "vehicle") == 0) config->vehicle_threshold = atof(value);
                else if (strcmp(key, "plate") == 0) config->plate_threshold = atof(value);
                else if (strcmp(key, "ocr") == 0) config->ocr_threshold = atof(value);
            }
        }
    }
    
    fclose(file);
    return 0;
}