#pragma once
#include "Attentions.h"
#include <fstream>
#include <math.h>

#pragma comment(lib, "Attentions.lib")

bool ReadBinaryFile(std::string file_path, unsigned char* data, unsigned int file_size);
bool isEqual(float* a, float* b, unsigned int length);