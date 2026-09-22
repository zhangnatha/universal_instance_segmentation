#pragma once

#include <algorithm>
#include <cstddef>
#include <opencv2/core.hpp>

// HSV 转 RGB 颜色空间转换
inline void HSVtoRGB(int *r, int *g, int *b, int h, int s, int v) {
  const float rgbMax = v * 2.55f;
  const float rgbMin = rgbMax * (100 - s) / 100.0f;
  const int sector = h / 60;
  const int difference = h % 60;
  const float adjustment = (rgbMax - rgbMin) * difference / 60.0f;

  switch (sector) {
    case 0: *r = rgbMax; *g = rgbMin + adjustment; *b = rgbMin; break;
    case 1: *r = rgbMax - adjustment; *g = rgbMax; *b = rgbMin; break;
    case 2: *r = rgbMin; *g = rgbMax; *b = rgbMin + adjustment; break;
    case 3: *r = rgbMin; *g = rgbMax - adjustment; *b = rgbMax; break;
    case 4: *r = rgbMin + adjustment; *g = rgbMin; *b = rgbMax; break;
    default: *r = rgbMax; *g = rgbMin; *b = rgbMax - adjustment; break;
  }
}

// 根据类别 ID 与总类别数动态计算可视化渲染颜色（OpenCV BGR 格式）
inline cv::Scalar get_class_color(int class_id, size_t num_classes = 1) {
  int r = 0, g = 0, b = 0;
  const int h = static_cast<int>(360.0 / std::max<size_t>(1, num_classes) * std::max(0, class_id)) % 360;
  HSVtoRGB(&r, &g, &b, h, 100, 100);
  return cv::Scalar(b, g, r);
}
