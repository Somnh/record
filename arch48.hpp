#pragma once

#if defined(__aarch64__)

#include <cstdint>
#include <arm_neon.h>

namespace glass {

inline float reduce_add_f32x4(float32x4_t x) {
  // 垂直累加 x 中的 4 个元素
  float32x2_t sum_low = vadd_f32(vget_low_f32(x), vget_high_f32(x)); // 将 4 个数的前两个和后两个相加
  float32x2_t sum = vpadd_f32(sum_low, sum_low);  // 将结果水平加法
  return vget_lane_f32(sum, 0);  // 提取最终结果
}


inline int32_t reduce_add_i32x4(int32x4_t x) {
  // 垂直累加 x 中的 4 个元素
  int32x2_t sum_low = vadd_s32(vget_low_s32(x), vget_high_s32(x)); // 将 4 个数的前两个和后两个相加
  int32x2_t sum = vpadd_s32(sum_low, sum_low);  // 将结果水平加法
  return vget_lane_s32(sum, 0);  // 提取最终结果
}
} // namespace glass

#endif