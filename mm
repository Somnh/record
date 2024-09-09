#include <arm_neon.h>

float L2SqrSQ8_ext_neon(const float *x, const uint8_t *y, int d, const float *mi, const float *dif) {
    float32x4_t sum = vdupq_n_f32(0.0f);      // 初始化 sum 为 0
    float32x4_t dot5 = vdupq_n_f32(0.5f);     // 初始化 0.5f 的向量
    float32x4_t const_255 = vdupq_n_f32(255.0f); // 初始化 255.0f 的向量

    for (int i = 0; i < d; i += 4) {
        // 加载 4 个 uint8_t 数据并转换为 float
        uint8x8_t zz_u8 = vld1_u8(y + i);     // 从 y 加载 4 个 uint8_t
        uint16x8_t zz_u16 = vmovl_u8(zz_u8);  // 将 uint8_t 转换为 uint16_t
        float32x4_t yy = vcvtq_f32_u32(vmovl_u16(vget_low_u16(zz_u16))); // 将下半部分转换为 float32

        // 添加 0.5f 并继续操作
        yy = vaddq_f32(yy, dot5);  // yy + 0.5

        // 加载 mi 和 dif
        float32x4_t mi_vec = vld1q_f32(mi + i);
        float32x4_t dif_vec = vld1q_f32(dif + i);

        // yy = yy * dif + mi * 255
        yy = vmulq_f32(yy, dif_vec);  // yy * dif
        yy = vmlaq_f32(yy, mi_vec, const_255); // yy + (mi * 255)

        // 加载 x
        float32x4_t x_vec = vld1q_f32(x + i);

        // d = (x * 255) - yy
        float32x4_t d = vsubq_f32(vmulq_f32(x_vec, const_255), yy);

        // sum += d * d
        sum = vfmaq_f32(sum, d, d); // sum = sum + (d * d)
    }

    // 水平加法将 sum 中的 4 个元素相加
    float32x2_t sum_low = vadd_f32(vget_low_f32(sum), vget_high_f32(sum)); // 先将低位和高位相加
    float32x2_t final_sum = vpadd_f32(sum_low, sum_low); // 水平加法
    return vget_lane_f32(final_sum, 0);  // 返回最终的累加结果
}
