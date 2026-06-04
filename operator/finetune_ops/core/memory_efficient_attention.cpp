/**
 * @file memory_efficient_attention.cpp
 * @brief Memory-efficient attention implementation
 */

#include "memory_efficient_attention.h"
#include "ops.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace ops {

void online_softmax_weighted_sum(
    const float* logits,
    const float* values,
    int64_t seq_len,
    int64_t head_dim,
    float* output,
    float max_val
) {
    // First pass: compute normalization denominator (use max_val for stability)
    double sum_exp = 0.0;
    for (int64_t j = 0; j < seq_len; ++j) {
        sum_exp += std::exp(logits[j] - max_val);
    }
    
    // Second pass: compute normalized weights and accumulate into output
    float inv_sum = 1.0f / static_cast<float>(sum_exp);
    for (int64_t j = 0; j < seq_len; ++j) {
        float weight = std::exp(logits[j] - max_val) * inv_sum;
        const float* v_row = values + j * head_dim;
        
        for (int64_t d = 0; d < head_dim; ++d) {
            output[d] += weight * v_row[d];
        }
    }
}

TensorPtr memory_efficient_attention(
    const TensorPtr& q,
    const TensorPtr& k,
    const TensorPtr& v,
    const TensorPtr& causal_mask,
    const MemoryEfficientAttentionConfig& config
) {
    // Validate input shapes
    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();
    const auto& v_shape = v->shape();
    
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::runtime_error("memory_efficient_attention: inputs must be 4D [B,H,S,D]");
    }
    
    int64_t batch = q_shape[0];
    int64_t n_head = q_shape[1];
    int64_t seq_len = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    // Validate k/v shapes match
    if (k_shape[0] != batch || k_shape[1] != n_head || k_shape[2] != seq_len || k_shape[3] != head_dim) {
        throw std::runtime_error("memory_efficient_attention: k shape mismatch");
    }
    if (v_shape[0] != batch || v_shape[1] != n_head || v_shape[2] != seq_len || v_shape[3] != head_dim) {
        throw std::runtime_error("memory_efficient_attention: v shape mismatch");
    }
    
    // Auto-compute scale factor
    float scale = (config.scale > 0) ? config.scale : (1.0f / std::sqrt(static_cast<float>(head_dim)));
    
    // Prepare causal mask (if provided)
    const float* mask_data = nullptr;
    if (causal_mask) {
        if (causal_mask->shape().size() != 2 || 
            causal_mask->shape()[0] != seq_len || 
            causal_mask->shape()[1] != seq_len) {
            throw std::runtime_error("memory_efficient_attention: causal_mask must be [S,S]");
        }
        mask_data = causal_mask->data<float>();
    }
    
    // Create output tensor
    auto context = zeros({batch, n_head, seq_len, head_dim}, kFloat32, kCPU);
    
    const float* q_data = q->data<float>();
    const float* k_data = k->data<float>();
    const float* v_data = v->data<float>();
    float* ctx_data = context->data<float>();
    
    // Main loop: iterate over batch, head, and query rows
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            // Base offset for current head
            int64_t head_offset = (b * n_head + h) * seq_len * head_dim;
            const float* q_head = q_data + head_offset;
            const float* k_head = k_data + head_offset;
            const float* v_head = v_data + head_offset;
            float* ctx_head = ctx_data + head_offset;
            
            // Process each query position
            for (int64_t i = 0; i < seq_len; ++i) {
                const float* q_row = q_head + i * head_dim;
                float* ctx_row = ctx_head + i * head_dim;
                
                // === Pass 1: compute scores and max (numerically stable) ===
                // Reuse per-row buffer to avoid repeated allocations
                static thread_local std::vector<float> scores_buf;
                if (scores_buf.size() < static_cast<size_t>(seq_len)) {
                    scores_buf.resize(seq_len);
                }
                float* scores = scores_buf.data();
                float max_score = -std::numeric_limits<float>::infinity();
                
                for (int64_t j = 0; j < seq_len; ++j) {
                    // Compute dot product: q[i] · k[j]
                    const float* k_row = k_head + j * head_dim;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        dot += q_row[d] * k_row[d];
                    }
                    
                    // Apply scale
                    float score = dot * scale;
                    
                    // Apply causal mask
                    if (config.use_causal_mask && j > i) {
                        score = -1e10f;  // Mask upper triangle
                    }
                    
                    // Apply extra mask (if provided)
                    if (mask_data) {
                        score += mask_data[i * seq_len + j];
                    }
                    
                    scores[j] = score; // Write into reusable buffer
                    max_score = std::max(max_score, score);
                }
                
                // === Pass 2: online softmax and accumulate into context ===
                // Initialize output row
                std::fill(ctx_row, ctx_row + head_dim, 0.0f);
                
                // Compute normalization denominator
                double sum_exp = 0.0;
                for (int64_t j = 0; j < seq_len; ++j) {
                    sum_exp += std::exp(scores[j] - max_score);
                }
                
                // Compute weighted sum
                float inv_sum = 1.0f / static_cast<float>(sum_exp);
                for (int64_t j = 0; j < seq_len; ++j) {
                    float weight = std::exp(scores[j] - max_score) * inv_sum;
                    const float* v_row = v_head + j * head_dim;
                    
                    for (int64_t d = 0; d < head_dim; ++d) {
                        ctx_row[d] += weight * v_row[d];
                    }
                }
            }
        }
    }
    
    // Enable gradient propagation if needed
    if (q->requires_grad() || k->requires_grad() || v->requires_grad()) {
        context->set_requires_grad(true);
        
        // Save attn_weights during forward for backward correctness
        std::vector<float> saved_aw(B * H * S * S);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                const float* q_bh = q->data<float>() + ((b * H + h) * S) * D;
                const float* k_bh = k->data<float>() + ((b * H + h) * S) * D;
                float* aw_bh = saved_aw.data() + ((b * H + h) * S) * S;
                for (int i = 0; i < S; ++i) {
                    float max_s = -1e30f;
                    for (int j = 0; j <= i; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < D; ++d) s += q_bh[i*D+d] * k_bh[j*D+d];
                        s *= scale; if (s > max_s) max_s = s;
                        aw_bh[i*S+j] = s;
                    }
                    for (int j = i+1; j < S; ++j) aw_bh[i*S+j] = -1e10f;
                    float sum_e = 0.0f;
                    for (int j = 0; j < S; ++j) { aw_bh[i*S+j] = std::exp(aw_bh[i*S+j] - max_s); sum_e += aw_bh[i*S+j]; }
                    for (int j = 0; j < S; ++j) aw_bh[i*S+j] /= sum_e;
                }
            }
        }
        
        context->set_grad_fn([q, k, v, causal_mask, scale, saved_aw = std::move(saved_aw)](const TensorPtr& grad_output) -> std::vector<TensorPtr> {
            int B = q->shape()[0], H = q->shape()[1], S = q->shape()[2], D = q->shape()[3];
            const float* go = grad_output->data<float>();
            const float* qd = q->data<float>(), *kd = k->data<float>(), *vd = v->data<float>();
            const float* aw = saved_aw.data();
            auto gq = zeros(q->shape(), q->dtype(), q->device());
            auto gk = zeros(k->shape(), k->dtype(), k->device());
            auto gv = zeros(v->shape(), v->dtype(), v->device());
            float* gqd = gq->data<float>(), *gkd = gk->data<float>(), *gvd = gv->data<float>();
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H; ++h) {
                    const float* go_bh = go + ((b*H+h)*S)*D;
                    const float* q_bh = qd + ((b*H+h)*S)*D;
                    const float* k_bh = kd + ((b*H+h)*S)*D;
                    const float* v_bh = vd + ((b*H+h)*S)*D;
                    const float* aw_bh = aw + ((b*H+h)*S)*S;
                    float* gq_bh = gqd + ((b*H+h)*S)*D;
                    float* gk_bh = gkd + ((b*H+h)*S)*D;
                    float* gv_bh = gvd + ((b*H+h)*S)*D;
                    // grad_v = aw^T @ go
                    for (int j = 0; j < S; ++j)
                        for (int d = 0; d < D; ++d) { float s = 0.0f; for (int i = 0; i < S; ++i) s += aw_bh[i*S+j]*go_bh[i*D+d]; gv_bh[j*D+d] = s; }
                    // grad_attn = go @ v^T
                    std::vector<float> ga(S*S, 0.0f);
                    for (int i = 0; i < S; ++i)
                        for (int j = 0; j < S; ++j) { float s = 0.0f; for (int d = 0; d < D; ++d) s += go_bh[i*D+d]*v_bh[j*D+d]; ga[i*S+j] = s; }
                    // softmax backward
                    for (int i = 0; i < S; ++i) {
                        float dot = 0.0f;
                        for (int j = 0; j < S; ++j) dot += ga[i*S+j]*aw_bh[i*S+j];
                        for (int j = 0; j < S; ++j) ga[i*S+j] = aw_bh[i*S+j]*(ga[i*S+j] - dot);
                    }
                    for (int i = 0; i < S; ++i) for (int j = i+1; j < S; ++j) ga[i*S+j] = 0.0f;
                    // grad_q = ga @ k * scale
                    for (int i = 0; i < S; ++i)
                        for (int d = 0; d < D; ++d) { float s = 0.0f; for (int j = 0; j < S; ++j) s += ga[i*S+j]*k_bh[j*D+d]; gq_bh[i*D+d] = s*scale; }
                    // grad_k = ga^T @ q * scale
                    for (int j = 0; j < S; ++j)
                        for (int d = 0; d < D; ++d) { float s = 0.0f; for (int i = 0; i < S; ++i) s += ga[i*S+j]*q_bh[i*D+d]; gk_bh[j*D+d] = s*scale; }
                }
            }
            return {gq, gk, gv};
        });
    }
}

} // namespace ops

