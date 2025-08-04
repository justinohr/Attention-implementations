#pragma once

#ifdef ATTENTIONS_EXPORTS
#define ATTENTIONS_API __declspec(dllexport)
#else
#define ATTENTIONS_API __declspec(dllimport)
#endif

#include <string.h>
#include <cmath>
#include "opencv2/core.hpp"

#ifdef _DEBUG
#pragma comment(lib, "opencv_world4120d.lib")
#else
#pragma comment(lib, "opencv_world4120.lib")
#endif

namespace nn
{
	class ATTENTIONS_API MultiHeadAttention
	{
	public:
		MultiHeadAttention() = default;
		MultiHeadAttention(int embedding_size, int num_heads, bool causal_masking = false);
		~MultiHeadAttention();
		bool load_weights(float* qkv_proj, float* qkv_bias, float* output_proj, float* output_bias);
		bool inference(float* input_vectors, float* output_vectors, int number_of_tokens);

	private:
		bool linear(const cv::Mat& input, const cv::Mat& proj, const cv::Mat& bias, cv::Mat& output);
		int m_nEmbeddingSize = 0;
		int m_nNumHeads = 0;
		int m_nHeadSize = 0;
		bool m_bCausalMasking = false;

		cv::Mat m_q_proj;
		cv::Mat m_q_bias;
		
		cv::Mat m_k_proj;
		cv::Mat m_k_bias;

		cv::Mat m_v_proj;
		cv::Mat m_v_bias;

		cv::Mat m_o_proj;
		cv::Mat m_o_bias;
	};
}
