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
	class BaseClass
	{
	protected:
		BaseClass() = default;
		~BaseClass() = default;
	
		bool linear(const cv::Mat& input, const cv::Mat& proj, const cv::Mat& bias, cv::Mat& output);
		void compute_causal_mask(cv::Mat& output, cv::Size sz);
	
		const float VERY_SMALL_NEGATIVE_FLOAT = -987654321.f;
	};

	class ATTENTIONS_API MultiHeadAttention : protected BaseClass
	{
	public:
		MultiHeadAttention() = default;
		MultiHeadAttention(int embedding_size, int num_heads);
		~MultiHeadAttention();
		bool load_weights(float* qkv_proj, float* qkv_bias, float* output_proj, float* output_bias);
		bool inference(float* input_vectors, float* output_vectors, int number_of_tokens, bool apply_causal_mask=false);

	private:
		int m_nEmbeddingSize = 0;
		int m_nNumHeads = 0;
		int m_nHeadSize = 0;

		cv::Mat m_q_proj;
		cv::Mat m_q_bias;
		
		cv::Mat m_k_proj;
		cv::Mat m_k_bias;

		cv::Mat m_v_proj;
		cv::Mat m_v_bias;

		cv::Mat m_o_proj;
		cv::Mat m_o_bias;
	};

	class ATTENTIONS_API GroupedQueryAttention : protected BaseClass
	{
	public:
		GroupedQueryAttention() = default;
		GroupedQueryAttention(int embedding_size, int num_query_heads, int num_kv_heads);
		~GroupedQueryAttention();
		bool load_weights(float* q_proj, float* q_bias, float* k_proj, float* k_bias, float* v_proj, float* v_bias, float* o_proj, float* o_bias);
		bool inference(float* input_vectors, float* output_vectors, int number_of_tokens, bool apply_causal_mask = false);

	private:
		int m_nEmbeddingSize = 0;
		int m_nHeadSize = 0;

		int m_nNumQueryHeads = 0;
		int m_nGroupSize = 0;
		int m_nNumKVHeads = 0;

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
