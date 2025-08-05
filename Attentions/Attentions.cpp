#include "pch.h"
#include "Attentions.h"

nn::MultiHeadAttention::MultiHeadAttention(int embedding_size, int num_heads, bool causal_masking)
{
	m_nEmbeddingSize = embedding_size;
	m_nNumHeads = num_heads;
	m_nHeadSize = embedding_size / num_heads;
	m_bCausalMasking = causal_masking;

}

nn::MultiHeadAttention::~MultiHeadAttention()
{

}

bool nn::MultiHeadAttention::load_weights(float* qkv_proj, float* qkv_bias, float* output_proj, float* output_bias)
{
	m_q_proj = cv::Mat(m_nEmbeddingSize, m_nEmbeddingSize, CV_32F);
	m_q_bias = cv::Mat(1, m_nEmbeddingSize, CV_32F);
	
	m_k_proj = cv::Mat(m_nEmbeddingSize, m_nEmbeddingSize, CV_32F);
	m_k_bias = cv::Mat(1, m_nEmbeddingSize, CV_32F);
	
	m_v_proj = cv::Mat(m_nEmbeddingSize, m_nEmbeddingSize, CV_32F);
	m_v_bias = cv::Mat(1, m_nEmbeddingSize, CV_32F);

	m_o_proj = cv::Mat(m_nEmbeddingSize, m_nEmbeddingSize, CV_32F);
	m_o_bias = cv::Mat(1, m_nEmbeddingSize, CV_32F);

	memcpy(m_q_proj.data, qkv_proj, sizeof(float) * m_nEmbeddingSize * m_nEmbeddingSize);
	memcpy(m_q_bias.data, qkv_bias, sizeof(float) * m_nEmbeddingSize);

	memcpy(m_k_proj.data, qkv_proj + m_nEmbeddingSize * m_nEmbeddingSize, sizeof(float) * m_nEmbeddingSize * m_nEmbeddingSize);
	memcpy(m_k_bias.data, qkv_bias + m_nEmbeddingSize, sizeof(float) * m_nEmbeddingSize);

	memcpy(m_v_proj.data, qkv_proj + 2 * m_nEmbeddingSize * m_nEmbeddingSize, sizeof(float) * m_nEmbeddingSize * m_nEmbeddingSize);
	memcpy(m_v_bias.data, qkv_bias + 2 * m_nEmbeddingSize, sizeof(float) * m_nEmbeddingSize);

	memcpy(m_o_proj.data, output_proj, sizeof(float) * m_nEmbeddingSize * m_nEmbeddingSize);
	memcpy(m_o_bias.data, output_bias, sizeof(float) * m_nEmbeddingSize);
	
	cv::transpose(m_q_proj, m_q_proj);
	cv::transpose(m_k_proj, m_k_proj);
	cv::transpose(m_v_proj, m_v_proj);
	cv::transpose(m_o_proj, m_o_proj);

	return true;
}

bool nn::MultiHeadAttention::linear(const cv::Mat& input, const cv::Mat& proj, const cv::Mat& bias, cv::Mat& output)
{
	cv::Mat bias_repeated;
	cv::repeat(bias, input.rows, 1, bias_repeated);
	output = input * proj + bias_repeated;
	return true;
}


bool nn::MultiHeadAttention::inference(float* input_vectors, float* output_vectors, int number_of_tokens)
{
	cv::Mat Q, K, V;
	cv::Mat x(number_of_tokens, m_nEmbeddingSize, CV_32F, input_vectors);

	std::atomic<bool> success = true;
	
#pragma omp parallel sections
	{
#pragma omp section
		{
			if (!linear(x, m_q_proj, m_q_bias, Q))
			{
				success = false;
			}
		}
#pragma omp section
		{
			if (!linear(x, m_k_proj, m_k_bias, K))
			{
				success = false;
			}
		}
#pragma omp section
		{
			if (!linear(x, m_v_proj, m_v_bias, V))
			{
				success = false;
			}
		}
	}
	
	if (!success)
	{
		return false;
	}
	
	cv::Mat head_outputs = cv::Mat::zeros(number_of_tokens, m_nEmbeddingSize, CV_32F);
	cv::Mat K_t = K.t();
#pragma omp parallel for
	for (int head_idx = 0; head_idx < m_nNumHeads; ++head_idx)
	{
		cv::Mat logit;
		
		logit = Q(cv::Rect(head_idx * m_nHeadSize, 0, m_nHeadSize, number_of_tokens))
			* K_t(cv::Rect(0, head_idx * m_nHeadSize, number_of_tokens, m_nHeadSize));

		logit /= sqrt(m_nHeadSize);

		cv::Mat exp_logit;
		cv::Mat softmax_output;
		
		cv::exp(logit, exp_logit);
		cv::Mat exp_summation, denominator;
		cv::reduce(exp_logit, exp_summation, 1, cv::REDUCE_SUM, CV_32F);
		cv::repeat(exp_summation, 1, exp_logit.cols, denominator);

		softmax_output = exp_logit / denominator;

		cv::Mat result;
		result = softmax_output * V(cv::Rect(head_idx * m_nHeadSize, 0, m_nHeadSize, number_of_tokens));

		cv::Mat roi = head_outputs(cv::Rect(head_idx * m_nHeadSize, 0, m_nHeadSize, number_of_tokens));
		result.copyTo(roi);
	}

	cv::Mat output;
	if (!linear(head_outputs, m_o_proj, m_o_bias, output))
	{
		return false;
	}

	memcpy(output_vectors, output.data, sizeof(float) * output.rows * output.cols);

	return true;
}
