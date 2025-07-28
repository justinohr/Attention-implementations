#include "pch.h"
#include "Attentions.h"
#include <string.h>
#include <cmath>


bool BaseModule::matmul(const float* a, int a_row, int a_col, const float* b, int b_row, int b_col, float* output)
{
	if (a_col != b_row)
	{
		return false;
	}

#pragma omp parallel for
	for (int idx = 0; idx < a_row * b_col; idx++)
	{
		int output_row_idx = idx / b_col;
		int output_col_idx = idx % b_col;

		float dot_product_result = 0.f;
		for (int k = 0; k < a_col; ++k)
		{
			dot_product_result += a[output_row_idx * a_col + k]
				* b[k * b_col + output_col_idx];
		}
		output[idx] = dot_product_result;
	}

	return true;
}

bool BaseModule::transpose(const float* input, int row, int col, float* output)
{
	for (int r = 0; r < row; ++r)
	{
		for (int c = 0; c < col; ++c)
		{
			output[c * row + r] = input[r * col + c];
		}
	}
	return true;
}

bool BaseModule::softmax(const float* logit, int row, int col, float* softmax)
{
#pragma omp parallel for
	for (int r = 0; r < row; ++r)
	{
		float max_val = logit[r * col];
		for (int c = 1; c < col; ++c)
		{
			if (max_val < logit[r * col + c])
			{
				max_val = logit[r * col + c];
			}
		}

		float summation = 0.f;
		for (int c = 0; c < col; ++c)
		{
			float exp_val = exp(logit[r * col + c] - max_val);
			softmax[r * col + c] = exp_val;
			summation += exp_val;
		}
#pragma omp simd
		for (int c = 0; c < col; ++c)
		{
			softmax[r * col + c] /= summation;
		}
	}
}


AttentionHead::AttentionHead(int embedding_size, int head_size, bool causal_masking)
{
	m_nEmbeddingSize = embedding_size;
	m_nHeadSize = head_size;
	m_bCausalMasking = causal_masking;
}

AttentionHead::~AttentionHead()
{
	if (query_projection)
	{
		delete[] query_projection;
	}
	if (key_projection)
	{
		delete[] key_projection;
	}
	if (value_projection)
	{
		delete[] value_projection;
	}
}

bool AttentionHead::inference(float* input_vectors, float* output_vectors, int number_of_tokens)
{
	float* Q = new float[number_of_tokens * m_nHeadSize];
	float* K = new float[number_of_tokens * m_nHeadSize];
	float* V = new float[number_of_tokens * m_nHeadSize];

	memset(Q, 0, sizeof(float) * number_of_tokens * m_nHeadSize);
	memset(K, 0, sizeof(float) * number_of_tokens * m_nHeadSize);
	memset(V, 0, sizeof(float) * number_of_tokens * m_nHeadSize);
	if (!matmul(input_vectors, number_of_tokens, m_nEmbeddingSize, query_projection, m_nEmbeddingSize, m_nHeadSize, Q))
	{
		return false;
	}
	if (!matmul(input_vectors, number_of_tokens, m_nEmbeddingSize, key_projection, m_nEmbeddingSize, m_nHeadSize, K))
	{
		return false;
	}
	if (!matmul(input_vectors, number_of_tokens, m_nEmbeddingSize, value_projection, m_nEmbeddingSize, m_nHeadSize, V))
	{
		return false;
	}

	float* logit_vectors = new float[number_of_tokens * number_of_tokens];
	float* softmax_vectors = new float[number_of_tokens * number_of_tokens];

	float* K_t = new float[number_of_tokens * m_nHeadSize];

	transpose(K, number_of_tokens, m_nHeadSize, K_t);
	matmul(Q, number_of_tokens, m_nHeadSize, K_t, m_nHeadSize, number_of_tokens, logit_vectors);
	float normalizer = sqrt(m_nHeadSize);
#pragma omp simd
	for (int idx = 0; idx < number_of_tokens * number_of_tokens; idx++)
	{
		logit_vectors[idx] /= normalizer;
	}
	softmax(logit_vectors, number_of_tokens, number_of_tokens, softmax_vectors);

	delete[] Q;
	delete[] K;
	delete[] K_t;
	delete[] logit_vectors;

	matmul(softmax_vectors, number_of_tokens, number_of_tokens, V, number_of_tokens, m_nHeadSize, output_vectors);
	delete[] softmax_vectors;
	delete[] V;
}

bool AttentionHead::LoadWeights(float* query_proj_weight, float* key_proj_weight, float* val_proj_weight)
{
	memcpy(query_projection, query_proj_weight, sizeof(float) * m_nEmbeddingSize * m_nHeadSize);
	memcpy(key_projection, key_proj_weight, sizeof(float) * m_nEmbeddingSize * m_nHeadSize);
	memcpy(value_projection, val_proj_weight, sizeof(float) * m_nEmbeddingSize * m_nHeadSize);
	return true;
}

MultiHeadAttention::MultiHeadAttention(int embedding_size, int num_heads, bool causal_masking)
{
	m_nEmbeddingSize = embedding_size;
	m_nNumHeads = num_heads;
	m_nHeadSize = embedding_size / num_heads;
	m_bCausalMasking = causal_masking;
}
