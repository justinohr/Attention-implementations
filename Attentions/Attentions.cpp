#include "pch.h"
#include "Attentions.h"

nn::Matrix::Matrix(int _row)
{
	row = _row;
	col = 1;
	data = new float[row];
}

nn::Matrix::Matrix(int _row, int _col)
{
	row = _row;
	col = _col;
	data = new float[row * col];
}

nn::Matrix::Matrix(int _row, int _col, float* _data)
{
	row = _row;
	col = _col;
	data = new float[row * col];
	memcpy(data, _data, sizeof(float) * row * col);
}

nn::Matrix::~Matrix()
{
	if (data)
	{
		delete[] data;
	}
}

nn::Matrix& nn::Matrix::operator=(const Matrix& rhs)
{
	if (this == &rhs)
	{
		return *this;
	}

	if (data)
	{
		delete[] data;
	}

	row = rhs.row;
	col = rhs.col;
	data = new float[row * col];
	memcpy(data, rhs.data, sizeof(float) * row * col);
}



bool nn::matmul(const Matrix& A, const Matrix& B, Matrix& output)
{
	if (A.col != B.row)
	{
		return false;
	}

	output = Matrix(A.row, B.col);

#pragma omp parallel for
	for (int idx = 0; idx < A.row * B.col; idx++)
	{
		int output_row_idx = idx / B.col;
		int output_col_idx = idx % B.col;

		float dot_product_result = 0.f;
		for (int k = 0; k < A.col; ++k)
		{
			dot_product_result += A.data[output_row_idx * A.col + k]
				* B.data[k * B.col + output_col_idx];
		}
		output.data[idx] = dot_product_result;
	}

	return true;
}

void nn::transpose(const Matrix& input, Matrix& output)
{
	output = Matrix(input.col, input.row);

	for (int r = 0; r < input.row; ++r)
	{
		for (int c = 0; c < input.col; ++c)
		{
			output.data[c * input.row + r] = input.data[r * input.col + c];
		}
	}
}

bool nn::linear(const Matrix& A, const Matrix& proj, const Matrix& bias, Matrix& output)
{
	if (A.col != bias.row)
	{
		return false;
	}
	output = Matrix(A.row, proj.row);

	if (!matmul(A, proj, output))
	{
		return false;
	}

#pragma parallel for
	for (int r = 0; r < output.row; ++r)
	{
		for (int c = 0; c < output.col; ++c)
		{
			output.data[r * output.col + c] += bias.data[c];
		}
	}

	return true;
}


void nn::softmax(const Matrix& logit, Matrix& softmax)
{
	int row = logit.row;
	int col = logit.col;

	softmax = Matrix(row, col);

#pragma omp parallel for
	for (int r = 0; r < row; ++r)
	{
		float max_val = logit.data[r * col];
		for (int c = 1; c < col; ++c)
		{
			if (max_val < logit.data[r * col + c])
			{
				max_val = logit.data[r * col + c];
			}
		}

		float summation = 0.f;
		for (int c = 0; c < col; ++c)
		{
			float exp_val = exp(logit.data[r * col + c] - max_val);
			softmax.data[r * col + c] = exp_val;
			summation += exp_val;
		}
#pragma omp simd
		for (int c = 0; c < col; ++c)
		{
			softmax.data[r * col + c] /= summation;
		}
	}
}

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

bool nn::MultiHeadAttention::LoadWeights(float* qkv_proj, float* qkv_bias, float* output_proj, float* output_bias)
{
	Matrix q_proj_t = Matrix(m_nEmbeddingSize, m_nEmbeddingSize);
	m_q_bias = Matrix(m_nEmbeddingSize);
	
	Matrix k_proj_t = Matrix(m_nEmbeddingSize, m_nEmbeddingSize);
	m_k_bias = Matrix(m_nEmbeddingSize);
	
	Matrix v_proj_t = Matrix(m_nEmbeddingSize, m_nEmbeddingSize);
	m_v_bias = Matrix(m_nEmbeddingSize);

	Matrix o_proj_t = Matrix(m_nEmbeddingSize, m_nEmbeddingSize);
	m_o_bias = Matrix(m_nEmbeddingSize);

	memcpy(q_proj_t.data, qkv_proj, sizeof(float) * m_nEmbeddingSize * m_nEmbeddingSize);
	memcpy(m_q_bias.data, qkv_bias, sizeof(float) * m_nEmbeddingSize);

	memcpy(k_proj_t.data, qkv_proj + m_nEmbeddingSize * m_nEmbeddingSize, sizeof(float) * m_nEmbeddingSize * m_nEmbeddingSize);
	memcpy(m_k_bias.data, qkv_bias + m_nEmbeddingSize, sizeof(float) * m_nEmbeddingSize);

	memcpy(v_proj_t.data, qkv_proj + 2 * m_nEmbeddingSize * m_nEmbeddingSize, sizeof(float) * m_nEmbeddingSize * m_nEmbeddingSize);
	memcpy(m_v_bias.data, qkv_bias + 2 * m_nEmbeddingSize, sizeof(float) * m_nEmbeddingSize);

	memcpy(o_proj_t.data, output_proj, sizeof(float) * m_nEmbeddingSize * m_nEmbeddingSize);
	memcpy(m_o_bias.data, output_bias, sizeof(float) * m_nEmbeddingSize);
	
	transpose(q_proj_t, m_q_proj);
	transpose(k_proj_t, m_k_proj);
	transpose(v_proj_t, m_v_proj);
	transpose(o_proj_t, m_o_proj);

	return true;
}

bool nn::MultiHeadAttention::inference(float* input_vectors, float* output_vectors, int number_of_tokens)
{
	Matrix Q, K, V;
	Matrix x(number_of_tokens, m_nEmbeddingSize, input_vectors);

	if (!linear(x, m_q_proj, m_q_bias, Q))
	{
		return false;
	}
	if (!linear(x, m_k_proj, m_k_bias, K))
	{
		return false;
	}
	if (!linear(x, m_v_proj, m_v_bias, V))
	{
		return false;
	}
	
	Matrix head_outputs(number_of_tokens, m_nEmbeddingSize);
	memset(head_outputs.data, 0, sizeof(float) * number_of_tokens * m_nEmbeddingSize);
	Matrix K_t;
	transpose(K, K_t);

	for (int head_idx = 0; head_idx < m_nNumHeads; ++head_idx)
	{
		Matrix logit(number_of_tokens, number_of_tokens);
		memset(logit.data, 0, sizeof(float) * number_of_tokens * number_of_tokens);
		for (int i = 0; i < number_of_tokens; ++i)
		{
			for (int j = 0; j < number_of_tokens; ++j)
			{
				for (int k = head_idx * m_nHeadSize; k < (head_idx + 1) * m_nHeadSize; ++k)
				{
					logit.data[i * number_of_tokens + j] += Q.data[i * Q.col + k] * K_t.data[k * K_t.col + j];
				}
			}
		}

		float normalizer = sqrt(m_nHeadSize);
		for (int i = 0; i < number_of_tokens; ++i)
		{
			for (int j = 0; j < number_of_tokens; ++j)
			{
				logit.data[i * number_of_tokens + j] /= normalizer;
			}
		}

		Matrix softmax_output;
		softmax(logit, softmax_output);
		for (int i = 0; i < number_of_tokens; ++i)
		{
			for (int j = head_idx * m_nHeadSize; j < (head_idx + 1) * m_nHeadSize; ++j)
			{
				for (int k = 0; k < number_of_tokens; ++k)
				{
					head_outputs.data[i * m_nEmbeddingSize + j] +=
						softmax_output.data[i * number_of_tokens + k] * V.data[k * m_nEmbeddingSize + j];
				}
			}
		}
	}

	

	nn::Matrix output;
	if (!linear(head_outputs, m_o_proj, m_o_bias, output))
	{
		return false;
	}

	memcpy(output_vectors, output.data, sizeof(float) * output.row * output.col);

	return true;
}
