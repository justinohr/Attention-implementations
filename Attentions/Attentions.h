#pragma once

#ifdef ATTENTIONS_EXPORTS
#define ATTENTIONS_API __declspec(dllexport)
#else
#define ATTENTIONS_API __declspec(dllimport)
#endif

#include <string.h>
#include <cmath>

namespace nn
{
	class Matrix
	{
	private:
		Matrix() = default;
		Matrix(int _row);
		Matrix(int _row, int _col);
		Matrix(int _row, int _col, float* _data);
		~Matrix();

		Matrix& operator=(const Matrix& rhs);

		int row = 0;
		int col = 0;
		float* data = nullptr;
		friend bool matmul(const Matrix& A, const Matrix& B, Matrix& output);
		friend void transpose(const Matrix& input, Matrix& output);
		friend bool linear(const Matrix& A, const Matrix& proj, const Matrix& bias, Matrix& output);
		friend void softmax(const Matrix& logit, Matrix& softmax);
		friend class MultiHeadAttention;
	};

	bool matmul(const Matrix& A, const Matrix& B, Matrix& output);
	void transpose(const Matrix& input, Matrix& output);
	bool linear(const Matrix& A, const Matrix& proj, const Matrix& bias, Matrix& output);
	void softmax(const Matrix& logit, Matrix& softmax);

	class ATTENTIONS_API MultiHeadAttention
	{
	public:
		MultiHeadAttention() = default;
		MultiHeadAttention(int embedding_size, int num_heads, bool causal_masking = false);
		~MultiHeadAttention();
		bool LoadWeights(float* qkv_proj, float* qkv_bias, float* output_proj, float* output_bias);
		bool inference(float* input_vectors, float* output_vectors, int number_of_tokens);

	private:
		int m_nEmbeddingSize = 0;
		int m_nNumHeads = 0;
		int m_nHeadSize = 0;
		bool m_bCausalMasking = false;

		Matrix m_q_proj;
		Matrix m_q_bias;
		
		Matrix m_k_proj;
		Matrix m_k_bias;

		Matrix m_v_proj;
		Matrix m_v_bias;

		Matrix m_o_proj;
		Matrix m_o_bias;
	};
}
