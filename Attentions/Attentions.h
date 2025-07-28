#pragma once

#ifdef ATTENTIONS_EXPORTS
#define ATTENTIONS_API __declspec(dllexport)
#else
#define ATTENTIONS_API __declspec(dllimport)
#endif

class BaseModule
{
protected:
	BaseModule() = default;
	~BaseModule() = default;	
	bool matmul(const float* a, int a_row, int a_col, const float* b, int b_row, int b_col, float* output);
	bool transpose(const float* input, int a_row, int a_col, float* output);
	bool softmax(const float* logit, int row, int col, float* softmax);
};

class AttentionHead : public BaseModule
{
	AttentionHead() = default;
	AttentionHead(int embedding_size, int head_size, bool causal_masking = false);
	bool LoadWeights(float* query_proj_weight, float* key_proj_weight, float* val_proj_weight);
	bool inference(float* input_vectors, float* output_vectors, int number_of_tokens);
	~AttentionHead();
private:
	int m_nEmbeddingSize;
	int m_nHeadSize;
	bool m_bCausalMasking;

	float* query_projection = nullptr; // Dxh
	float* key_projection = nullptr; // Dxh
	float* value_projection = nullptr; // Dxh

	friend class MultiHeadAttention;
};
class ATTENTIONS_API MultiHeadAttention
{
public:
	MultiHeadAttention() = default;
	MultiHeadAttention(int embedding_size, int num_heads, bool causal_masking=false);
	~MultiHeadAttention() = default;
	bool inference(float* input_vectors, float* output_vectors, int number_of_tokens);

private:
	int m_nEmbeddingSize;
	int m_nNumHeads;
	int m_nHeadSize;
	bool m_bCausalMasking;

	float* query_projection;
	float* key_projection;
	float* value_projection;
};