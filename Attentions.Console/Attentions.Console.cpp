#include "Attentions.Console.h"

int main()
{
	int number_of_tokens = 1000;
	int embedding_size = 768;
	int num_heads = 64;

	nn::MultiHeadAttention multi_head_attn(embedding_size, num_heads);

	float* qkv_proj = new float[3 * embedding_size * embedding_size];
	float* qkv_bias = new float[3 * embedding_size];
	float* out_proj = new float[embedding_size * embedding_size];
	float* out_bias = new float[embedding_size];

	float* input_vectors = new float[number_of_tokens * embedding_size];
	float* output_vectors = new float[number_of_tokens * embedding_size];
	float* gt_vectors = new float[number_of_tokens * embedding_size];
	float* causal_output_vectors = new float[number_of_tokens * embedding_size];
	float* causal_gt_vectors = new float[number_of_tokens * embedding_size];
	

	if (!ReadBinaryFile("qkv_proj.raw", (unsigned char*)qkv_proj, sizeof(float) * 3 * embedding_size * embedding_size))
	{
		return EXIT_FAILURE;
	}

	if (!ReadBinaryFile("qkv_bias.raw", (unsigned char*)qkv_bias, sizeof(float) * 3 * embedding_size))
	{
		return EXIT_FAILURE;
	}

	if (!ReadBinaryFile("out_proj.raw", (unsigned char*)out_proj, sizeof(float) * embedding_size * embedding_size))
	{
		return EXIT_FAILURE;
	}

	if (!ReadBinaryFile("out_bias.raw", (unsigned char*)out_bias, sizeof(float) * embedding_size))
	{
		return EXIT_FAILURE;
	}

	if (!ReadBinaryFile("input.raw", (unsigned char*)input_vectors, sizeof(float) * number_of_tokens * embedding_size))
	{
		return EXIT_FAILURE;
	}

	if (!ReadBinaryFile("output.raw", (unsigned char*)gt_vectors, sizeof(float) * number_of_tokens * embedding_size))
	{
		return EXIT_FAILURE;
	}

	if (!ReadBinaryFile("causal_output.raw", (unsigned char*)causal_gt_vectors, sizeof(float) * number_of_tokens * embedding_size))
	{
		return EXIT_FAILURE;
	}

	multi_head_attn.load_weights(qkv_proj, qkv_bias, out_proj, out_bias);
	clock_t start_time = clock();
	multi_head_attn.inference(input_vectors, output_vectors, number_of_tokens);
	clock_t end_time = clock();

	multi_head_attn.inference(input_vectors, causal_output_vectors, number_of_tokens, true);

	if (isEqual(gt_vectors, output_vectors, number_of_tokens * embedding_size) && isEqual(causal_gt_vectors, causal_output_vectors, number_of_tokens * embedding_size))
	{
		printf("MultiHeadAttention test: PASS\n");
		printf("Elapsed time: %lf seconds\n\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
	}
	else
	{
		printf("MultiHeadAttention test: FAIL\n\n");
	}

	delete[] qkv_proj;
	delete[] qkv_bias;
	delete[] out_proj;
	delete[] out_bias;

	delete[] input_vectors;
	delete[] output_vectors;
	delete[] gt_vectors;
	delete[] causal_output_vectors;
	delete[] causal_gt_vectors;

	int num_kv_heads = 2;
	int num_groups = num_heads / num_kv_heads;
	int head_size = embedding_size / num_heads;

	input_vectors = new float[number_of_tokens * embedding_size];
	output_vectors = new float[number_of_tokens * embedding_size];
	gt_vectors = new float[number_of_tokens * embedding_size];
	causal_output_vectors = new float[number_of_tokens * embedding_size];
	causal_gt_vectors = new float[number_of_tokens * embedding_size];

	float* q_proj = new float[number_of_tokens * embedding_size];
	float* k_proj = new float[number_of_tokens * num_kv_heads * head_size];
	float* v_proj = new float[number_of_tokens * num_kv_heads * head_size];
	float* o_proj = new float[number_of_tokens * embedding_size];

	float* q_bias = new float[embedding_size];
	float* k_bias = new float[num_kv_heads * head_size];
	float* v_bias = new float[num_kv_heads * head_size];
	float* o_bias = new float[embedding_size];

	if (!ReadBinaryFile("q_proj_gqa.raw", (unsigned char*)q_proj, sizeof(float) * number_of_tokens * embedding_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("k_proj_gqa.raw", (unsigned char*)k_proj, sizeof(float) * number_of_tokens * num_kv_heads * head_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("v_proj_gqa.raw", (unsigned char*)v_proj, sizeof(float) * number_of_tokens * num_kv_heads * head_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("o_proj_gqa.raw", (unsigned char*)o_proj, sizeof(float) * number_of_tokens * embedding_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("q_bias_gqa.raw", (unsigned char*)q_bias, sizeof(float) * embedding_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("k_bias_gqa.raw", (unsigned char*)k_bias, sizeof(float) * num_kv_heads * head_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("v_bias_gqa.raw", (unsigned char*)v_bias, sizeof(float) * num_kv_heads * head_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("o_bias_gqa.raw", (unsigned char*)o_bias, sizeof(float) * embedding_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("input_gqa.raw", (unsigned char*)input_vectors, sizeof(float) * number_of_tokens * embedding_size))
	{
		return EXIT_FAILURE;
	}

	if (!ReadBinaryFile("output_gqa.raw", (unsigned char*)gt_vectors, sizeof(float) * number_of_tokens * embedding_size))
	{
		return EXIT_FAILURE;
	}
	if (!ReadBinaryFile("causal_output_gqa.raw", (unsigned char*)causal_gt_vectors, sizeof(float) * number_of_tokens * embedding_size))
	{
		return EXIT_FAILURE;
	}

	nn::GroupedQueryAttention grouped_query_attn(embedding_size, num_heads, num_kv_heads);

	grouped_query_attn.load_weights(q_proj, q_bias, k_proj, k_bias, v_proj, v_bias, o_proj, o_bias);
	start_time = clock();
	grouped_query_attn.inference(input_vectors, output_vectors, number_of_tokens);
	end_time = clock();

	grouped_query_attn.inference(input_vectors, causal_output_vectors, number_of_tokens, true);


	if (isEqual(gt_vectors, output_vectors, number_of_tokens * embedding_size) && isEqual(causal_gt_vectors, causal_output_vectors, number_of_tokens * embedding_size))
	{
		printf("GroupedQueryAttention test: PASS\n");
		printf("Elapsed time: %lf seconds\n\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
	}
	else
	{
		printf("GroupedQueryAttention test: FAIL\n\n");
	}

	delete[] input_vectors;
	delete[] output_vectors;
	delete[] gt_vectors;
	delete[] causal_output_vectors;
	delete[] causal_gt_vectors;

	delete[] q_proj;
	delete[] k_proj;
	delete[] v_proj;
	delete[] o_proj;

	delete[] q_bias;
	delete[] k_bias;
	delete[] v_bias;
	delete[] o_bias;


	return 0;
}

bool ReadBinaryFile(std::string file_path, unsigned char* data, unsigned int file_size)
{
	std::ifstream is(file_path, std::ifstream::binary);

	if (is)
	{
		is.read((char*)data, file_size);
		is.close();
		return true;
	}
	else
	{
		return false;
	}
}

bool isEqual(float* a, float* b, unsigned int length)
{
	for (int i = 0; i < length; ++i)
	{
		if (abs(a[i] - b[i]) > 1e-6 )
		{
			return false;
		}
	}
	return true;
}