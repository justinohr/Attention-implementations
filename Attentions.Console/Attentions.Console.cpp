#include "Attentions.Console.h"

int main()
{
	int number_of_tokens = 1000;
	int embedding_size = 768;
	int num_heads = 64;
	bool causal_masking = false;

	nn::MultiHeadAttention multi_head_attn(embedding_size, num_heads, causal_masking);

	float* qkv_proj = new float[3 * embedding_size * embedding_size];
	float* qkv_bias = new float[3 * embedding_size];
	float* out_proj = new float[embedding_size * embedding_size];
	float* out_bias = new float[embedding_size];

	float* input_vectors = new float[number_of_tokens * embedding_size];
	float* output_vectors = new float[number_of_tokens * embedding_size];
	float* gt_vectors = new float[number_of_tokens * embedding_size];

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

	multi_head_attn.load_weights(qkv_proj, qkv_bias, out_proj, out_bias);
	clock_t start_time = clock();
	multi_head_attn.inference(input_vectors, output_vectors, number_of_tokens);
	clock_t end_time = clock();

	if (isEqual(gt_vectors, output_vectors, number_of_tokens * embedding_size))
	{
		printf("PASS\n");
		printf("Elapsed time: %lf seconds", (double)(end_time - start_time) / CLOCKS_PER_SEC);
	}
	else
	{
		printf("FAIL");
	}


	delete[] qkv_proj;
	delete[] qkv_bias;
	delete[] out_proj;
	delete[] out_bias;

	delete[] input_vectors;
	delete[] output_vectors;

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