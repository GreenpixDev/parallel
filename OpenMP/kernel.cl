__kernel void main(
	__global const unsigned char* pInputVector1,
	__global const unsigned char* pInputVector2,
	__global unsigned char* pOutputVectorHost,
	int elementsNumber
) {
	int iJob = get_global_id(0);

	if (iJob >= elementsNumber) return;

	pOutputVectorHost[iJob] = min(255, pInputVector1[iJob] + pInputVector2[iJob]);
}