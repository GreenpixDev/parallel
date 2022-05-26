__kernel void main(
	__global const int* pInputVector1,
	__global const int* pInputVector2,
	__global int* pOutputVectorHost,
	int elementsNumber
) {
	int iJob = get_global_id(0);

	if (iJob >= elementsNumber) return;

	pOutputVectorHost[iJob] = min(255, pInputVector1[iJob] + pInputVector2[iJob]);
}