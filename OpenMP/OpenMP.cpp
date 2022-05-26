// Перед запуском
// 1. Включить OpenMP
// 2. Установить SDK для OpenCL
// 3. Добавить в с/c++ -> препроцессор -> определения препроцессора: _CRT_SECURE_NO_WARNINGS

#pragma comment(lib, "Win32/OpenCL.lib")

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <time.h>
#include <omp.h>
#include <intrin.h>
#include "lodepng.h"
#include <CL/cl.hpp>

#define __CL_ENABLE_EXCEPTIONS

std::string readFileToString(std::string fileName) {
    std::string file_content;
    std::getline(std::ifstream(fileName), file_content, '\0');
    return file_content;
}

void readImage(
    std::vector<unsigned char>& image,
    const std::string& filename,
    unsigned width,
    unsigned height
) {
    unsigned error;

    if ((error = lodepng::decode(image, width, height, filename))) {
        std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}

void writeImage(
    std::vector<unsigned char>& image,
    const std::string& filename,
    unsigned width,
    unsigned height
) {
    unsigned error;

    if ((error = lodepng::encode(filename, image, width, height))) {
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}

void sequentialAdd(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    long double beginTimestamp = omp_get_wtime();
    for (int i = 0; i < (int)inputImage1.size(); i++) {
        //outputImage[i] = std::min(255, inputImage1[i] + inputImage2[i]);
    }
    long double endTimestamp = omp_get_wtime();

    std::cout << "Sequential Add: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;
}

void openmpAdd(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    long double beginTimestamp = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < (int)inputImage1.size(); i++) {
        //outputImage[i] = std::min(255, inputImage1[i] + inputImage2[i]);
    }
    long double endTimestamp = omp_get_wtime();

    std::cout << "OpenMP Add: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;
}

void openclAdd(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    int* pInputVector1 = new int[inputImage1.size()];
    for (int i = 0; i < inputImage1.size(); i++) {
        pInputVector1[i] = inputImage1[i];
    }
    int* pInputVector2 = new int[inputImage2.size()];
    for (int i = 0; i < inputImage2.size(); i++) {
        pInputVector2[i] = inputImage2[i];
    }
    int* pOutputVector = new int[outputImage.size()];

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer clInputVector1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputImage1.size() * sizeof(int), pInputVector1);
    cl::Buffer clInputVector2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputImage2.size() * sizeof(int), pInputVector2);
    cl::Buffer clOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, outputImage.size() * sizeof(int), pOutputVector);

    std::string sourceCode = readFileToString("kernel.cl");

    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    cl::Program program = cl::Program(context, source);
    program.build(devices);
    cl::Kernel kernel(program, "main");

    int iArg = 0;
    kernel.setArg(iArg++, clInputVector1);
    kernel.setArg(iArg++, clInputVector2);
    kernel.setArg(iArg++, clOutputVector);

    long double beginTimestamp = omp_get_wtime();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(outputImage.size()), cl::NDRange(128));
    queue.finish();
    long double endTimestamp = omp_get_wtime();

    std::cout << "OpenCL Add: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;

    queue.enqueueReadBuffer(clOutputVector, CL_TRUE, 0, outputImage.size() * sizeof(int), pOutputVector);
    for (int i = 0; i < outputImage.size(); i++) {
        outputImage[i] = (unsigned char)pOutputVector[i];
    }

    delete[] pInputVector1;
    delete[] pInputVector2;
    delete[] pOutputVector;
}

void sequentialVectorizationAdd(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    long double beginTimestamp = omp_get_wtime();
    for (int i = 0; i < (int)inputImage1.size() - 16; i += 16) {
        __m128i part1 = _mm_loadu_si128((__m128i*) & inputImage1[i]);
        __m128i part2 = _mm_loadu_si128((__m128i*) & inputImage2[i]);

        __m128i result = _mm_adds_epu8(part1, part2);

        _mm_storeu_si128((__m128i*) & outputImage[i], result);
    }
    long double endTimestamp = omp_get_wtime();

    std::cout << "Sequential Vectorization Add: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;
}

void openmpVectorizationAdd(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    long double beginTimestamp = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < (int)inputImage1.size() - 16; i += 16) {
        __m128i part1 = _mm_loadu_si128((__m128i*) & inputImage1[i]);
        __m128i part2 = _mm_loadu_si128((__m128i*) & inputImage2[i]);

        __m128i result = _mm_adds_epu8(part1, part2);

        _mm_storeu_si128((__m128i*) & outputImage[i], result);
    }
    long double endTimestamp = omp_get_wtime();

    std::cout << "OpenMP Vectorization Add: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;
}

void sequentialSub(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    long double beginTimestamp = omp_get_wtime();
    for (int i = 0; i < (int)inputImage1.size(); i++) {
        //outputImage[i] = std::min(255, inputImage1[i] - inputImage2[i] + 256);
    }
    long double endTimestamp = omp_get_wtime();

    std::cout << "Sequential Sub: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;
}

void openmpSub(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    long double beginTimestamp = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < (int)inputImage1.size(); i++) {
        //outputImage[i] = std::min(255, inputImage1[i] - inputImage2[i] + 256);
    }
    long double endTimestamp = omp_get_wtime();

    std::cout << "OpenMP Sub: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;
}

void sequentialVectorizationSub(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    long double beginTimestamp = omp_get_wtime();
    for (int i = 0; i < (int)inputImage1.size() - 16; i += 16) {
        __m128i part1 = _mm_loadu_si128((__m128i*) & inputImage1[i]);
        __m128i part2 = _mm_loadu_si128((__m128i*) & inputImage2[i]);

        __m128i result = _mm_subs_epu8(part1, part2);

        _mm_storeu_si128((__m128i*) & outputImage[i], result);
    }
    long double endTimestamp = omp_get_wtime();

    std::cout << "Sequential Vectorization Sub: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;
}

void openmpVectorizationSub(
    std::vector<unsigned char>& inputImage1,
    std::vector<unsigned char>& inputImage2,
    std::vector<unsigned char>& outputImage
) {
    outputImage.clear();
    outputImage.resize(inputImage1.size());

    long double beginTimestamp = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < (int)inputImage1.size() - 16; i += 16) {
        __m128i part1 = _mm_loadu_si128((__m128i*) & inputImage1[i]);
        __m128i part2 = _mm_loadu_si128((__m128i*) & inputImage2[i]);

        __m128i result = _mm_subs_epu8(part1, part2);

        _mm_storeu_si128((__m128i*) & outputImage[i], result);
    }
    long double endTimestamp = omp_get_wtime();

    std::cout << "OpenMP Vectorization Sub: " << (endTimestamp - beginTimestamp) * 1000 << " ms" << std::endl;
}

const unsigned width = 2400;
const unsigned height = 2400;

int main() {
    std::string outputFilename;

    std::vector<unsigned char> image1;
    std::vector<unsigned char> image2;
    std::vector<unsigned char> outputImage;

    readImage(image1, "resources/2400x2400 1.png", width, height);
    readImage(image2, "resources/2400x2400 2.png", width, height);

    outputFilename = "output/sequential add.png";
    sequentialAdd(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);

    outputFilename = "output/openmp add.png";
    openmpAdd(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);

    outputFilename = "output/opencl add.png";
    openclAdd(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);

    outputFilename = "output/sequential vectorization add.png";
    sequentialVectorizationAdd(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);

    outputFilename = "output/openmp vectorization add.png";
    openmpVectorizationAdd(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);

    outputFilename = "output/sequential sub.png";
    sequentialSub(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);

    outputFilename = "output/openmp sub.png";
    openmpSub(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);

    outputFilename = "output/sequential vectorization sub.png";
    sequentialVectorizationSub(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);

    outputFilename = "output/openmp vectorization sub.png";
    openmpVectorizationSub(image1, image2, outputImage);
    writeImage(outputImage, outputFilename, width, height);
}
