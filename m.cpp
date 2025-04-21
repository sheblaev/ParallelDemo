#include <stdio.h>
#include <stdlib.h> 
#include <strings.h>
#include <math.h> 
#include "lodepng.h" 
#include <x86intrin.h>

#include <chrono>
#include <iostream>

#include <boost/asio.hpp>

using namespace std;

const int WINDOW_WIDTH  = 19200;
const int WINDOW_HEIGHT = 10800;
const int N_THREADS = 12;

const int   MAX_ITERATION_DEPTH = 256;
const float MAX_RADIUS          = 2.0f;

const float MAX_RADIUS_2 = MAX_RADIUS * MAX_RADIUS;
const float ASPECT_RATIO = (float)(WINDOW_WIDTH) / WINDOW_HEIGHT;

const float MAGNIFIER_OFFSET = -0.3f;
const float SHIFT_X_OFFSET   = -0.5f;

const int VEC_SIZE=8;

#define FOR_VEC for (int i = 0; i < VEC_SIZE; ++i)
#define ALIGN alignas(VEC_SIZE * 4)

void mandelbrot_naive(unsigned char* pixels, float magnifier, float shiftX) 
{
    shiftX    += SHIFT_X_OFFSET;
    magnifier += MAGNIFIER_OFFSET;

    const float inv_magnifier = 1.0f / magnifier;
    const float COLOR_SCALE = 255.0f / MAX_ITERATION_DEPTH;

    const float pixel_step_x = ASPECT_RATIO * inv_magnifier * (2.0f / WINDOW_WIDTH);
    const float pixel_step_y = inv_magnifier                * (2.0f / WINDOW_HEIGHT);

    float c_y = -1.0f * inv_magnifier;

    for (int screen_y = 0; screen_y < WINDOW_HEIGHT; screen_y++,
            c_y += pixel_step_y) 
    {
        float c_x = shiftX - ASPECT_RATIO * inv_magnifier;

        for (int screen_x = 0; screen_x < WINDOW_WIDTH; screen_x++, 
                c_x += pixel_step_x) 
        {
            int iterations = 0;

            float z_x  = 0.0f;
            float z_y  = 0.0f;
            float z_x2 = 0.0f;
            float z_y2 = 0.0f;

            while (z_x2 + z_y2 < MAX_RADIUS_2 &&
                    iterations < MAX_ITERATION_DEPTH) 
            {
                z_y = 2 * z_x * z_y + c_y;
                z_x = z_x2 - z_y2 + c_x;

                z_x2 = z_x * z_x;
                z_y2 = z_y * z_y;

                ++iterations;
            }

            unsigned char r = 0;
            unsigned char g = 0;
            unsigned char b = 0;
            if (iterations < MAX_ITERATION_DEPTH)
            {
                float iterNormalized = iterations * COLOR_SCALE;
                r = (iterNormalized / 2);
                g = (iterNormalized * 2 + 2);
                b = (iterNormalized * 2 + 5);
            }

            int pixelIndex = (screen_y * WINDOW_WIDTH + screen_x) * 4;
            pixels[pixelIndex + 0] = r;
            pixels[pixelIndex + 1] = g;
            pixels[pixelIndex + 2] = b;
            pixels[pixelIndex + 3] = 255;
        }
    }
}

void mandelbrot_openmp(unsigned char* pixels, float magnifier, float shiftX)
{
    shiftX    += SHIFT_X_OFFSET;
    magnifier += MAGNIFIER_OFFSET;

    const float invMagnifier = 1.0f / magnifier;
    const float COLOR_SCALE = 255.0f / MAX_ITERATION_DEPTH;

    const float c_step_x = ASPECT_RATIO * invMagnifier * (2.0f / WINDOW_WIDTH);
    const float c_step_y = invMagnifier * (2.0f / WINDOW_HEIGHT);

    const float vec_c_step_x = c_step_x * VEC_SIZE;

#pragma omp parallel for schedule(guided, 1)
    for (int screenY = 0; screenY < WINDOW_HEIGHT; screenY++) 
    {
        float c_y = -1.0f * invMagnifier + c_step_y * screenY;

        ALIGN float _c_x[VEC_SIZE];

        const float c_x = shiftX - ASPECT_RATIO * invMagnifier;
        for (int i = 0; i < VEC_SIZE; ++i)
             _c_x[i] = c_x + c_step_x * i;

        for (int screenX = 0; screenX < WINDOW_WIDTH; screenX += VEC_SIZE) 
        {
            ALIGN float _z_x [VEC_SIZE] = {0};
            ALIGN float _z_y [VEC_SIZE] = {0};
            ALIGN float _z_xy[VEC_SIZE] = {0};
            ALIGN float _z_x2[VEC_SIZE] = {0};
            ALIGN float _z_y2[VEC_SIZE] = {0};
            ALIGN int _iterations[VEC_SIZE] = {0};

            for (int it = 0; it < MAX_ITERATION_DEPTH; ++it) 
            {
                ALIGN float _radius2[VEC_SIZE] = {0};
                for (int i = 0; i < VEC_SIZE; ++i)
                     _radius2[i] = _z_x2[i] + _z_y2[i];

                ALIGN int _is_active[VEC_SIZE] = {0};
                for (int i = 0; i < VEC_SIZE; ++i)
                     _is_active[i] = _radius2[i] < MAX_RADIUS_2;

                bool all_zero = true;
                for (int i = 0; i < VEC_SIZE; ++i)
                {
                    if (_is_active[i] != 0)
                    {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero)
                {
                    break;
                }

                for (int i = 0; i < VEC_SIZE; ++i)
                     _z_x[i] = _z_x2[i] - _z_y2[i] + _c_x[i];
                for (int i = 0; i < VEC_SIZE; ++i)
                    _z_y[i] = 2 * _z_xy[i]     + c_y;

                for (int i = 0; i < VEC_SIZE; ++i)
                     _iterations[i] += _is_active[i];

                for (int i = 0; i < VEC_SIZE; ++i)
                     _z_x2[i] = _z_x[i] * _z_x[i];
                for (int i = 0; i < VEC_SIZE; ++i)
                     _z_y2[i] = _z_y[i] * _z_y[i];
                for (int i = 0; i < VEC_SIZE; ++i)
                    _z_xy[i] = _z_x[i] * _z_y[i];
            }

            for (int i = 0; i < VEC_SIZE; ++i)
            {
                unsigned char r = 0;
                unsigned char g = 0;
                unsigned char b = 0;
                if (_iterations[i] < MAX_ITERATION_DEPTH)
                {
                    float iterNormalized = _iterations[i] * COLOR_SCALE;
                    r = (iterNormalized / 2);
                    g = (iterNormalized * 2 + 2);
                    b = (iterNormalized * 2 + 5);
                }

                int pixelIndex = (screenY * WINDOW_WIDTH + screenX + i) * 4;
                pixels[pixelIndex + 0] = r;
                pixels[pixelIndex + 1] = g;
                pixels[pixelIndex + 2] = b;
                pixels[pixelIndex + 3] = 255;
            }

            for (int i = 0; i < VEC_SIZE; ++i) 
                _c_x[i] += vec_c_step_x;
        }
    }
}


void mandelbrot_vectorized_ranged(unsigned char* pixels, float magnifier, float shiftX,
        int y_from, int y_to)
{
    shiftX    += SHIFT_X_OFFSET;
    magnifier += MAGNIFIER_OFFSET;

    const float invMagnifier = 1.0f / magnifier;
    const float COLOR_SCALE = 255.0f / MAX_ITERATION_DEPTH;

    const float c_step_x = ASPECT_RATIO * invMagnifier * (2.0f / WINDOW_WIDTH);
    const float c_step_y = invMagnifier * (2.0f / WINDOW_HEIGHT);

    __m256 _01234567 = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f,
            3.0f, 2.0f, 1.0f, 0.0f);
    __m256 _8_c_step_x = _mm256_set1_ps(c_step_x * 8);
    __m256   _c_step_x = _mm256_set1_ps(c_step_x);
    __m256   _c_step_y = _mm256_set1_ps(c_step_y);

    __m256 _maxRadius2 = _mm256_set1_ps(MAX_RADIUS_2);

    float c_y = -1.0f * invMagnifier + c_step_y * y_from;
    __m256 _c_y = _mm256_set1_ps(c_y);
    for (int screenY = y_from; screenY < y_to; ++screenY, c_y += c_step_y)
    {
        float c_x = shiftX - ASPECT_RATIO * invMagnifier;
        __m256 _c_x = _mm256_set1_ps(c_x);

        // _cx + ( 7dx, 6dx, 5dx, 4dx, 3dx, 2dx, 1dx, 0 )
        _c_x = _mm256_add_ps(_c_x, _mm256_mul_ps(_c_step_x, _01234567));
        for (int screenX = 0; screenX < WINDOW_WIDTH; screenX += 8)
        {
            __m256 _z_x = _mm256_setzero_ps();
            __m256 _z_y = _mm256_setzero_ps();

            __m256 _z_x2 = _mm256_setzero_ps();
            __m256 _z_y2 = _mm256_setzero_ps();
            __m256 _z_xy = _mm256_setzero_ps();

            __m256i _iterations = _mm256_setzero_si256();

            for (int iteration = 0; iteration < MAX_ITERATION_DEPTH; iteration++)
            {
                __m256 _radius2 = _mm256_add_ps(_z_x2, _z_y2);

                __m256 _cmpMask = _mm256_cmp_ps(_radius2, _maxRadius2, _CMP_LT_OQ);
                int mask = _mm256_movemask_ps(_cmpMask);
                // for each float:
                //      if (radius^2 < maxRadius^2)
                //          mask = -1
                //      else
                //          mask = 0

                if (!mask)
                    break;

                // x = x^2 - y^2 + cx
                // y = 2xy + cy
                _z_x = _mm256_add_ps(_c_x, _mm256_sub_ps(_z_x2, _z_y2));
                _z_y = _mm256_add_ps(_c_y, _mm256_mul_ps(_mm256_set1_ps(2.0f), _z_xy));

                // if (radius^2 < maxRadiud^2) iteration++
                _iterations = _mm256_sub_epi32(_iterations, _mm256_castps_si256(_cmpMask));

                _z_x2 = _mm256_mul_ps(_z_x, _z_x);
                _z_y2 = _mm256_mul_ps(_z_y, _z_y);
                _z_xy = _mm256_mul_ps(_z_x, _z_y);
            }

            int iterationsArray[8] = {};
            _mm256_storeu_si256((__m256i*)iterationsArray, _iterations);
            for (int i = 0; i < 8; i++)
            {
                unsigned char r = 0;
                unsigned char g = 0;
                unsigned char b = 0;
                if (iterationsArray[i] < MAX_ITERATION_DEPTH)
                {
                    float iterNormalized = iterationsArray[i] * COLOR_SCALE;
                    r = (iterNormalized / 2);
                    g = (iterNormalized * 2 + 2);
                    b = (iterNormalized * 2 + 5);
                }

                int pixelIndex = (screenY * WINDOW_WIDTH + screenX + i) * 4;
                pixels[pixelIndex + 0] = r;
                pixels[pixelIndex + 1] = g;
                pixels[pixelIndex + 2] = b;
                pixels[pixelIndex + 3] = 255;
            }

            _c_x = _mm256_add_ps(_c_x, _8_c_step_x);
        }
        _c_y = _mm256_add_ps(_c_y, _c_step_y);
    }
}


void mandelbrot_vectorized(unsigned char* pixels, float magnifier, float shiftX)
{
    mandelbrot_vectorized_ranged(pixels, magnifier, shiftX, 0, WINDOW_HEIGHT);
}


void mandelbrot_thread_pool(unsigned char* pixels, float magnifier, float shiftX)
{

    int threadNumbers = 8;
    boost::asio::thread_pool pool(threadNumbers);

    const int STEP = 8;

    for (int i = 0; i < WINDOW_HEIGHT; i += STEP)
    {

        boost::asio::post(pool, std::bind(mandelbrot_vectorized_ranged, pixels, magnifier, shiftX, i, i+STEP));

    }
    pool.join();


    return;
}
void writePng(const char* filename, const unsigned char* image, unsigned width, unsigned height)
{
    unsigned char* png;
    long unsigned int pngsize;
    int error = lodepng_encode32(&png, &pngsize, image, width, height);
    if(!error)
        lodepng_save_file(png, pngsize, filename);
    if(error) 
        printf("error %u: %s\n", error, lodepng_error_text(error));

}


int main() {
    size_t SIZE   = WINDOW_WIDTH*WINDOW_HEIGHT * 4;
    const char* output = "out.png";
    unsigned char* finish = (unsigned char*)malloc(SIZE*sizeof(unsigned char)); 

    if (finish == NULL) {
        return -1;
    }
    auto begin = chrono::steady_clock::now();

    mandelbrot_thread_pool(finish, 1.0, 0.0);
    auto end = chrono::steady_clock::now();
    auto elapsed_ms = chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "The time: " << elapsed_ms.count() << " ms\n";

    writePng(output, finish, WINDOW_WIDTH, WINDOW_HEIGHT); 
    free(finish);
    return 0;

}
