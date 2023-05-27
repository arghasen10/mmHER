#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>

#define FRAME_SIZE 12 * 128 * 256 * 4
#define BIN_FILENAME "1684598876.bin"
#define TOTAL_FRAME_NUMBER 799

typedef struct {
    float real;
    float imag;
} Complex;

int16_t* sliceArray(const int16_t* arr, int length, int start, int step, int* result_length) {
    int sliced_length = (length - start + step - 1) / step;
    int16_t* sliced_arr = (int16_t*)malloc(sizeof(int16_t) * sliced_length);

    int j = 0;
    for (int i = start; i < length; i += step) {
        sliced_arr[j] = arr[i];
        j++;
    }

    *result_length = sliced_length;
    return sliced_arr;
}

int16_t* complexFrame(int16_t* bin_frame){
    int bin_frame_length = FRAME_SIZE / sizeof(int16_t);
    int sliced_length;
    int16_t* A1 = sliceArray(bin_frame, bin_frame_length, 0, 4, &sliced_length);
    int16_t* A2 = sliceArray(bin_frame, bin_frame_length, 2, 4, &sliced_length);
    int16_t* A3 = sliceArray(bin_frame, bin_frame_length, 1, 4, &sliced_length);
    int16_t* A4 = sliceArray(bin_frame, bin_frame_length, 3, 4, &sliced_length);

    Complex* np_frame = (Complex*)malloc(sizeof(Complex) * (bin_frame_length / 4));
    int j = 0 ;
    for (int i = 0; i < bin_frame_length / 4; i++, j+=2) {
        np_frame[j].real = (float)A1[i];
        np_frame[j].imag = (float)A2[i];
        np_frame[j+1].real = (float)A3[(i)];
        np_frame[j+1].imag = (float)A4[(i)];
    }
    return np_frame;
}

Complex**** reshapeAndTranspose(const Complex* np_frame, int np_frame_length, int* shape) {
    int dim1 = 128;
    int dim2 = 3;
    int dim3 = 4;
    int dim0 = np_frame_length / (dim1 * dim2 * dim3);

    Complex**** frameWithChirp = (Complex****)malloc(sizeof(Complex***) * dim1);

    int i, j, k, l;
    for (i = 0; i < dim1; i++) {
        frameWithChirp[i] = (Complex***)malloc(sizeof(Complex**) * dim2);
        for (j = 0; j < dim2; j++) {
            frameWithChirp[i][j] = (Complex**)malloc(sizeof(Complex*) * dim3);
            for (k = 0; k < dim3; k++) {
                frameWithChirp[i][j][k] = (Complex*)malloc(sizeof(Complex) * dim0);
                for (l = 0; l < dim0; l++) {
                    int index = i * dim2 * dim3 * dim0 + j * dim3 * dim0 + k * dim0 + l;
                    frameWithChirp[i][j][k][l] = np_frame[index];
                }
            }
        }
    }
    shape[0] = dim1;
    shape[1] = dim2;
    shape[2] = dim3;
    shape[3] = dim0;
    Complex**** transposedFrame = (Complex****)malloc(sizeof(Complex***) * dim2);
    for (int i = 0; i < dim2; i++) {
        transposedFrame[i] = (Complex***)malloc(sizeof(Complex**) * dim3);
        for (int j = 0; j < dim3; j++) {
            transposedFrame[i][j] = (Complex**)malloc(sizeof(Complex*) * dim1);
            for (int k = 0; k < dim1; k++) {
                transposedFrame[i][j][k] = (Complex*)malloc(sizeof(Complex) * dim0);
                for (int l = 0; l < dim0; l++) {
                    transposedFrame[i][j][k][l].real = frameWithChirp[k][i][j][l].real;
                    transposedFrame[i][j][k][l].imag = frameWithChirp[k][i][j][l].imag;
                }
            }
        }
    }
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                free(frameWithChirp[i][j][k]);
            }
            free(frameWithChirp[i][j]);
        }
        free(frameWithChirp[i]);
    }
    free(frameWithChirp);

    shape[0] = dim2;
    shape[1] = dim3;
    shape[2] = dim1;
    shape[3] = dim0;
    return transposedFrame;
}



int main() {
FILE* ADCBinFile;
    int16_t* bin_frame;
    Complex* np_frame;
    int frame_no;

    ADCBinFile = fopen(BIN_FILENAME, "rb");
    if (ADCBinFile == NULL) {
        printf("Failed to open file.\n");
        return 1;
    }
    for (frame_no = 0; frame_no < TOTAL_FRAME_NUMBER; frame_no++) {
        bin_frame = (int16_t*)malloc(sizeof(int16_t) * (FRAME_SIZE / sizeof(int16_t)));
        fread(bin_frame, sizeof(int16_t), FRAME_SIZE / sizeof(int16_t), ADCBinFile);
        Complex* np_frame = complexFrame(bin_frame);
        printf("Size of np_frame %d\n", sizeof(np_frame));
        
        int shape[4];
        int np_frame_length = 128 * 3 * 4 * 256;  // Adjust the length according to your data

        // Reshape and transpose np_frame
        
        Complex**** frameWithChirp = reshapeAndTranspose(np_frame, np_frame_length, shape);
        printf("frameWithChirp shape: %d, %d, %d, %d\n", shape[0], shape[1], shape[2], shape[3]);

        free(frameWithChirp);
        free(bin_frame);
        free(np_frame);
        break;
    }

    fclose(ADCBinFile);
    
    return 0;
}
