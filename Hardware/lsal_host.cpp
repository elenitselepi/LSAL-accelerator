/********************************************************************************
 * The Host code running in the Arm CPU. Make sure to study and understand the OpenCL API code
 *
 ********************************************************************************/

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include "ap_int.h"

#define N 128
#define M 65536

const short GAP_k = -1;
const short GAP_l = -1;
const short MATCH = 2;
const short MISS_MATCH = -1;
const short CENTER = 0;
const short NORTH = 1;
const short NORTH_WEST = 2;
const short WEST = 3;

 /***************************************************************************************
  * This is the golden code which runs in the CPU (and is the same code that you developed for x86 / Arm)
  * It will be used to verify the correct functionality of the HW implementation.
  * Its usefulness is mainly when you perform software emulation (sw_emu).
  ***************************************************************************************/

void compute_matrices_sw(char *string1, char *string2, int *max_index, int *similarity_matrix, short *direction_matrix)
{
    int i,j;

    // Following values are used for the N, W, and NW values wrt. similarity_matrix[i]
    int north = 0;
    int west = 0;
    int northwest = 0;
    int max_value = 0;
    int match = 0; // s(di,qj)
    int test_val;
    int dir;//direction


    //Here the real computation starts. Place your code whenever is required.
    // Scan the N*M array row-wise starting from the second row.
row:for(i = 1; i < M+1; i++) //+1 padding
{
        column:for(j = 0; j < N; j++)
        {
            max_value = 0;
            if (j == 0)
            {
                // first column.
                west = 0;
                northwest = 0;
            }
            else
            {
                northwest = similarity_matrix[(i-1)*N +(j-1)];
                west = similarity_matrix[i * N +(j-1)] + GAP_k;
            }

            north = similarity_matrix[(i-1)*N + j] + GAP_l;
            match = ( string1[j] == string2[i-1] ) ? MATCH : MISS_MATCH;
            test_val = northwest + match;
            if(test_val > max_value)
            {
                max_value = test_val;
                dir = NORTH_WEST;
            }
            if(north > max_value)
            {
                max_value = north;
                dir = NORTH;
            }
            if(west > max_value)
            {
                max_value = west;
                dir = WEST;
            }
            if(max_value <= 0)
            {
                max_value = 0;
                dir = CENTER;
            }

            similarity_matrix[i*N+j] = max_value ;
            if(max_value > max_index[2])
            {
                max_index[0] = i-1;
                max_index[1] = j;
                max_index[2] = max_value;
            }
            direction_matrix[(i-1)*N+j] = dir;
        }
    }
}

/*
Given an event, this function returns the kernel execution time in ms
*/
double getTimeDifference(cl_event event) {
    cl_ulong time_start = 0;
    cl_ulong time_end = 0;
    double total_time = 0.0f;

    clGetEventProfilingInfo(event,
    CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start,
    NULL);
    clGetEventProfilingInfo(event,
    CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end,
    NULL);
    total_time = time_end - time_start;
    return total_time / 1000000.0; // To convert nanoseconds to milliseconds
}

void compress(ap_uint<32> * array, int index, char val)
{
	switch(val)
	{
		case 'A': array[index / 8] |= (0 << ((index % 8) * 4));
			break;
		case 'C': array[index / 8] |= (1 << ((index % 8) * 4));
			break;
		case 'G': array[index / 8] |= (3 << ((index % 8) * 4));
			break;
		case 'T': array[index / 8] |= (2 << ((index % 8) * 4));
			break;
		case 'P': array[index / 8] |= (4 << ((index % 8) * 4));
			break;
	}
}

short* order_matrix_blocks(short* in_matrix)
{
    short *out_matrix = (short *) malloc(sizeof(short) * M *N);

    int num_diag = 0;
    int store_elem = 1;
    int store_index;
    int tmp_i = 0;
    int tmp_j = 0;
    int i = 0;
    int j;

    for (i = 0; i < N*(N + M - 1);){  // DIRECTION_MATRIX_SIZE = N * (N + M - 1)
        if(num_diag < M) {
            tmp_j = num_diag;
            tmp_i = 0;
        } else {
            tmp_i = (num_diag + 1) - M;
            tmp_j = M - 1;
        }

        for (j = 0; j < store_elem; j++) {
            store_index = tmp_j * N + tmp_i;
            out_matrix[store_index] = in_matrix[i];
            tmp_j--;
            tmp_i++;
            i++;
        }
        num_diag++;
        if(num_diag >= M){
            store_elem--;
            i = num_diag * N + N - store_elem;
        } else {
            if(store_elem != N){
                store_elem++;
                i = num_diag * N;
            }
        }
    }

    return out_matrix;
}


/*
return a random number between 0 and limit inclusive.
*/
int rand_lim(int limit) {

    int divisor = RAND_MAX / (limit + 1);
    int retval;

    do {
        retval = rand() / divisor;
    } while (retval > limit);

    return retval;
}

/*
Fill the string with random values
*/
void fillRandom(char* string, int dimension) {
    //fill the string with random letters..
    static const char possibleLetters[] = "ATCG";

    string[0] = '-';

    int i;
    for (i = 0; i < dimension; i++) {
        int randomNum = rand_lim(3);
        string[i] = possibleLetters[randomNum];
    }
}

/*
Fill the string with random values
*/
void fillRandomdatabase(char* src,char* string, int dimension ) {
    //fill the string with random letters..
    static const char possibleLetters[] = "ATCG";
    int i;

    for(i = 0; i < N-1 ; i++)
    {
        string[i] = 'P';
    }

    strncpy(&string[N-1],src,strlen(src));

    for(i = M +(N-1); i < M+2*(N-1) ; i++)
    {
        string[i] = 'P';
    }

}

int load_file_to_memory(const char *filename, char **result) {
    size_t size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        *result = NULL;
        return -1; // -1 means file opening fail
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *) malloc(size + 1);
    if (size != fread(*result, sizeof(char), size, f)) {
        free(*result);
        return -2; // -2 means file reading fail
    }
    fclose(f);
    (*result)[size] = 0;
    return size;
}

/*******************************************************************************
*   Host program running on the Arm CPU.
*   The code is written using the OpenCL API.
*   We have provided multiple comments for you to understand where each thing
*******************************************************************************/
int main(int argc, char** argv) {
    printf("starting HOST code \n");
    fflush(stdout);
    int err;                            // error code returned from api calls
    cl_uint matrix_size;


    if (argc != 2) {
        printf("%s <input xclbin file>\n", argv[0]);
        return EXIT_FAILURE;
    }


    matrix_size = N*M;

    char *query = (char*) malloc(sizeof(char) * N);
    char *database = (char*) malloc(sizeof(char) * (M+2*(N-1)));

    char *database_sw = (char*) malloc(sizeof(char) * M);
    short *direction_matrix = (short*) malloc(sizeof(short) * (M + 2*(N-1)) * N);
    short *direction_matrix_ordered ;
    ap_int<16> *max_index = (ap_int<16> *) malloc(2 * sizeof(ap_int<16>) ); //0 is the index 1 is the value

    ap_uint<32> *query_compressed = (ap_uint<32> *) malloc(sizeof(ap_uint<32>) * (N/8 +1));
	ap_uint<32> *database_compressed = (ap_uint<32> *) malloc(sizeof(ap_uint<32>) * ((M+2*(N-1))/8 +1));


    printf("array defined! \n");
    fflush(stdout);

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    cl_mem input_query;
    cl_mem input_database;
    cl_mem output_similarity_matrix;
    cl_mem output_direction_matrix;
    cl_mem output_max_index;

    fillRandom(query, N);
    fillRandom(database_sw, M);
    fillRandomdatabase(database_sw,database, M);

    cl_uint i,j;


    for(i = 0; i < N; i++)
	{
		compress(query_compressed, i, query[i]);
	}

	for(i = 0; i < (M+2*(N-1)); i++)
	{
		compress(database_compressed, i, database[i]);
	}


    memset(direction_matrix, 0, sizeof(short) * N * (M + 2*(N-1)));

/**********************************************
* 			Xilinx OpenCL Initialization
*
* We must follow specific steps to get the necessary
* information and handlers, in order to be able
* to use the available accelerator device (FPGA).
* After every step, we always check for any errors
* that might have occured. In case of error, the
* program aborts and exits immediately.
* *********************************************/

/**************************************************
    * Step 1:
* Get available OpenCL platforms and devices.
* In our case, it is a Xilinx FPGA device.
* If the underlying platform has other accelerators
* available, we could use them too (e.g. GPU, CPU).
    **************************************************/
    printf("GET platform \n");
    err = clGetPlatformIDs(1, &platform_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("GET platform vendor \n");
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 1000,
            (void *) cl_platform_vendor, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_VENDOR %s\n", cl_platform_vendor);
    printf("GET platform name \n");
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 1000,
            (void *) cl_platform_name, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_NAME %s\n", cl_platform_name);

    // Connect to a compute device
    //
    int fpga = 1;

    printf("get device FPGA is %d  \n", fpga);
    err = clGetDeviceIDs(platform_id,
            fpga ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU, 1,
            &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    /*********************************************
    * Step 2 : Create Context
    *********************************************/
    printf("create context \n");
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

/*********************************************
    * Step 3 : Create Command Queue
    *********************************************/
    printf("create queue \n");
    commands = clCreateCommandQueue(context, device_id,
    CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        printf("Error: code %i\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    cl_int binary_status;

    /*********************************************
    * Step 4 : Load Hardware Binary File (*.xclbin) from disk
    **********************************************/
    unsigned char *kernelbinary;
    char *xclbin = argv[1];
    printf("loading %s\n", xclbin);
    int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
    if (n_i < 0) {
        printf("failed to load kernel from xclbin: %s\n", xclbin);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

/********************************************************
    * Step 5 : Create program using the loaded hardware binary file
    ********************************************************/
    size_t n = n_i;
    // Create the compute program from offline
    printf("create program with binary \n");
    program = clCreateProgramWithBinary(context, 1, &device_id, &n,
            (const unsigned char **) &kernelbinary, &binary_status, &err);
    free(kernelbinary);

    if ((!program) || (err != CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n",
                err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    /**************************************************************
    *  Step 6 for 1 Compute Unit: Create Kernels - the actual handler of the kernel that
    *           we will be using. We first create a program, and then
    *           obtain the kernel handler from the program.
    **************************************************************/
    printf("build program \n");
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    //
    printf("create kernel \n");
    kernel = clCreateKernel(program, "compute_matrices", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    /**************************************************************-
    * Step 7 : Create buffers.
    * We do not need to allocate separate memory space (malloc), because
    * on a MPSoC system (e.g. ZCU102 board), we map the memory space that is
    * allocated at clCreateBuffer, to a usable memory space for our
    * host application. We also do not need to use free for any reason.
    * See Xilinx UG1393 for detailed information.
    **************************************************************/
    input_query = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(ap_uint<32>) * N,
    NULL, NULL);
    input_database = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(ap_uint<32>) * (M+2*(N-1)),
    NULL, NULL);
    output_direction_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(short) * (M + 2*(N-1)) * N, NULL, NULL);
    output_max_index = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(ap_int<16>),
    NULL, NULL);


/**************************************************************
    * Step 8 : Write the Input Data to the Write Buffers of the device memory
    **************************************************************/
    err = clEnqueueWriteBuffer(commands, input_query, CL_TRUE, 0,
            sizeof(ap_uint<32>) * N, query_compressed, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array a!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    err = clEnqueueWriteBuffer(commands, input_database, CL_TRUE, 0,
            sizeof(ap_uint<32>) * (M+2*(N-1)), database_compressed, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array a!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }


    err = clEnqueueWriteBuffer(commands, output_direction_matrix, CL_TRUE, 0,
                sizeof(short) * (M + 2*(N-1)) * N, direction_matrix, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to write to source array a!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    /**************************************************************
    * Step 9: Set the arguments to our compute kernel
    **************************************************************/
    err = 0;
    printf("set arg 0 \n");
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_query);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 0! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("set arg 1 \n");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_database);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 1! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("set arg 2 \n");
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_max_index);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 2! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    printf("set arg 3 \n");
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem),
            &output_direction_matrix);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }



    /**************************************************************
    * Step 10: Place the Kernel in the Queue for Execution
    **************************************************************/
    cl_event enqueue_kernel;
    printf("LAUNCH task \n");
    err = clEnqueueTask(commands, kernel, 0, NULL, &enqueue_kernel);

    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, &enqueue_kernel);

    /**************************************************************
    * Step 11: Read back the results from the device to verify the output
    **************************************************************/
    cl_event readMax, readSimilarity, readDirections;

    err = clEnqueueReadBuffer(commands, output_direction_matrix, CL_TRUE, 0,
            sizeof(short) * (M + 2*(N-1)) * N, direction_matrix, 0, NULL,
            &readDirections);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read array! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    err = clEnqueueReadBuffer(commands, output_max_index, CL_TRUE, 0,
            2*sizeof(ap_int<16>), max_index, 0, NULL, &readMax);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read array! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }


    clWaitForEvents(1, &readDirections);
    clWaitForEvents(1, &readMax);

    double executionTime = getTimeDifference(enqueue_kernel);

clReleaseMemObject(input_database);
    clReleaseMemObject(input_query);
    clReleaseMemObject(output_direction_matrix);
    clReleaseMemObject(output_max_index);
    //clReleaseMemObject(output_similarity_matrix);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

/**************************************************************
* Run the same algorithm in the Host Unit and compare for verification
**************************************************************/
    int * similarity_matrix_sw = ( int *) malloc(
            sizeof( int) * (M+1)*N); //pading
    short * direction_matrix_sw = ( short*) malloc(
            sizeof( short) * matrix_size);
    int * max_index_sw = ( int *) malloc(
                sizeof( int)*3);


    for(i = 0; i < (M+1)*N; i++){
        similarity_matrix_sw[i] = 0;
    }
    compute_matrices_sw(query, database_sw, max_index_sw, similarity_matrix_sw, direction_matrix_sw);

    direction_matrix_ordered = order_matrix_blocks(direction_matrix+((N-1)*N));


    printf("N: %d, M: %d\n",N,M);

    printf("both ended\n");

    printf(" execution time is %lf ms \n", executionTime);
    for (i = 0; i < matrix_size; i++) {
        if (direction_matrix_sw[i] != direction_matrix_ordered[i]) {
            printf("Error, mismatch in the results, i + %d, SW: %d, HW %d \n",
                    i, direction_matrix_sw[i], direction_matrix_ordered[i]);
            return EXIT_FAILURE;
        }
    }

    printf("computation ended!- RESULTS CORRECT \n");

    /**************************************************************
    * Clean up everything and, then, shutdown
    **************************************************************/

    free(query);
    free(database);
    free(database_sw);
    free(direction_matrix);
    free(max_index);
    free(direction_matrix_ordered);
    free(similarity_matrix_sw);
    free(direction_matrix_sw);
    free(max_index_sw);

    return EXIT_SUCCESS;
}


