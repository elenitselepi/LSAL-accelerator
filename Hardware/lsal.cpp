#include <stdio.h>
#include <string.h>
#include "ap_int.h"

#define N 128
#define M 65536

#define GAP_k -1
#define GAP_l -1
#define MATCH  2
#define MISS_MATCH  -1
#define CENTER 0
#define NORTH 1
#define NORTH_WEST 2
#define WEST 3

/**********************************************************************************************
 * LSAL algorithm
 * Inputs:
 *          string1 is the query[N]
 *          string2 is the database[M]
 *          input sizes N, M
 * Outputs:
 *           max_index is the location of the highest similiarity score
 *           similarity and direction matrices. Note that these two matrices are initialized with zeros.
 **********************************************************************************************/

void compute_matrices(ap_uint<32> *string1, ap_uint<32> *string2, ap_int<16> *max_index, short *direction_matrix)
{
    int i,j,k;

    // Following values are used for the N, W, and NW values wrt. similarity_matrix[i]
    ap_int<16> north = 0;
    ap_int<16> west = 0;
    ap_int<16> northwest = 0;
    ap_int<16> max_value = 0;

    int match = 0;
    int test_val;
    int dir;

    ap_int<512> similarity_buffer[6];

    #pragma HLS ARRAY_PARTITION variable=similarity_buffer dim=1 complete

    short direction_buffer[N];
    ap_uint<32> query_buffer [N/8+1];
    ap_uint<32> database_buffer [N/8+1];
	#pragma HLS ARRAY_PARTITION variable=database_buffer dim=1 complete
    ap_uint<32> database[(M+2*(N-1)) /8 +1];

    ap_int<16> max_index_buffer[2];
    ap_int<16> max_value_buffer[2*N];
	#pragma HLS ARRAY_PARTITION variable=max_value_buffer dim=1 complete

    memcpy(query_buffer,string1,(N/8+1)*sizeof(ap_uint<32>));
    memcpy(database,string2,((M+2*(N-1))/8+1)*sizeof(ap_uint<32>));
    memcpy(database_buffer,database,(N/8 +1)*sizeof(ap_uint<32>));

    memset(max_value_buffer, 0, 2*N*sizeof(ap_int<16>));
    memset(max_index_buffer,0,2*sizeof(ap_int<16>));

    memset(similarity_buffer, 0, 6*(512/8));


    DB:for(i = N-1; i < (M + 2*(N-1)) ; i++)
    {
		#pragma HLS PIPELINE II=38

        column:for(j = 0 ; j < N; j++)
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
                northwest = similarity_buffer[(j-1)/64].range(((j-1)%64)*8+7,((j-1)%64)*8); // from ((i+1)%3)*N it became 0, stable in the first row
                west = similarity_buffer[(N+j-1)/64].range(((N+j-1)%64)*8+7,((N+j-1)%64)*8) + GAP_k; // from ((i+2)%3)*N it became N, stable in the second row
            }

            north = similarity_buffer[(j+N)/64].range(((j+N)%64)*8+7,((j+N)%64)*8) + GAP_l; // from ((i+2)%3)*N it became N, stable in the second row

            match = ( query_buffer[j/8].range((j%8)*4+3,(j%8)*4) == database_buffer[(N- 1- j)/8].range(((N- 1- j)%8)*4+3,((N- 1- j)%8)*4) ) ? MATCH : MISS_MATCH;

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

            similarity_buffer[(2*N+j)/64].range(((2*N+j)%64)*8+7,((2*N+j)%64)*8) = max_value ; // stable in the last row

            if(max_value > max_value_buffer[j]) //0*N+j
			{

				max_value_buffer[j] = max_value;
				max_value_buffer[1*N+j] = (i - j - (N-1))* N + j; //i*N+j
			}

            direction_buffer[j] = dir;
        }

        for(k=0; k<N-1 ; k++)
		{
        	database_buffer[k/8].range(((k%8)*4+3),(k%8)*4) = database_buffer[(k+1)/8].range((((k+1)%8)*4+3),((k+1)%8)*4);
		}

        database_buffer[k/8].range(((k%8)*4+3),(k%8)*4) = database[(1+i)/8].range((((i+1)%8)*4+3),((i+1)%8)*4);

        memcpy(&direction_matrix[i*N],direction_buffer, N*sizeof(short));

        for(k=0; k<N ; k++)
        {
            similarity_buffer[(k)/64].range(((k)%64)*8+7,((k)%64)*8) =similarity_buffer[(N+k)/64].range(((N+k)%64)*8+7,((N+k)%64)*8);
            similarity_buffer[(N+k)/64].range(((N+k)%64)*8+7,((N+k)%64)*8) =similarity_buffer[(2*N+k)/64].range(((2*N+k)%64)*8+7,((2*N+k)%64)*8);
        }
    }

    max_index_buffer[0] = max_value_buffer[0];
	for(i=1;i<N;i++)
	{
		if(max_value_buffer[i] > max_index_buffer[0])
		{
			max_index_buffer[0] = max_value_buffer[i]; //value
			max_index_buffer[1] = max_value_buffer[N+i]; //index
		}
	}

	memcpy(max_index,max_index_buffer,2*sizeof(ap_int<16>));

}  // end of function



