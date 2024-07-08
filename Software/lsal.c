#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>


const short GAP_k = -1;
const short GAP_l = -1;
const short MATCH = 2;
const short MISS_MATCH = -1;
const short CENTER = 0;
const short NORTH = 1;
const short NORTH_WEST = 2;
const short WEST = 3;
// static long int cnt_ops=0;
// static long int cnt_bytes=0;

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

void compute_matrices(char *string1, char *string2, int *max_index, int **similarity_matrix, short **direction_matrix, int N , int M)
{
    int i;
	int j;

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
   for(i = 1; i < M; i++) 
   {
        for(j = 0; j < N; j++) 
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
				northwest = similarity_matrix[i-1][j-1]; 
				west = similarity_matrix[i][j-1] + GAP_k;
			}

			north = similarity_matrix[i-1][j] + GAP_l; 
			match = ( string1[j] == string2[i] ) ? MATCH : MISS_MATCH; 
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

			similarity_matrix[i][j] = max_value ;
			if(max_value > max_index[2])
			{
				max_index[0] = i;
				max_index[1] = j;
				max_index[2] = max_value;
			}
			direction_matrix[i][j] = dir;
		}
	}   

}  // end of function


/*
 return a random number in [0, limit].
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
	string[dimension] = '\0';

}

/* ******************************************************************/
int main(int argc, char** argv) {

    clock_t t1, t2;

	if (argc != 3) {
		printf("%s <Query Size N> <DataBase Size M>\n", argv[0]);
		return EXIT_FAILURE;
	}

    printf("Starting Local Alignment Code \n");
	fflush(stdout);

	/* Typically, M >> N */
	int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    char *query = (char*) malloc(sizeof(char) * N + 1 );
	char *database = (char*) malloc(sizeof(char) * M + 1);
	int **similarity_matrix = (int**) malloc(sizeof(int*) * M);
	short **direction_matrix = (short**) malloc(sizeof(short*) * M);
	int *max_index = (int *) malloc(3 * sizeof(int)); //0 is the index 1 is the value
	max_index[0] = 0;

	int i;
	for(i = 0; i < M; i++)
	{
		similarity_matrix[i] = (int*) malloc(sizeof(int) * N);
		direction_matrix[i] = (short*) malloc(sizeof(short) * N);
	}


/* Create the two input strings by calling a random number generator */
	fillRandom(query, N);
	fillRandom(database, M);

	for(int i = 0; i < M; i++)
	{
		for(int j=0; j < N; j++)
		{
			similarity_matrix[i][j] = 0;
			direction_matrix[i][j] = 0;
		}
	}


    t1 = clock();
	compute_matrices(query, database, max_index, similarity_matrix, direction_matrix,N,M);
	t2 = clock();

    printf(" max index is in position (%d, %d) = %d \n", max_index[0], max_index[1], max_index[2] );
	printf(" execution time of LSAL SW algorithm is %f sec \n", (double)(t2-t1) / CLOCKS_PER_SEC);

	
	for(i = 0; i < M; i++)
	{
		free(similarity_matrix[i]);
		free(direction_matrix[i]);
	}
	free(similarity_matrix);
	free(direction_matrix);
	free(max_index);
    free(query);
    free(database);

	return EXIT_SUCCESS;
}
