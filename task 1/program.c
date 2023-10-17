#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>

#define N 1000000000

int main(){
	long long * array = malloc(N * sizeof(long long));
    for(long long i=0;i<N;i++){
        array[i] =i;
    }

    
    long long sum1 = 0 ;

    double start_time1 = omp_get_wtime();
    for(long long i=0;i<N;i++)
	{
		sum1 = sum1+array[i];
	}
	double end_time1 = omp_get_wtime();	
	printf("Serial execution Sum of the array: %llu\n",sum1);
	printf("Serial Execution Time: %f\n",(end_time1-start_time1));

    long long sum2 = 0 ;
	double start_time2 = omp_get_wtime();

	#pragma omp parallel for reduction(+:sum2)
		for(long long j=0;j<N;j++)
		{
			sum2 = sum2+array[j];
		}

	double end_time2 = omp_get_wtime();	
	printf("Parallel Reduction Sum of the array: %llu\n",sum2);
	printf("Parallel Reduction Execution Time: %f\n",(end_time2-start_time2));


	
	return(0);
}