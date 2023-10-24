#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include<time.h>

#define size 2000

void initialise(double **mat){
    srand(time(NULL));
    for(int i=0;i<size;i++){
        mat[i] = (double *)malloc(size * sizeof(double));
        for(int j=0;j<size;j++){
            mat[i][j] = rand()%100+1;
        }
    }
}

void initialise2(double **mat){
    for(int i=0;i<size;i++){
        mat[i] = (double *)malloc(size * sizeof(double));
        for(int j=0;j<size;j++){
            mat[i][j] = 0;
        }
    }
}

void print(double **mat){
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            printf("%.2f\t",mat[i][j]);
        }
        printf("\n");
    }
}


void lu_decompostion_ser(double **mat, double **l, double **u)
{
	for(int i=0;i< size;i++){
        // u Triangular
        for (int j = i; j < size; j++)
        {
            // Summation of L(i, j) * U(j, k)
            double sum = 0;
            for (int k = 0; k < i; k++)
                sum += (l[i][k] * u[k][j]);
 
            // Evaluating U(i, k)
            u[i][j] = mat[i][j] - sum;
        }
 
        // l Triangular
        for (int j = i; j < size; j++) 
        {
            if (i == j)
                l[i][i] = 1; // Diagonal as 1
            else
            {
                // Summation of L(k, j) * U(j, i)
                int sum = 0;
                for (int k = 0; k < i; k++)
                    sum += (l[j][k] * u[k][i]);
 
                // Evaluating L(k, i)
                l[j][i] = (mat[j][i] - sum) / u[i][i];
            }
        }
    }
}


void lu_decomposition_par(double** A, double** L, double** U)
{
	#pragma omp parallel
    {
        for (int i = 0; i < size; i++) {
            #pragma omp for
            for (int k = i; k < size; k++) {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += (L[i][j] * U[j][k]);
                }
                U[i][k] = A[i][k] - sum;
            }

            #pragma omp for
            for (int k = i; k < size; k++) {
                if (i == k) {
                    L[i][i] = 1.0;
                } else {
                    double sum = 0.0;
                    for (int j = 0; j < i; j++) {
                        sum += (L[k][j] * U[j][i]);
                    }
                    L[k][i] = (A[k][i] - sum) / U[i][i];
                }
            }
        }
    }
}

int main(){

    double **mat,**l,**u;
    mat = (double **)malloc(size * sizeof(double *));
    l = (double **)malloc(size * sizeof(double *));
    u = (double **)malloc(size * sizeof(double *));
    // double mat[size][size];
    // double l[size][size]={0};
    // double u[size][size]={0};

    initialise(mat);
    initialise2(l);
    initialise2(u);
    // print(mat);
    double start_time1 = omp_get_wtime();
    lu_decompostion_ser(mat,l,u);
    double end_time1 = omp_get_wtime();
    printf("Serial Execution Time: %f\n",(end_time1-start_time1));

    initialise2(l);
    initialise2(u);
    // print(mat);
    double start_time2 = omp_get_wtime();
    lu_decomposition_par(mat,l,u);
    double end_time2 = omp_get_wtime();
    printf("Parallel Execution Time: %f\n",(end_time2-start_time2));
    // printf("\n\n");
    // print(l);
    // printf("\n\n");
    // print(u);

    return 0;


}