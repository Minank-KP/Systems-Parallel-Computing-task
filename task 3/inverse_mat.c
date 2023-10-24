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
            if(i==j)
                mat[i][j]=1;
            else
                mat[i][j] = 0;
        }
    }
}

void print(double **mat){
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            printf("%.4f\t",mat[i][j]);
        }
        printf("\n");
    }
}

void swapRows(double *row1,double *row2){
    for(int i=0;i<size;i++){
        double temp = row1[i];
        row1[i] = row2[i];
        row2[i] = temp;
    }
}

void inverse_matrix_serial(double** mat,double** inv)
{
    for (int i = 0; i < size; i++)
    {
        if (mat[i][i] == 0)
        {
            for (int j = i + 1; j < size; j++)
            {
                if (mat[j][i] != 0.0)
                {
                    swapRows(mat[i], mat[j]);       //swap with another row with mat[j][i] !=0
                    break;
                }
                if (j == size - 1)
                {
                    printf("Inverse does not exist for this matrix\n\n");
                    exit(0);
                }
            }
        }
        double scale = mat[i][i];
        for (int j = 0; j < size; j++)
        {
            mat[i][j] = mat[i][j] / scale;
            inv[i][j] = inv[i][j] / scale;
        }
        if (i < size - 1)
        {
            for (int row = i + 1; row < size; row++)
            {
                double factor = mat[row][i];
                for (int col = 0; col < size; col++)
                {
                    mat[row][col] -= factor * mat[i][col];
                    inv[row][col] -= factor * inv[i][col];
                }
            }
        }
    }
    for (int last = size - 1; last >= 1; last--)
    {
        for (int row = last - 1; row >= 0; row--)
        {
            double factor = mat[row][last];
            for (int col = 0; col < size; col++)
            {
                mat[row][col] -= factor * mat[last][col];
                inv[row][col] -= factor * inv[last][col];
            }
        }
    }
}

void inverse_matrix_parallel(double** mat, double** inv)
{
    for (int i = 0; i < size; i++)
    {
        if (mat[i][i] == 0)
        {
            for (int j = i + 1; j < size; j++)
            {
                if (mat[j][i] != 0.0)
                {
                    swapRows(mat[i], mat[j]);       //swap with another row with mat[j][i] !=0
                    break;
                }
                if (j == size - 1)
                {
                    printf("Inverse does not exist for this matrix\n\n");
                    exit(0);
                }
            }
        }
        double scale = mat[i][i];
        #pragma omp parallel for
        for (int col = 0; col < size; col++)
        {
            mat[i][col] = mat[i][col] / scale;
            inv[i][col] = inv[i][col] / scale;
        }
        if (i < size - 1)
        {
            #pragma omp parallel for
            for (int row = i + 1; row < size; row++)
            {
                double factor = mat[row][i];
                for (int col = 0; col < size; col++)
                {
                    mat[row][col] -= factor * mat[i][col];
                    inv[row][col] -= factor * inv[i][col];
                }
            }
        }
    }
    for (int last = size - 1; last >= 1; last--)
    {
        #pragma omp parallel for
        for (int row = last - 1; row >= 0; row--)
        {
            double factor = mat[row][last];
            for (int col = 0; col < size; col++)
            {
                mat[row][col] -= factor * mat[last][col];
                inv[row][col] -= factor * inv[last][col];
            }
        }
    }
}


int main(){

    double **mat,**inv;
    mat = (double **)malloc(size * sizeof(double *));
    inv = (double **)malloc(size * sizeof(double *));
    initialise(mat);
    initialise2(inv);
    // print(mat);
    // print(inv);

    double start_time1 = omp_get_wtime();
    inverse_matrix_serial(mat,inv);
    double end_time1 = omp_get_wtime();
    printf("Serial Execution Time: %f\n",(end_time1-start_time1));

    initialise2(inv);
    // print(mat);
    double start_time2 = omp_get_wtime();
    inverse_matrix_parallel(mat,inv);
    double end_time2 = omp_get_wtime();
    printf("Parallel Execution Time: %f\n",(end_time2-start_time2));
    // print(inv);

    return 0;
}