#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define size 15
#define mod 3
#define add_size 3
#define threads 16

typedef struct {
    uint32_t vector_pat[size];

    uint32_t *vector_index;
    uint32_t size_vector_index;
    uint32_t capacity_vector_index;
} pat;




static inline void check_vectors(int start, int depth, uint32_t selected[add_size],uint32_t pat[size]);
static inline void index_to_vector(uint64_t index, uint32_t vector[size], int num_values);
static inline void patt(uint32_t in[size],uint32_t out[size] );
static inline int check_vector_in_list(uint32_t a[size], pat all_pats[], uint32_t size_vectors_pats);
static inline int compare_vectors(uint32_t a[size], uint32_t b[size]);
static inline void pat_init(pat *p);
static inline void pat_push(pat *p, uint32_t value);
static inline void pat_destroy(pat *p);

static inline void print_vector(unsigned int vector[]);
static inline bool add_size_vector_2(uint32_t add[add_size][size] );
static inline bool add_size_vector(uint32_t add[add_size][size] );
static inline bool check_checksum_vector(uint32_t in[size] );
int total_vectors = 1;
int total_vectors_serach = 0;
static inline void check_pat(int start, int depth, uint32_t selected[add_size], pat all_pats);


// Definición de la estructura
int all_pats_size = 0;


pat all_pats[100000];

int main(int argc, char const *argv[])
{

    //prueba de que existe en una longitud de 19
    if (size>18){
        uint32_t add[add_size][size];
        uint32_t temp_1[] = {0, 0, 0, 1, 1, 2, 2, 1, 1, 0, 1, 2, 0, 2, 0, 2, 1, 2, 0};
        uint32_t temp_2[] = {1, 1, 2, 2, 1, 1, 1, 2, 0, 0, 2, 2, 1, 0, 2, 0, 0, 0, 0};
        uint32_t temp_3[] = {0, 0, 1, 0, 1, 0, 2, 2, 0, 1, 1, 1, 2, 0, 0, 2, 2, 1, 2};
        for (size_t i = 0; i < 19; i++){
            add[0][i] = temp_1[i];
            add[1][i] = temp_2[i];
            add[2][i] = temp_3[i];
        }
        if (add_size_vector(add)){
            add_size_vector_2(add);


        }
    }
    





    for (int i = 0; i < size; i++) {
        total_vectors *= mod;
    }

    // for (int i = 0; i < total_vectors; i++) {
    //     index_to_vector(i, vector,  mod);
    //     print_vector(vector);
    // }
   
    // printf("-------------------------------\n");

    printf("Buscando  size %i mod %i add_size %i \n", size,mod,add_size);
    
    printf("total_vectors %i \n", total_vectors);

    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (int i = 1; i <= total_vectors; i++) {
        uint32_t vector_temp[size];
        index_to_vector(i, vector_temp, mod);   

        if(check_checksum_vector(vector_temp)){
            uint32_t pat_temp[size];
            patt(vector_temp,pat_temp);
            
            #pragma omp critical
            {
                if(check_vector_in_list(pat_temp,all_pats,all_pats_size) == (-1) ){
                    pat_init(&all_pats[all_pats_size]);
                    for (size_t j = 0; j < size; j++){
                        all_pats[all_pats_size].vector_pat[j]=pat_temp[j];
                        
                    }
                    pat_push(&all_pats[all_pats_size], i);
                    all_pats_size++;
                }
                else if (check_vector_in_list(pat_temp,all_pats,all_pats_size) == (-2)) {

                    
                }else{
                    int index = check_vector_in_list(pat_temp,all_pats,all_pats_size);

                    pat_push(&all_pats[index], i);
                }
            }
        }
    }



    printf("Pat distintos totales %i \n",all_pats_size);
    pat all_pats_reduced[all_pats_size];
    int all_pats_reduced_size = 0;
    for (size_t i = 0; i < all_pats_size; i++){
        if (all_pats[i].size_vector_index>=add_size){
            pat_init(&all_pats_reduced[all_pats_reduced_size]);
            all_pats_reduced[all_pats_reduced_size] = all_pats[i];
            all_pats_reduced_size++;
        }
    }


    // for (size_t i = 0; i < all_pats_size; i++){
    //     pat_destroy(&all_pats[i]);
    // }
    
    printf("Pat distintos totales con aparicion mayor a la suma requerida %i \n",all_pats_reduced_size);

    int un_decimo=all_pats_reduced_size/10;
    int un_cuarto=all_pats_reduced_size/4;
    int un_medio=all_pats_reduced_size/2;
    int tres_cuartos=un_cuarto+un_medio;
    // exit(1);
    // #pragma omp parallel for num_threads(threads) schedule(dynamic)
    // for (int i = 0; i <= all_pats_reduced_size- add_size; i++) {
        
    //     for (size_t j = 0; j < all_pats_reduced[i].size_vector_index - add_size; j++){
    //         uint32_t selected[add_size];

    //         selected[0] = all_pats_reduced[i].vector_index[j];

    //         check_pat(j+1, 1, selected, all_pats_reduced[i]);
            
    //     }

    //     if(i%512 == 0)
    //         printf("%i\n",i);

    //     if(i==(un_decimo))
    //         printf("voy 1/10 %i\n",i);

    //     if(i==(un_cuarto))
    //         printf("voy 1/4 %i\n",i);
        
    //     if(i==(un_medio))
    //         printf("voy 1/2 %i\n",i);
        
    //     if(i==(tres_cuartos))
    //         printf("voy 3/4 %i\n",i);
    // }

    // printf("Termine \n");




    #pragma omp parallel num_threads(threads)
    {
        for (int i = 0; i <= all_pats_reduced_size - add_size; i++) {

            if (all_pats_reduced[i].size_vector_index < add_size) {
                continue;
            }

            size_t limit = all_pats_reduced[i].size_vector_index - add_size;
            #pragma omp for schedule(dynamic)
            for (size_t j = 0; j <= limit; j++) {

                uint32_t selected[add_size];

                selected[0] = all_pats_reduced[i].vector_index[j];

                check_pat(j + 1, 1, selected, all_pats_reduced[i]);
            }

            // if(i%512 == 0)
                printf("%i\n",i);

            if(i==(un_decimo))
                printf("voy 1/10 %i\n",i);

            if(i==(un_cuarto))
                printf("voy 1/4 %i\n",i);
            
            if(i==(un_medio))
                printf("voy 1/2 %i\n",i);
            
            if(i==(tres_cuartos))
                printf("voy 3/4 %i\n",i);
        }
    }
    printf("Termine \n");






// #define CHUNK 64

// #pragma omp parallel num_threads(threads)
// {
//     #pragma omp single
//     {
//         for (int i = 0; i <= all_pats_reduced_size - add_size; i++) {

//             if (all_pats_reduced[i].size_vector_index < add_size) {
//                 continue;
//             }

//             size_t limit = all_pats_reduced[i].size_vector_index - add_size;

//             for (size_t start = 0; start <= limit; start += CHUNK) {

//                 size_t end = start + CHUNK;
//                 if (end > limit + 1) {
//                     end = limit + 1;
//                 }

//                 #pragma omp task firstprivate(i, start, end)
//                 {
//                     for (size_t j = start; j < end; j++) {
//                         uint32_t selected[add_size];

//                         selected[0] = all_pats_reduced[i].vector_index[j];

//                         check_pat(j + 1, 1, selected, all_pats_reduced[i]);
//                     }
//                 }
//             }
//             if(i==(un_decimo))
//                 printf("voy 1/10 %i\n",i);

//             if(i==(un_cuarto))
//                 printf("voy 1/4 %i\n",i);
            
//             if(i==(un_medio))
//                 printf("voy 1/2 %i\n",i);
            
//             if(i==(tres_cuartos))
//                 printf("voy 3/4 %i\n",i);
        

//         }
//     }
// }


    for (size_t i = 0; i < all_pats_size; i++){
        pat_destroy(&all_pats[i]);
    }
    printf("Termine \n");
    return 0;

}




static inline void pat_init(pat *p)
{
    p->vector_index = NULL;
    p->size_vector_index = 0;
    p->capacity_vector_index = 0;
}




static inline void pat_push(pat *p, uint32_t value)
{
    if (p->size_vector_index == p->capacity_vector_index) {

        uint32_t new_capacity =
            (p->capacity_vector_index == 0) ? 8 : p->capacity_vector_index * 2;

        uint32_t *tmp = realloc(
            p->vector_index,
            new_capacity * sizeof(uint32_t));

        if (tmp == NULL) {
            perror("realloc");
            exit(EXIT_FAILURE);
        }

        p->vector_index = tmp;
        p->capacity_vector_index = new_capacity;
    }

    p->vector_index[p->size_vector_index++] = value;
}


static inline void pat_destroy(pat *p)
{
    free(p->vector_index);

    p->vector_index = NULL;
    p->size_vector_index = 0;
    p->capacity_vector_index = 0;
}


static inline void check_pat(int start, int depth, uint32_t selected[add_size], pat all_pats){

    
    
    if (depth == add_size){
        uint32_t add[add_size][size];

        for (int i = 0; i < add_size; i++) {
            uint64_t row = selected[i];
            index_to_vector(row, add[i], mod);
        }
        bool condition = add_size_vector(add);

        if (condition){
            #pragma omp critical
            {
            for (int i = 0; i < add_size; i++) {
                printf("%i-",selected[i]);
            }
            printf("\n-------------------------------\n");
            // total_vectors_serach=total_vectors_serach+1;
            // for (int i = 0; i < add_size; i++) {
            //     for (size_t j = 0; j < size; j++){
            //         printf("%d ", add[i][j]);
            //     }
            //     printf("\n");
            // }
            // printf("------------------------------------------");
            // printf("\n");
            // add_size_vector_2(add);
            // // exit(1);    
            }
        }
        
        
        return;
    }

    for (size_t i = start; i <= all_pats.size_vector_index - (add_size - depth); i++){
        selected[depth] = all_pats.vector_index[i];
        check_pat(i+1, depth+1, selected, all_pats);
    }


}









static inline void print_vector(unsigned int vector[]){
    for (int i = 0; i < size; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");

}

static inline void patt(uint32_t in[size],uint32_t out[size] ){


    int j =0;
    for (size_t i = 0; i < size; i++){

        if (in[i]!=0){
            out[j]=in[i];
            j++;
        }
    }

    for (size_t i = j; i < size; i++){
        out[i] = 0;
    }
}


static inline bool check_checksum_vector(uint32_t in[size] ){

    int j =0;
    int k =0;
    for (size_t i = 0; i < size; i++){

        j = in[i] + j;
        k = (in[i]*in[i]) + k;
    }
    if((j%mod==0) && (k%mod ==0) )
        return 1;
    else
        return 0;

}





// c=a+b
static inline void add_vectors(uint32_t a[size],uint32_t b[size],uint32_t c[size] ){
    for (size_t i = 0; i < size; i++){

        c[i] = (a[i] + b [i]) % mod;
    }


}


//return the index of the vector in the list of all pats, if the pat is not in the list return -1
static inline int check_vector_in_list(uint32_t a[size], pat all_pats[], uint32_t size_vectors_pats) {

    for (size_t i = 0; i < size_vectors_pats; i++){
        uint32_t reverse_vector[size];
        for (size_t j = 0; j < size; j++){
            reverse_vector[j]=(3-a[j])%3;
            if (reverse_vector[j]<0){
                reverse_vector[j]=0;
            }
            
        }
        
        if (compare_vectors(a, all_pats[i].vector_pat) ){
            return i;
        }
        if ( compare_vectors(reverse_vector, all_pats[i].vector_pat)){
            return -2;
        }
    }
    
    return -1;
}



static inline int compare_vectors(uint32_t a[size], uint32_t b[size]) {
    return memcmp(a, b, size * sizeof(uint32_t)) == 0;
}



static inline void copy_vectors(uint32_t a[size], uint32_t b[size]) {
    memcpy(b, a, size * sizeof(uint32_t));
}

static inline bool add_size_vector(uint32_t add[add_size][size] ){


    uint32_t pat_x[size];
    uint32_t pat_c[size];
    uint32_t c[size];

    patt(add[0], pat_x);

    // for (size_t i = 1; i < add_size; i++) {
    //     patt(add[i], pat_c);

    //     if (!compare_vectors(pat_c, pat_x)) {
    //         return false;
    //     }
    // }

    for (size_t i = 0; i < add_size; i++) {

        copy_vectors(add[i], c);

        for (size_t t = 1; t < add_size; t++) {
            size_t j = (i + t) % add_size;

            add_vectors(c, add[j], c);
            patt(c, pat_c);

            if (!compare_vectors(pat_c, pat_x)) {
                return false;
            }
        }
    }
    return true;
}


static inline bool add_size_vector_2(uint32_t add[add_size][size]) {

    uint32_t pat_x[size];
    uint32_t pat_c[size];
    uint32_t c[size];

    patt(add[0], pat_x);

    for (size_t i = 1; i < add_size; i++) {
        patt(add[i], pat_c);

        if (!compare_vectors(pat_c, pat_x)) {
            return false;
        }
    }

    #pragma omp critical
    {
        printf("pat a seguir\n");
        print_vector(pat_x);
    }

    for (size_t i = 0; i < add_size; i++) {

        copy_vectors(add[i], c);

        for (size_t t = 1; t < add_size; t++) {

            size_t j = (i + t) % add_size;

            add_vectors(c, add[j], c);
            patt(c, pat_c);

            if (!compare_vectors(pat_c, pat_x)) {
                #pragma omp critical
                {
                    printf("Vector no valido \n");
                    print_vector(pat_c);
                    print_vector(pat_x);
                }
                return false;
            }

            #pragma omp critical
            {
                printf("suma acumulada desde %zu agregando %zu\n", i, j);
                print_vector(pat_c);
            }
        }
    }

    #pragma omp critical
    {
        printf("vector valido\n");
    }

    return true;
}

static inline void check_vectors(int start, int depth, uint32_t selected[add_size],uint32_t pat[size]){

    
    
    if (depth == add_size){
        uint32_t add[add_size][size];

        for (int i = 0; i < add_size; i++) {
            uint64_t row = selected[i];
            index_to_vector(row, add[i], mod);
        }
        bool condition = add_size_vector(add);

        if (condition){
            #pragma omp critical
            {
            // for (int i = 0; i < add_size; i++) {
            //     printf("%i-",selected[i]);
            // }
            // printf("\n-------------------------------\n");
            total_vectors_serach=total_vectors_serach+1;
            // for (int i = 0; i < add_size; i++) {
            //     for (size_t j = 0; j < size; j++){
            //         printf("%d ", add[i][j]);
            //     }
            //     printf("\n");
            // }
            // printf("------------------------------------------");
            // printf("\n");
            // add_size_vector_2(add);
            // // exit(1);    
            }
        }
        
        
        return;
    }

    for (size_t i = start; i <= total_vectors - (add_size - depth); i++){
        selected[depth] = i;

        uint32_t vector_temp[size];

        index_to_vector(i, vector_temp, mod);   

        if(check_checksum_vector(vector_temp)){
            uint32_t pat_temp[size];
            patt(vector_temp, pat_temp);
            // check_vectors(i+1, depth+1, selected,pat_temp);
            if (compare_vectors(pat_temp, pat)) {
                check_vectors(i+1, depth+1, selected,pat_temp);
            }
        }

    }


}




static inline void index_to_vector(uint64_t index, uint32_t vector[size], int num_values) {
    for (int i = size - 1; i >= 0; i--) {
        // vector[i] = values[index % num_values];
        vector[i] = index % num_values;

        index /= num_values;
    }
}

