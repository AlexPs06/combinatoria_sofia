#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>

//variables globales que definen el tamaño del problea
#define size 19 //define la cantidad de posiciones del vector
#define mod 3 //define la cantidad de elementos existentes 
#define add_size 3 //tamaño de las sumas a realizar
#define threads 16 //Cantidad de hilos a usar para la busqueda de pats 




#define CUDA_CHECK(call)                                       \
do {                                                           \
    cudaError_t cuda_error = (call);                            \
    if (cuda_error != cudaSuccess) {                            \
        fprintf(                                               \
            stderr,                                            \
            "CUDA error en %s:%d: %s\n",                       \
            __FILE__,                                          \
            __LINE__,                                          \
            cudaGetErrorString(cuda_error)                      \
        );                                                     \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while (0)

typedef struct {
    uint32_t vector_pat[size];

    uint32_t *vector_index;
    uint32_t size_vector_index;
    uint32_t capacity_vector_index;
} pat;

typedef struct {
    uint32_t selected[add_size];
} valid_combination;



static inline int check_vector_in_list(uint32_t a[size], pat all_pats[], uint32_t size_vectors_pats);
static inline void pat_init(pat *p);
static inline void pat_push(pat *p, uint32_t value);
static inline void pat_destroy(pat *p);
void start_timer(void);
void end_timer(void);
static inline void print_vector(unsigned int vector[]);
static inline bool add_size_vector_2(uint32_t add[add_size][size] );

static inline bool check_vector_constraint(uint32_t in[size] );
int total_vectors = 1;
int total_vectors_serach = 0;
static inline void check_pat(int start, int depth, uint32_t selected[add_size], pat all_pats);


//CUDA

__device__ __host__ static inline void index_to_vector(uint64_t index, uint32_t vector[size], int num_values);
__device__ __host__ static inline bool add_size_vector(uint32_t add[add_size][size] );
__global__ void check_pat_CUDA(const uint32_t *vector_index,uint32_t size_vector_index, valid_combination *results, uint32_t *count, uint32_t max_results);
__host__ __device__  static inline void patt(uint32_t in[size],uint32_t out[size] );
__host__ __device__ static inline void copy_vectors(uint32_t a[size], uint32_t b[size]);
__device__ __host__ static inline void add_vectors(uint32_t a[size],uint32_t b[size],uint32_t c[size] );
__host__ __device__  int compare_vectors(const uint32_t a[size],const uint32_t b[size]);


// Definición de la estructura
size_t all_pats_size = 0;
time_t start_time;


pat all_pats[500000];

int main(int argc, char const *argv[])
{
    uint32_t pat_conocido[size];
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
        patt(temp_1,pat_conocido);
    }
    



    //contador de tiempo para calcular cuanto tardo en realizar la busqueda, este es el inicio 
    start_timer();

    //cantidad de vectores totales que exiten
    for (int i = 0; i < size; i++) {
        total_vectors *= mod;
    }
    printf("total_vectors %i \n", total_vectors);


    //busqueda en paralelo de los vectores usando openmp 
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (int i = 1; i <= total_vectors; i++) {
        //creacion del vector temporal del tamaño del vector a buscar
        uint32_t vector_temp[size];
        index_to_vector(i, vector_temp, mod);   


        //comprobamos las restriciones del vector. (si la suma de los elementos modulo el numero de elementos es 0)
        if(check_vector_constraint(vector_temp)){
            uint32_t pat_temp[size];
            patt(vector_temp,pat_temp);
            
            #pragma omp critical
            {
                //comprobamos si el vector que paso las restricciones no tiene un equivalente o es el equivalente de alguno ya existente 
                if(check_vector_in_list(pat_temp,all_pats,all_pats_size) == (-1) ){
                    
                    //añadimos el vector si es nuevo
                    pat_init(&all_pats[all_pats_size]);
                    for (size_t j = 0; j < size; j++){
                        all_pats[all_pats_size].vector_pat[j]=pat_temp[j];
                        
                    }
                    pat_push(&all_pats[all_pats_size], i);
                    all_pats_size++;
                }
                //si es un equivalente se evita
                else if (check_vector_in_list(pat_temp,all_pats,all_pats_size) == (-2)) {

                //si el vector aparece y no es un equivalente sino directamente el mismo se añade la aparicion a la lista de indices
                }else{
                    int index = check_vector_in_list(pat_temp,all_pats,all_pats_size);

                    pat_push(&all_pats[index], i);
                }
            }
        }
    }



    
    //se imprime y se crea una variable del tamaño de todos los pat distintos encontrados
    printf("Pat distintos totales %lu \n",all_pats_size);
    pat *all_pats_reduced = (pat *)malloc(all_pats_size * sizeof(pat));

    if (all_pats_reduced == NULL) {
        fprintf(stderr, "No se pudo reservar all_pats_reduced\n");
        exit(EXIT_FAILURE);
    }

    //reducimos la variable a solo los que aparezcan mas de la cantidad de la suma buscada
    size_t all_pats_reduced_size = 0;
    for (size_t i = 0; i < all_pats_size; i++) {
        if (all_pats[i].size_vector_index >= add_size) {
            all_pats_reduced[all_pats_reduced_size] = all_pats[i];
            all_pats_reduced_size++;

        }
    }

    //hacemos un realloc para reservar la memoria exacta necesaria y no memoria extra
    pat *tmp = (pat *)realloc(all_pats_reduced,all_pats_reduced_size * sizeof(pat));

    if (tmp != NULL || all_pats_reduced_size == 0) {
        all_pats_reduced = tmp;
    }

    
    //recalculamos los pat que si se revisaran
    printf("Pat distintos totales con aparicion mayor a la suma requerida %lu \n",all_pats_reduced_size);


    size_t un_decimo=all_pats_reduced_size/10;
    size_t un_cuarto=all_pats_reduced_size/4;
    size_t un_medio=all_pats_reduced_size/2;
    size_t tres_cuartos=un_cuarto+un_medio;


    //valor maximo de la combinatoria a comprobar
    uint64_t max_vector_index_size = 0;


    //ordenamos por cantidad de aparicion de indices cada pat, usando el metodo
    //burbuja (esto se puede mejorar pero es irrelevante)
    for (size_t i = 0; i < all_pats_reduced_size; i++) {
        for (size_t j = 0; j < all_pats_reduced_size-i-1; j++) {

            if (all_pats_reduced[j].size_vector_index > all_pats_reduced[j+1].size_vector_index) {
                pat aux = all_pats_reduced[j];
                all_pats_reduced[j] = all_pats_reduced[j+1]; 
                all_pats_reduced[j+1]=aux;
            }
        }
    }



    //comprobación del vector que sabemos que existe, que debe de existir en el caso que tenemos
    //Se eliminara una vez se acaben las pruebas de tamaño 19
    size_t indice = 0;
    if(check_vector_in_list(pat_conocido,all_pats_reduced,all_pats_reduced_size) >= 0){
        printf("%i indice del pat conocido\n", check_vector_in_list(pat_conocido,all_pats_reduced,all_pats_reduced_size) );
        indice=check_vector_in_list(pat_conocido,all_pats_reduced,all_pats_reduced_size);
    }

    //impresion de la cantidad distintas de tamaños de vectores a usar, se puede eliminar
    for (size_t i = 0; i < all_pats_reduced_size; i++) {
        if(i ==0 || all_pats_reduced[i].size_vector_index != all_pats_reduced[i-1].size_vector_index)
            printf("%i\n",all_pats_reduced[i].size_vector_index);
    }
    //asigancion de la mayor cantidad de valores a probar
    max_vector_index_size=all_pats_reduced[all_pats_reduced_size-1].size_vector_index;

    //condicion de comprobación para ver si es factible realizar la busqueda
    if (max_vector_index_size < add_size) {
        fprintf(stderr, "No existen grupos suficientes para comprobar\n");
        return;
    }else{
        printf("Tamaño maximo a comprobar %lu\n",max_vector_index_size);
    }


    //vector de indices a copiar a cuda
    //las variables con d al inicio o al final indican divice que es decir la tarjeta de cuda
    //las variables con h al inicio o al final indican el host es decir variables que tienen valor en la cpu
    //es necesario renombrar para un mejor entendimiento
    
    
    uint32_t *d_vector_index = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_vector_index,max_vector_index_size * sizeof(uint32_t)));

    //cantidad de hilos a usar por bloque
    const uint32_t threads_cuda = 256;


    //variable de los resulados del pat en cuda
    valid_combination *pat_results_d;
    uint32_t *pat_results_count;

    uint32_t max_results = 10000; // ajusta según tu caso

    cudaMalloc(&pat_results_d, max_results * sizeof(valid_combination));
    cudaMalloc(&pat_results_count, sizeof(uint32_t));
                                // i<all_pats_reduced_size
    for (size_t i = 0; i<all_pats_reduced_size ; i++) {

        const size_t count =
            all_pats_reduced[i].size_vector_index;

        if (count < add_size) {
            continue;
        }
        if(i==indice){
            printf("voy a hacer el indice %lu\n",i);
            printf("Combinatoria a realizar %lu\n",count);
        }
        printf("indice: %lu\n",i);

        //copiado de memoria del vector de pat a cuda
        CUDA_CHECK(cudaMemcpy(d_vector_index,all_pats_reduced[i].vector_index,count * sizeof(uint32_t),cudaMemcpyHostToDevice));

        //division de los bloques de trabajo para la tarjeta de video
        const size_t work_items = count - 2;
        const size_t blocks = (work_items + threads_cuda - 1) / threads_cuda;

        //ejecucion del kernel de cuda, se copia los valores previos y se ejecuta la comprobación
        check_pat_CUDA<<<blocks, threads_cuda>>>(d_vector_index,count,pat_results_d,pat_results_count,max_results );

        //copiamos la cantidad de resultados a una variable en cpu para poder usarla 
        uint32_t pat_results_count_cpu;
        CUDA_CHECK(cudaMemcpy(&pat_results_count_cpu, pat_results_count,sizeof(uint32_t),cudaMemcpyDeviceToHost));
        
        //si los resultados son mayores a 0 se copia al vector del tamaño de la combinación
        if(pat_results_count_cpu>0){
            valid_combination *pat_results_cpu =(valid_combination *)malloc(pat_results_count_cpu * sizeof(valid_combination));
            CUDA_CHECK(cudaMemcpy(pat_results_cpu,pat_results_d,pat_results_count_cpu * sizeof(valid_combination),cudaMemcpyDeviceToHost));

            //Escritura en archivo de texto de los resultados
            FILE *fp = fopen("resultados-1.txt", "a");

            // Escribir el PAT actual
            fprintf(fp, "PAT: ");
            for (int p = 0; p < size; p++) {
                fprintf(fp, "%u ", all_pats_reduced[i].vector_pat[p]);
            }
            fprintf(fp, "\n");

            // Escribir las combinaciones válidas
            for (uint32_t r = 0; r < pat_results_count_cpu; r++) {
                fprintf(fp, "  %u %u %u\n",
                        pat_results_cpu[r].selected[0],
                        pat_results_cpu[r].selected[1],
                        pat_results_cpu[r].selected[2]);
            }

            fprintf(fp, "-------------------------------\n");

            fclose(fp);
            free(pat_results_cpu);
        }


        CUDA_CHECK(cudaGetLastError());
        

        if(i==(un_decimo))
            printf("voy 1/10 %lu\n",i);

        if(i==(un_cuarto))
            printf("voy 1/4 %lu\n",i);
        
        if(i==(un_medio))
            printf("voy 1/2 %lu\n",i);
        
        if(i==(tres_cuartos))
            printf("voy 3/4 %lu\n",i);

        
        // break;
    }

    /*
    * Espera a que terminen todos los kernels y detecta
    * errores ocurridos durante la ejecución.
    */
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_vector_index));




    for (size_t i = 0; i < all_pats_size; i++){
        pat_destroy(&all_pats[i]);
    }
    printf("Termine \n");
    printf("Busqueda size %i mod %i add_size %i \n", size,mod,add_size);
    
    end_timer();
    return 0;

}




void start_timer(void)
{
    time(&start_time);

    printf("=====================================\n");
    printf("Inicio : %s", ctime(&start_time));
    printf("=====================================\n");
}

void end_timer(void)
{
    time_t end_time;
    time(&end_time);

    double elapsed = difftime(end_time, start_time);

    int hours   = (int)elapsed / 3600;
    int minutes = ((int)elapsed % 3600) / 60;
    int seconds = (int)elapsed % 60;

    printf("\n=====================================\n");
    printf("Inicio : %s", ctime(&start_time));
    printf("Fin    : %s", ctime(&end_time));
    printf("-------------------------------------\n");
    printf("Tiempo total: %02d:%02d:%02d\n",
           hours, minutes, seconds);
    printf("=====================================\n");
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

        uint32_t *tmp = (uint32_t*)realloc(
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
            }
        }
        
        
        return;
    }

    for (size_t i = start; i <= all_pats.size_vector_index - (add_size - depth); i++){
        selected[depth] = all_pats.vector_index[i];
        check_pat(i+1, depth+1, selected, all_pats);
    }


}



__global__ void check_pat_CUDA(const uint32_t *vector_index,uint32_t size_vector_index, valid_combination *results, uint32_t *count, uint32_t max_results) {
   

    const size_t start = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    const size_t stride = (size_t)blockDim.x * gridDim.x;

    uint32_t add[add_size][size];
    uint32_t selected[add_size];

    for (size_t i = start;i < (size_t)size_vector_index - 2; i += stride) {

        selected[0] = vector_index[i];
        index_to_vector(selected[0], add[0], mod);

        for (size_t j = i + 1;j < (size_t)size_vector_index - 1;j++) {

            selected[1] = vector_index[j];
            index_to_vector(selected[1], add[1], mod);

            for (size_t k = j + 1;k < size_vector_index;k++) {

                selected[2] = vector_index[k];
                index_to_vector(selected[2], add[2], mod);

                const bool condition =
                    add_size_vector(add);

                if (condition) {
                    uint32_t pos = atomicAdd(count, 1);
                    if (pos < max_results) {
                        results[pos].selected[0] = selected[0];
                        results[pos].selected[1] = selected[1];
                        results[pos].selected[2] = selected[2];
                    }
                }
            }
        }
    }
}







static inline void print_vector(unsigned int vector[]){
    for (int i = 0; i < size; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");

}

__host__ __device__ static inline void patt(uint32_t in[size],uint32_t out[size] ){


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


static inline bool check_vector_constraint(uint32_t in[size] ){

    int j =0;
    int k =0;
    int cantidad_uno=0;
    int cantidad_dos=0;

    for (size_t i = 0; i < size; i++){

        j = in[i] + j;
        k = (in[i]*in[i]) + k;
        if(in[i]==1)
            cantidad_uno++;
        if(in[i]==2)
            cantidad_dos++;

    }
    if((j%mod==0) && (k%mod ==0) && (cantidad_dos==cantidad_uno) )
        return 1;
    else
        return 0;

}





// c=a+b
__device__ __host__ static inline void add_vectors(uint32_t a[size],uint32_t b[size],uint32_t c[size] ){
    for (size_t i = 0; i < size; i++){

        c[i] = (a[i] + b [i]) % mod;
    }


}


//return the index of the vector in the list of all pats, if the pat is not in the list return -1
static inline int check_vector_in_list(uint32_t a[size], pat all_pats[], uint32_t size_vectors_pats) {
    
    for (size_t i = 0; i < size_vectors_pats; i++){
        uint32_t reverse_vector[size]={0};
        for (size_t j = 0; j < size; j++){
            // logica original
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



__host__ __device__ int compare_vectors(const uint32_t a[size],const uint32_t b[size]){
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i])
            return 0;
    }

    return 1;
}



__host__ __device__ static inline void copy_vectors(uint32_t a[size], uint32_t b[size]) {
    for (int i = 0; i < size; i++) {
        b[i] = a[i];
    }
}

__device__ __host__ static inline bool add_size_vector(uint32_t add[add_size][size] ){


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






__host__ __device__ 
static inline void index_to_vector(uint64_t index, uint32_t vector[size], int num_values) {
    for (int i = size - 1; i >= 0; i--) {
        // vector[i] = values[index % num_values];
        vector[i] = index % num_values;

        index /= num_values;
    }
}

