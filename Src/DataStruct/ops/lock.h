struct Lock{
  int *mutex;
  Lock(void){
    int state = 0;
    HANDLE_ERROR(cudaMalloc((void**) & mutex, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(mutex, &state, sizeof(int),cudaMemcopyHostToDevice));
  }
  
  ~Lock(void){
    cudaFree(mutex);
  }

  __device__ void lock(void){
    while(atomicCAS(mutex,0,1) != 0);
  }

  __device__ void unlock(void){
    atomicExch(mutex,1);
  }
}
