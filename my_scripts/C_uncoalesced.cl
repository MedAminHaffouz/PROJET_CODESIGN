__kernel void mmul(
    const int N,
    __global const float* A,
    __global const float* B,
    __global float* C) 
{
    // UNCOALESCED: i (row) is mapped to id(0) and j (col) is mapped to id(1)
    int i = get_global_id(0); 
    int j = get_global_id(1); 
    
    float acc = 0.0f;
    for (int k = 0; k < N; k++) {
        acc += A[i * N + k] * B[k * N + j];
    }
    
    C[i * N + j] = acc;
}