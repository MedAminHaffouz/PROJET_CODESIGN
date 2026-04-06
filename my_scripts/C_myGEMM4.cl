#define TS 16     // Tile Size
#define WIDTH 4   // Vector width (float4)

__kernel void mmul(
    const int N,
    __global const float* A,
    __global const float* B,
    __global float* C) 
{
    // Thread identifiers 
    const int row = get_local_id(1); 
    const int col = get_local_id(0); // Goes from 0 to 3 (because TS/WIDTH)
    
    const int globalRow = TS * get_group_id(1) + row;
    const int globalCol = (TS / WIDTH) * get_group_id(0) + col;

    // Local memory for the 16x16 tile
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialize accumulator as a single float4 vector
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    const int numTiles = N / TS;
    
    for (int t = 0; t < numTiles; t++) {
        // Load A and B into local memory using vector loads (vload4)
        // This pulls 4 floats from global memory in a single fast instruction
        float4 a_vec = vload4(0, &A[globalRow * N + t * TS + col * WIDTH]);
        float4 b_vec = vload4(0, &B[(t * TS + row) * N + globalCol * WIDTH]);
        
        // Store into local memory
        Asub[row][col * WIDTH + 0] = a_vec.x;
        Asub[row][col * WIDTH + 1] = a_vec.y;
        Asub[row][col * WIDTH + 2] = a_vec.z;
        Asub[row][col * WIDTH + 3] = a_vec.w;

        Bsub[row][col * WIDTH + 0] = b_vec.x;
        Bsub[row][col * WIDTH + 1] = b_vec.y;
        Bsub[row][col * WIDTH + 2] = b_vec.z;
        Bsub[row][col * WIDTH + 3] = b_vec.w;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the math 
        for (int k = 0; k < TS; k++) {
            // Read 1 scalar from A, and 1 vector (float4) from B
            float a_val = Asub[row][k];
            float4 b_val = (float4)(Bsub[k][col * WIDTH + 0], 
                                    Bsub[k][col * WIDTH + 1], 
                                    Bsub[k][col * WIDTH + 2], 
                                    Bsub[k][col * WIDTH + 3]);
            // Vector math: multiply scalar by vector and add to accumulator
            acc += a_val * b_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the float4 result back to global matrix C in one instruction
    vstore4(acc, 0, &C[globalRow * N + globalCol * WIDTH]);
}