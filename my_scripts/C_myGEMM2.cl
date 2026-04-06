#define TS 16 // Tile Size (WSZ in your exam)

__kernel void mmul(
    const int N,
    __global const float* A,
    __global const float* B,
    __global float* C) 
{
    // Thread identifiers
    const int row = get_local_id(1); // Local row ID (0 to 15)
    const int col = get_local_id(0); // Local col ID (0 to 15)
    
    // Global identifiers
    const int globalRow = TS * get_group_id(1) + row; 
    const int globalCol = TS * get_group_id(0) + col; 

    // Local memory for a TSxTS tile (The "Cache")
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float acc = 0.0f;
    const int numTiles = N / TS;

    // Loop over all tiles
    for (int t = 0; t < numTiles; t++) {
        
        // 1. Load one tile of A and B into local memory
        Asub[row][col] = A[globalRow * N + (t * TS + col)];
        Bsub[row][col] = B[(t * TS + row) * N + globalCol];

        // 2. Synchronize to ensure the whole WorkGroup has finished loading the tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Perform computation for this single tile using the fast local memory
        for (int k = 0; k < TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }

        // 4. Synchronize before loading the next tile so we don't overwrite data being used
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result back to global memory
    C[globalRow * N + globalCol] = acc;
}