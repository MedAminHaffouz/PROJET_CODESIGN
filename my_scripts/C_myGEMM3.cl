#define TS 16  // Tile Size (The local memory block)
#define WPT 4  // Work-Per-Thread (How many elements 1 thread calculates)
#define RTS 4  // Reduced Tile Size (TS / WPT)

__kernel void mmul(
    const int N,
    __global const float* A,
    __global const float* B,
    __global float* C) 
{
    // Thread identifiers 
    const int row = get_local_id(1); // Local row ID (0 to 15)
    const int col = get_local_id(0); // Local col ID (Now only 0 to 3!)
    
    const int globalRow = TS * get_group_id(1) + row;
    const int globalCol = TS * get_group_id(0) + col;

    // Local memory (Still here from Improvement 1!)
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // NEW: Private registers to hold the 4 calculations
    float acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    const int numTiles = N / TS;
    
    for (int t = 0; t < numTiles; t++) {
        // Load A and B into local memory. 
        // Because we only have 4 threads per row now, each thread must load 4 elements to fill the 16-wide tile.
        for (int w = 0; w < WPT; w++) {
            Asub[row][col + w * RTS] = A[globalRow * N + (t * TS + col + w * RTS)];
            Bsub[row][col + w * RTS] = B[(t * TS + row) * N + (TS * get_group_id(0) + col + w * RTS)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the math for the 4 elements
        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                // Multiplying data from Local Memory and storing in Instant Registers (acc)
                acc[w] += Asub[row][k] * Bsub[k][col + w * RTS];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the 4 results from registers back to the global matrix C
    for (int w = 0; w < WPT; w++) {
        C[globalRow * N + (globalCol + w * RTS)] = acc[w];
    }
}