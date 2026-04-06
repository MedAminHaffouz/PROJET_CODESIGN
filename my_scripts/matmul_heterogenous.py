#
# Heterogeneous Matrix Multiplication (NVIDIA + Intel)
#
import pyopencl as cl
import numpy as np
from time import time

# Define Matrix and Split Sizes
N = 8192
ROWS_NV = 7632    # NVIDIA's share (93.14%)
ROWS_INT = 560    # Intel's share (6.86%)
COUNT = 10        # Number of times to run the test

AVAL = 2.0
BVAL = 3.0

print(f"Initializing Matrices for N={N}...")
# Create full matrices
h_A = np.full(N * N, AVAL, dtype=np.float32)
h_B = np.full(N * N, BVAL, dtype=np.float32)

# SLICE MATRIX A:
# NVIDIA gets the first 7632 rows. Intel gets the remaining 560 rows.
# Matrix B is needed completely by both devices to do the math.
h_A_nv = h_A[: ROWS_NV * N]
h_A_int = h_A[ROWS_NV * N :]

# Output buffers
h_C_nv = np.empty(ROWS_NV * N, dtype=np.float32)
h_C_int = np.empty(ROWS_INT * N, dtype=np.float32)

# -------------------------------------------------------------------------
# 1. AUTO-DETECT DEVICES
# -------------------------------------------------------------------------
platforms = cl.get_platforms()
dev_nv = None
dev_int = None

for p in platforms:
    for d in p.get_devices():
        if "NVIDIA" in d.name:
            dev_nv = d
        elif "Portable" in p.name or "Intel" in d.name or "pthread" in d.name:
            dev_int = d

if not dev_nv or not dev_int:
    print("Error: Could not automatically find both NVIDIA and Intel devices.")
    exit(1)

print(f"\n--- Heterogeneous Devices Found ---")
print(f"Device 1 (Uncoalesced): {dev_nv.name}")
print(f"Device 2 (myGEMM3)    : {dev_int.name}\n")

# -------------------------------------------------------------------------
# 2. CREATE SEPARATE CONTEXTS & QUEUES
# -------------------------------------------------------------------------
ctx_nv = cl.Context([dev_nv])
queue_nv = cl.CommandQueue(ctx_nv)

ctx_int = cl.Context([dev_int])
queue_int = cl.CommandQueue(ctx_int)

# -------------------------------------------------------------------------
# 3. CREATE MEMORY BUFFERS FOR BOTH DEVICES
# -------------------------------------------------------------------------
# NVIDIA Buffers
d_A_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A_nv)
d_B_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_C_nv = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY, size=h_C_nv.nbytes)

# Intel Buffers
d_A_int = cl.Buffer(ctx_int, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A_int)
d_B_int = cl.Buffer(ctx_int, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_C_int = cl.Buffer(ctx_int, cl.mem_flags.WRITE_ONLY, size=h_C_int.nbytes)

# -------------------------------------------------------------------------
# 4. COMPILE KERNELS
# -------------------------------------------------------------------------
prg_nv = cl.Program(ctx_nv, open("C_uncoalesced.cl").read()).build()
mmul_nv = prg_nv.mmul
mmul_nv.set_scalar_arg_dtypes([np.int32, None, None, None])

prg_int = cl.Program(ctx_int, open("C_myGEMM3.cl").read()).build()
mmul_int = prg_int.mmul
mmul_int.set_scalar_arg_dtypes([np.int32, None, None, None])

# -------------------------------------------------------------------------
# 5. EXECUTE CONCURRENTLY
# -------------------------------------------------------------------------
print(f"Starting {COUNT} Heterogeneous Matrix Multiplications...")
start_time = time()

for i in range(COUNT):
    # Dispatch NVIDIA (Global sizes match the uncoalesced i,j logic)
    # i=row (ROWS_NV), j=col (N)
    global_dim_nv = (ROWS_NV, N)
    local_dim_nv = (16, 16)
    mmul_nv(queue_nv, global_dim_nv, local_dim_nv, np.int32(N), d_A_nv, d_B_nv, d_C_nv)

    # Dispatch Intel (myGEMM3 uses WPT=4, so X-dimension N is divided by 4)
    WPT = 4
    global_dim_int = (int(N / WPT), ROWS_INT)
    local_dim_int = (int(16 / WPT), 16)
    mmul_int(queue_int, global_dim_int, local_dim_int, np.int32(N), d_A_int, d_B_int, d_C_int)

    # Wait for BOTH devices to finish their halves!
    queue_nv.finish()
    queue_int.finish()

run_time = time() - start_time

# -------------------------------------------------------------------------
# 6. RETRIEVE AND MERGE RESULTS
# -------------------------------------------------------------------------
cl.enqueue_copy(queue_nv, h_C_nv, d_C_nv)
cl.enqueue_copy(queue_int, h_C_int, d_C_int)

# Stitch the matrix back together!
h_C_final = np.concatenate((h_C_nv, h_C_int))

print(f"\nEnd of Multiplications.")
print(f"Total time for {COUNT} runs: {run_time:.4f} seconds")

# Calculate Combined GFLOPS
# Operations per run = 2 * N^3
operations_total = COUNT * (2.0 * (N ** 3))
gflops = (operations_total / run_time) / 1e9

print(f"Total Heterogeneous Performance: {gflops:.2f} GFLOPS/sec")