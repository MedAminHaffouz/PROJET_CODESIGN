#
# Matrix Multiplication: 1D Register Tiling (myGEMM3)
#
from helper import *
from definitions import *
import numpy
import pyopencl as cl
from time import time
import os

# A[N][N], B[N][N], C[N][N]
N = 8192;

# Number of elements in the matrix
size = N * N

# Fill Matrices
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

# --------------------------------------------------------------------------------
# CHOOSE KERNEL TO EXECUTE 
# --------------------------------------------------------------------------------
kernel_name = "C_myGEMM3.cl"

print (f"Matrix multiplication {N}*{N} repeated {COUNT} times")
print (f"Executing kernel: {kernel_name} (1D Register Tiling)\n")

# --------------------------------------------------------------------------------
# CHOOSE localsize : 2, 4, 8 , 16 or 32
# --------------------------------------------------------------------------------
kernel_size = input("Please enter a value for localsize. (Use 16 to match the kernel) :\n")

if (kernel_size in ['4','8','16','32'] ):
    localsize = int(kernel_size)
    print ("Blocks Size is", localsize, "*", localsize, "\n")
else:
    print ("=== No valid input. Default Size 16 will be used. Block Size = 16*16")
    localsize = 16

# Set up OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Reset host buffers 
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)

# Load and compile Kernel
kernelsource = open(kernel_name).read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])

# Do the multiplication COUNT times
print ("\n Starting ", COUNT, " OpenCL Matrix Multiplications")
start_time = time()

for i in range(COUNT):    
    try:
        # CRITICAL CHANGE: 
        # Work-Per-Thread (WPT) is 4. Each thread does 4 columns of work.
        # We must reduce the X dimension mapping by a factor of 4.
        WPT = 4
        
        global_dim = (int(N/WPT), N)
        local_dim = (int(localsize/WPT), localsize)

        mmul(queue, global_dim, local_dim, numpy.int32(N), d_a, d_b, d_c)
        queue.finish()
    except Exception as e:
        print (f" === Error for localsize = {localsize} : {e} ===\n")
        break

run_time = time() - start_time
print ("\n End of", COUNT, "Matrix Multiplications\n")

# Calculate Performance
results(N, COUNT, run_time)

# Read the result h_C
cl.enqueue_copy(queue, h_C, d_c)