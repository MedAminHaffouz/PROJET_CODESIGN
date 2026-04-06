import pyopencl as cl

platforms = cl.get_platforms()

for i, platform in enumerate(platforms):
    print(f"\n{'='*60}")
    print(f"Platform [{i}]: {platform.name}")
    print(f"  Vendor  : {platform.vendor}")
    print(f"  Version : {platform.version}")

    devices = platform.get_devices()
    for j, device in enumerate(devices):
        print(f"\n  Device [{j}]: {device.name}")
        print(f"    Type                  : {cl.device_type.to_string(device.type)}")
        print(f"    Compute Units         : {device.max_compute_units}")
        print(f"    Max Clock Freq        : {device.max_clock_frequency} MHz")
        print(f"    Global Memory         : {device.global_mem_size / (1024**3):.2f} GiB")
        print(f"    Global Cache          : {device.global_mem_cache_size / 1024:.0f} KiB")
        print(f"    Global Cache Line     : {device.global_mem_cacheline_size} B")
        print(f"    Local Memory          : {device.local_mem_size / 1024:.0f} KiB")
        print(f"    Constant Memory       : {device.max_constant_buffer_size / 1024:.0f} KiB")
        print(f"    Max Work Group Size   : {device.max_work_group_size}")
        print(f"    Max Work Item Sizes   : {device.max_work_item_sizes}")