import pyopencl as cl

if __name__ == "__main__":
    for platform in cl.get_platforms():
        print('PLATFORM: {} ({})'.format(platform.name, platform.vendor))
        indent = '\t'
        print(indent + 'Version: ' + platform.version)
        print(indent + 'Profile: ' + platform.profile)
        print(indent + 'Extensions: ' + ', '.join(platform.extensions.strip().split(' ')))
        for device in platform.get_devices():
            print('\t\tDevice: {} ({})'.format(device.name, device.vendor))
            for name, flag in [('Version', device.version),
                               ('Type', cl.device_type.to_string(device.type)),
                               ('Extensions', ', '.join(e for e in device.extensions.strip().split(' ') if e)),
                               ('Memory (global)', str(device.global_mem_size)),
                               ('Memory (local)', str(device.local_mem_size)),
                               ('Address bits', str(device.address_bits)),
                               ('Max work item dims', str(device.max_work_item_dimensions)),
                               ('Max work item sizes', str(device.max_work_item_sizes)),
                               ('Max work group size', str(device.max_work_group_size)),
                               ('Max compute units', str(device.max_compute_units)),
                               ('Driver version', device.driver_version),
                               ('Image support', str(bool(device.image_support))),
                               ('Little endian', str(bool(device.endian_little))),
                               ('Device available', str(bool(device.available))),
                               ('Compiler available', str(bool(device.compiler_available))),
                               ('Mem Base Addr Align', device.get_info(cl.device_info.MEM_BASE_ADDR_ALIGN))
                               ]:
                print('\t\t\t{0:<25}{1:<10}'.format(name + ':', flag))
            print()
