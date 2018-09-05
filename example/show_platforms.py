import pyopencl as cl
if __name__ == "__main__":
    indent = ''
    for platform in cl.get_platforms():
        print(indent + '{} ({})'.format(platform.name, platform.vendor))
        indent = '\t'
        print(indent + 'Version: ' + platform.version)
        print(indent + 'Profile: ' + platform.profile)
        print(indent + 'Extensions: ' + ', '.join(platform.extensions.strip().split(' ')))
        for device in platform.get_devices():
            # device = platform.get_devices()[0]
            print(indent + '{} ({})'.format(device.name, device.vendor))

            indent = '\t\t\t'
            flags = [('Version', device.version),
                     ('Type', cl.device_type.to_string(device.type)),
                     ('Extensions', ', '.join(device.extensions.strip().split(' '))),
                     ('Memory (global)', str(device.global_mem_size)),
                     ('Memory (local)', str(device.local_mem_size)),
                     ('Address bits', str(device.address_bits)),
                     ('Max work item dims', str(device.max_work_item_dimensions)),
                     ('Max work group size', str(device.max_work_group_size)),
                     ('Max compute units', str(device.max_compute_units)),
                     ('Driver version', device.driver_version),
                     ('Image support', str(bool(device.image_support))),
                     ('Little endian', str(bool(device.endian_little))),
                     ('Device available', str(bool(device.available))),
                     ('Compiler available', str(bool(device.compiler_available)))]

            [print(indent + '{0:<25}{1:<10}'.format(name + ':', flag)) for name, flag in flags]
            # print("Device: ", device)
            # device.
            # print("  MEM BASE ADDR ALIGN", device.get_info(cl.device_info.MEM_BASE_ADDR_ALIGN))
