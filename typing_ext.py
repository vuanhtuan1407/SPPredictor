# add_argument() in argparse lib is not working with type=Union, define a function and use instead of type Union
def union_devices(devices):
    if isinstance(devices, int) or (isinstance(devices, list) and all(isinstance(device, int) for device in devices)):
        return devices
    else:
        return 'auto'
