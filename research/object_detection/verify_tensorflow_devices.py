from tensorflow.python.client import device_lib
def get_devices():
    return [x.name for x in device_lib.list_local_devices()]
print (get_devices())