"""Darwin resource/swap telemetry, with unavailable fields explicitly labeled."""

import ctypes
import json
import os
import re
import resource
import subprocess
import threading
import time


def sysctl(name):
    result = subprocess.run(["/usr/sbin/sysctl","-n",name],capture_output=True,text=True,timeout=5)
    if result.returncode:
        raise OSError(result.stderr.strip())
    return result.stdout.strip()


def swap_snapshot():
    try:
        raw = sysctl("vm.swapusage")
        matches = re.findall(r"(total|used|free)\s*=\s*([\d.]+)([KMG])",raw)
        values = {name+"_bytes":round(float(number)*{"K":1024,"M":1024**2,"G":1024**3}[unit]) for name,number,unit in matches}
        if len(values)!=3:
            raise ValueError("Unrecognized vm.swapusage output")
        return {"available":True,"scope":"system-wide",**values,"raw":raw}
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:
        return {"available":False,"scope":"system-wide","reason":str(error)}


# Exact field layout from the installed macOS SDK sys/resource.h,
# struct rusage_info_v4; all fields after ri_uuid are uint64_t.
FIELDS = "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles billed_energy serviced_energy interval_max_phys_footprint runnable_time".split()


class RUsageV4(ctypes.Structure):
    _fields_ = [("uuid",ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in FIELDS]


def process_memory():
    result = {"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "peak_rss_source":"Darwin getrusage(RUSAGE_SELF), bytes"}
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib",use_errno=True)
        fn = library.proc_pid_rusage
        fn.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        fn.restype = ctypes.c_int
        data = RUsageV4()
        if fn(os.getpid(),4,ctypes.byref(data)) != 0:
            raise OSError(ctypes.get_errno(),"proc_pid_rusage")
        result.update(current_rss_bytes=data.resident_size,current_physical_footprint_bytes=data.phys_footprint,
                      peak_physical_footprint_bytes=data.lifetime_max_phys_footprint,
                      footprint_source="Darwin proc_pid_rusage RUSAGE_INFO_V4; includes charged compressed memory")
    except OSError as error:
        result.update(footprint_unavailable=str(error))
    return result


def snapshot():
    return {"monotonic_seconds":time.monotonic(),"process":process_memory(),"swap":swap_snapshot()}


class Monitor:
    def __init__(self,path,interval=2):
        self.path,self.interval=path,interval
        self.stop_event=threading.Event()

    def __enter__(self):
        self.stream=self.path.open("x")
        self.sample()
        def run():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread=threading.Thread(target=run,daemon=True)
        self.thread.start()
        return self

    def sample(self):
        self.stream.write(json.dumps(snapshot())+"\n")
        self.stream.flush()

    def __exit__(self,*exc):
        self.stop_event.set()
        self.thread.join(timeout=10)
        self.sample()
        self.stream.close()
