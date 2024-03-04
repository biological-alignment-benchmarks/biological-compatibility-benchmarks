# -*- coding: utf-8 -*-

#
# Creative Commons Attribution Share Alike 4.0 International License
# Roland Pihlakas, 2017, roland@simplify.ee
# Aya, 2013, https://stackoverflow.com/users/172176/aya
#


# https://stackoverflow.com/questions/16779497/how-to-set-memory-limit-for-thread-or-process-in-python

import ctypes
import os
import psutil



PROCESS_SET_QUOTA = 0x100
PROCESS_TERMINATE = 0x1
JobObjectExtendedLimitInformation = 9


# https://msdn.microsoft.com/en-us/library/windows/desktop/ms684147(v=vs.85).aspx

JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200

JOB_OBJECT_LIMIT_WORKINGSET = 0x00000001

JOB_OBJECT_LIMIT_PRESERVE_JOB_TIME = 0x00000040   # Preserves any job time limits you previously set. As long as this flag is set, you can establish a per-job time limit once, then alter other limits in subsequent calls. This flag cannot be used with JOB_OBJECT_LIMIT_JOB_TIME.
JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000 


# https://stackoverflow.com/questions/6266820/working-example-of-createjobobject-setinformationjobobject-pinvoke-in-net
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008 
JOB_OBJECT_LIMIT_AFFINITY = 0x00000010 
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800 
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400 
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200 
JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004 
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000 
JOB_OBJECT_LIMIT_PRESERVE_JOB_TIME = 0x00000040 
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x00000020 
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100 
JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002 
JOB_OBJECT_LIMIT_SCHEDULING_CLASS = 0x00000080 
JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x00001000 
JOB_OBJECT_LIMIT_WORKINGSET = 0x00000001


class IO_COUNTERS(ctypes.Structure):
  _fields_ = [('ReadOperationCount', ctypes.c_uint64),
              ('WriteOperationCount', ctypes.c_uint64),
              ('OtherOperationCount', ctypes.c_uint64),
              ('ReadTransferCount', ctypes.c_uint64),
              ('WriteTransferCount', ctypes.c_uint64),
              ('OtherTransferCount', ctypes.c_uint64)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
  _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64),
              ('PerJobUserTimeLimit', ctypes.c_int64),
              ('LimitFlags', ctypes.c_uint32),
              ('MinimumWorkingSetSize', ctypes.c_void_p),
              ('MaximumWorkingSetSize', ctypes.c_void_p),
              ('ActiveProcessLimit', ctypes.c_uint32),
              ('Affinity', ctypes.c_void_p),
              ('PriorityClass', ctypes.c_uint32),
              ('SchedulingClass', ctypes.c_uint32)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
  _fields_ = [('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
              ('IoInfo', IO_COUNTERS),
              ('ProcessMemoryLimit', ctypes.c_void_p),
              ('JobMemoryLimit', ctypes.c_void_p),
              ('PeakProcessMemoryUsed', ctypes.c_void_p),
              ('PeakJobMemoryUsed', ctypes.c_void_p)]

## https://msdn.microsoft.com/en-us/library/windows/desktop/hh448384(v=vs.85).aspx
#class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
#  _fields_ = [('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
#              ('IoInfo', IO_COUNTERS),
#              ('ProcessMemoryLimit', ctypes.c_void_p),
#              ('JobMemoryLimit', ctypes.c_void_p),
#              ('PeakProcessMemoryUsed', ctypes.c_void_p),
#              ('PeakJobMemoryUsed', ctypes.c_void_p)]



# Set memory limit for process with specfied 'pid', to specified 'size' in bytes
def set_mem_commit_limit(pid, size, min_free_swap = None, retry_count = 10):

  for i in range(0, retry_count):   # assigning to job fails sometimes
    try:
      if (set_mem_commit_limit_worker(pid, size, min_free_swap)):
        break
    except Exception as msg:  # possibly already assigned to job object? (this can be done only once)
      print(msg)
      pass

#/ def set_mem_commit_limit(pid, size, min_free_swap, retry_count):


def set_mem_commit_limit_worker(pid, size, min_free_swap = None):


  if min_free_swap:
    swap_status = psutil.swap_memory()
    max_commit_limit = swap_status.free - min_free_swap
    size = min(size, max_commit_limit)  # NB!


  job_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
  out_size = ctypes.c_uint32()


  # child processes inherit the job object so it is not necessary to check for the parent


  # https://msdn.microsoft.com/en-us/library/windows/desktop/ms682409(v=vs.85).aspx
  # If the object existed before the function call, the function returns a handle to the existing job object and GetLastError returns ERROR_ALREADY_EXISTS.
  job = ctypes.windll.kernel32.CreateJobObjectW(None, None)   # NB! CreateJobObjectW not CreateJobObjectA
  # assert job != 0

  success = ctypes.windll.kernel32.QueryInformationJobObject(
    job,
    JobObjectExtendedLimitInformation,
    ctypes.POINTER(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)(job_info),
    ctypes.sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION),
    ctypes.POINTER(ctypes.c_uint32)(out_size)
  )
  # assert success


  #job_info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
  #job_info.ProcessMemoryLimit  = size

  job_info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
  job_info.JobMemoryLimit = size

  # job_info.MaximumWorkingSetSize = size


  success = ctypes.windll.kernel32.SetInformationJobObject(
    job,
    JobObjectExtendedLimitInformation,
    ctypes.POINTER(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)(job_info),
    ctypes.sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
  )
  # assert success

  process = ctypes.windll.kernel32.OpenProcess(
    PROCESS_SET_QUOTA | PROCESS_TERMINATE,
    False, pid
  )
  # assert process != 0

  success_assign = ctypes.windll.kernel32.AssignProcessToJobObject(job, process)
  # assert success

  # NB! comment-out: due to JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE flag do not close the job handle here
  # success = ctypes.windll.kernel32.CloseHandle(job)
  # # assert success

  success = ctypes.windll.kernel32.CloseHandle(process)
  # assert success

  return success_assign

#/ def set_mem_commit_limit(pid, size, min_free_swap):
