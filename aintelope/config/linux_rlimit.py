# -*- coding: utf-8 -*-

#
# Creative Commons Attribution Share Alike 4.0 International License
# Roland Pihlakas, 2018, roland@simplify.ee
#


# Set memory limits for current process, to specified 'size' in bytes
def set_mem_limits(data_size, address_space_size):


  import resource


  rlimits = [
    (resource.RLIMIT_DATA, (int(data_size), int(data_size))) if data_size else None,
    (resource.RLIMIT_AS, (int(address_space_size), int(address_space_size))) if address_space_size else None,   # The maximum area (in bytes) of address space which may be taken by the process.   # includes memory that is from shared libraries - https://stackoverflow.com/questions/7880784/what-is-rss-and-vsz-in-linux-memory-management

    # (resource.RLIMIT_RSS, (int(mem_limit), int(mem_limit))),   # The maximum resident set size that should be made available to the process.  # Ignored in Linux 2.4.30 and higher - https://superuser.com/questions/239796/limits-conf-to-set-memory-limits

    # (resource.RLIMIT_STACK, (int(8 * 1024 * 1024), int(8 * 1024 * 1024))),
    # (resource.RLIMIT_MEMLOCK, (int(64 * 1024), int(64 * 1024))),   # The maximum address space which may be locked in memory.  # 64kB is the default soft and hard limit
    # (resource.RLIMIT_MSGQUEUE, (0, 0)),   # The number of bytes that can be allocated for POSIX message queues.
    # (resource.RLIMIT_SIGPENDING, (0, 0)),
    # (resource.RLIMIT_NICE, (10, 10)),   # -20 is the highest priority and 19 is the lowest priority
    # (resource.RLIMIT_SBSIZE, (0, 0)),   # The maximum size (in bytes) of socket buffer usage for this user. This limits the amount of network memory, and hence the amount of mbufs, that this user may hold at any time.  # Availability: FreeBSD 9 or later.
    # (resource.RLIMIT_CPU, (cpu_timelimit, cpu_timelimit)),
    # (resource.RLIMIT_NPROC, (0, 0)),  # disable subprocess creation

    # (resource.RLIMIT_FSIZE, (0, 0)),   # disable file writing
    # (resource.RLIMIT_NOFILE, (16, 16)),   # The maximum number of open file descriptors for the current process.   # 1024 is the default soft limit, so lets make it a hard limit
    # (resource.RLIMIT_CORE, (0, 0)),  # The maximum size (in bytes) of a core file that the current process can create. 
  ]


  for rlimit in rlimits:
    if rlimit:  # not None
      try:        
        resource.setrlimit(rlimit[0], rlimit[1])
      except Exception as msg:
        print(msg)
        pass


#/ def set_mem_limits(data_size, address_space_size):
