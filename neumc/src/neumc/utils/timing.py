def format_time(s):
    isec = int(s)
    secs = isec % 60
    mins = isec // 60
    hours = mins // 60
    mins %= 60

    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def remaining_time(start, time, n, i):
    t = time - start
    dt = t / i
    return dt * (n - i)
