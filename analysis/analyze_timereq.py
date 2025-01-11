import glob
import re
import sys


def parse_time_to_minutes(time_str):
    """
    Convert time from 'real' format (e.g., '113m8.788s') to minutes as a float.
    """
    match = re.match(r'(\d+)m([\d.]+)s', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes + seconds / 60
    return None


def parse_file(file_path):
    """
    Parse the task name, real time, and number of requests from a file.
    """
    task_name = None
    real_time = None
    num_requests = None
    task_finished_successfully = False

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Executing TASK:"):
                task_name = line.split(":")[1].strip().replace(" ", "_")
            elif re.search(r'100%\|', line):
                match = re.search(r'(\d+)/(\d+)', line)
                if match:
                    num_requests = int(match.group(2))
            elif line.startswith("real"):
                real_time = parse_time_to_minutes(line.split()[1])
            elif re.search(r'\|\s+Tasks\s+\|', line):  # Match any amount of whitespace around "Tasks"
                task_finished_successfully = True

    if task_finished_successfully:
        return task_name, real_time, num_requests
    else:
        return task_name, None, None




def main(pattern):
    """
    Process all files matching the pattern, sort the results, and print them.
    """
    files = glob.glob(pattern)
    results = []

    for file_path in files:
        task_name, real_time, num_requests = parse_file(file_path)
        if task_name and real_time and num_requests is not None:
            results.append((task_name, real_time, num_requests))
        else:
            results.append((task_name, None, None))

    # Sort the results alphabetically by task_name
    results.sort(key=lambda x: x[0])

    # Print the results
    print(f"Task Name\tReal Time (minutes)\tNumber of Requests")
    for task_name, real_time, num_requests in results:
        if real_time and num_requests is not None:
            print(f"{task_name}\t{real_time:.2f}\t{num_requests}")
        else:
            print(f"{task_name}\tTask did not finish successfully")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_timereq.py '<file_pattern>'")
        sys.exit(1)

    pattern = sys.argv[1]
    main(pattern)
